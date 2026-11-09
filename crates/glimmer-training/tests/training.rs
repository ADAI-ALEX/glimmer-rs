//! Training-loop correctness: learning, resumption, SFT masking and accumulation.
//! Runs on the pure-Rust CPU backend so it works in CI.

use glimmer_core::Workspace;
use glimmer_core::config::*;
use glimmer_core::metrics::{MetricRecord, read_metrics};
use glimmer_data::manifest::{DatasetKind, DatasetManifest, SplitStats};
use glimmer_data::sft::{SftExample, write_examples};
use glimmer_data::split::SplitSpec;
use glimmer_data::tokens::write_tokens;
use glimmer_tokenizer::Tokenizer;
use glimmer_tokenizer::chat::{Message, render_with_mask};
use glimmer_training::{StopReason, TrainControl, TrainEvent, TrainOptions, train};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn temp_ws(name: &str) -> Workspace {
    let root = std::env::temp_dir().join(format!("glimmer-train-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("glimmer.toml"), "").unwrap();
    Workspace::open(root).unwrap()
}

fn manifest(name: &str, kind: DatasetKind, vocab: usize) -> DatasetManifest {
    let mut splits = BTreeMap::new();
    splits.insert("train".into(), SplitStats::default());
    DatasetManifest {
        name: name.into(),
        kind,
        description: "test".into(),
        tokenizer: "data/tokenizers/bytes.json".into(),
        tokenizer_sha256: String::new(),
        vocab_size: vocab,
        split: SplitSpec::default(),
        splits,
        sources: vec![],
        preprocessing: vec![],
        created_unix: 0,
        glimmer_version: "test".into(),
    }
}

/// A tiny pretraining corpus: a short phrase repeated (trivially learnable).
fn pretrain_ws(name: &str) -> (Workspace, Tokenizer) {
    let ws = temp_ws(name);
    let tok = Tokenizer::bytes_only();
    tok.save(ws.tokenizers_dir().join("bytes.json")).unwrap();
    let text = "g'day mate, how ya going? ".repeat(200);
    let ids: Vec<u16> = tok.encode(&text).into_iter().map(|t| t as u16).collect();
    let dir = ws.datasets_dir().join("tiny");
    write_tokens(&dir.join("train.bin"), &ids).unwrap();
    write_tokens(&dir.join("val.bin"), &ids[..2000]).unwrap();
    write_tokens(&dir.join("test.bin"), &ids[..2000]).unwrap();
    manifest("tiny", DatasetKind::Pretrain, tok.vocab_size()).save(&dir).unwrap();
    (ws, tok)
}

fn config(stage: Stage, dataset: &str, steps: usize) -> TrainingConfig {
    TrainingConfig {
        run: RunConfig { name: "test".into(), stage, seed: 42, backend: BackendKind::Cpu, export_as: None },
        model: ModelSpec { preset: Some("micro".into()), context_length: Some(32), ..Default::default() },
        data: DataConfig { dataset: dataset.into(), tokenizer: None },
        init: InitConfig::default(),
        optimizer: OptimizerConfig { lr: 3e-3, min_lr: 3e-4, weight_decay: 0.01, ..Default::default() },
        schedule: ScheduleConfig { kind: LrSchedule::Cosine, warmup_steps: 10, max_steps: steps, decay_fraction: 0.2 },
        batch: BatchConfig { batch_size: 8, grad_accum: 1 },
        eval: EvalConfig { every: 25, batches: 2, sample_prompts: false },
        checkpoint: CheckpointConfig { every: 0, keep_last: 1 },
        early_stopping: EarlyStoppingConfig::default(),
    }
}

fn step_losses(run_dir: &Path) -> Vec<(usize, f64)> {
    read_metrics(run_dir.join("metrics.jsonl"))
        .unwrap()
        .into_iter()
        .filter_map(|r| match r {
            MetricRecord::Step(s) => Some((s.step, s.loss)),
            _ => None,
        })
        .collect()
}

#[test]
fn pretraining_overfits_tiny_corpus() {
    let (ws, _) = pretrain_ws("overfit");
    let ctl = TrainControl::new();
    let mut evals = Vec::new();
    let summary = train(&ws, TrainOptions::new(config(Stage::Pretrain, "tiny", 150)), &ctl, &mut |e| {
        if let TrainEvent::Eval(m) = e {
            evals.push(m.val_loss);
        }
    })
    .unwrap();
    assert_eq!(summary.reason, StopReason::Completed);
    assert_eq!(summary.steps_completed, 150);
    let losses = step_losses(&summary.run_dir);
    let first = losses[0].1;
    let last = summary.final_train_loss.unwrap();
    assert!(first > 4.5, "initial loss {first} should be near ln(264)≈5.6");
    assert!(last < 0.6, "model failed to fit a repeated phrase: final loss {last}");
    assert!(evals.last().unwrap() < &0.6, "validation loss {evals:?}");
    assert!(summary.best_checkpoint.unwrap().join("model.bpk").is_file());
    std::fs::remove_dir_all(&ws.root).ok();
}

#[test]
fn resumed_training_matches_uninterrupted_run() {
    let (ws, _) = pretrain_ws("resume");
    let mut cfg = config(Stage::Pretrain, "tiny", 30);
    cfg.checkpoint.every = 15;
    cfg.checkpoint.keep_last = 5;

    // Uninterrupted reference run.
    let ctl = TrainControl::new();
    let full = train(&ws, TrainOptions::new(cfg.clone()), &ctl, &mut |_| {}).unwrap();
    let reference = step_losses(&full.run_dir);

    // Second run: stop after the step-15 checkpoint, then resume from it.
    let mut opts = TrainOptions::new(cfg.clone());
    let run_b: PathBuf = ws.runs_dir().join("resume-b");
    opts.run_dir = Some(run_b.clone());
    let ctl = TrainControl::new();
    let mut stopped = false;
    train(&ws, opts, &ctl, &mut |e| {
        if let TrainEvent::Step(s) = &e
            && s.step == 15
            && !stopped
        {
            stopped = true;
        }
        if let TrainEvent::Event(ev) = &e
            && ev.message.contains("step-0000015")
        {
            ctl.stop();
        }
    })
    .unwrap();
    let ckpt = run_b.join("checkpoints").join("step-0000015");
    assert!(ckpt.join("optimizer.bpk").is_file(), "periodic checkpoint missing");
    let mut resume = TrainOptions::new(cfg);
    resume.resume = Some(ckpt);
    let ctl = TrainControl::new();
    let resumed = train(&ws, resume, &ctl, &mut |_| {}).unwrap();
    assert_eq!(resumed.steps_completed, 30);

    let after: Vec<(usize, f64)> = step_losses(&run_b).into_iter().filter(|(s, _)| *s > 15).collect();
    let expected: Vec<(usize, f64)> = reference.into_iter().filter(|(s, _)| *s > 15).collect();
    assert_eq!(after.len(), expected.len());
    for ((s1, a), (s2, b)) in after.iter().zip(&expected) {
        assert_eq!(s1, s2);
        assert!(
            (a - b).abs() < 1e-4,
            "step {s1}: resumed loss {a} != uninterrupted {b} (optimizer/scheduler/RNG state not restored)"
        );
    }
    std::fs::remove_dir_all(&ws.root).ok();
}

#[test]
fn sft_learns_assistant_turns() {
    let ws = temp_ws("sft");
    let tok = Tokenizer::bytes_only();
    tok.save(ws.tokenizers_dir().join("bytes.json")).unwrap();
    let pairs = [("hi", "g'day!"), ("bye", "hooroo!"), ("tea?", "cuppa, ta"), ("food?", "snag!")];
    let mut examples = Vec::new();
    for _ in 0..8 {
        for (u, a) in pairs {
            let (tokens, mask) = render_with_mask(&tok, &[Message::user(u), Message::assistant(a)]);
            examples.push(SftExample { tokens, mask });
        }
    }
    let dir = ws.datasets_dir().join("chat");
    write_examples(&dir.join("train.sft"), &examples).unwrap();
    write_examples(&dir.join("val.sft"), &examples[..4]).unwrap();
    write_examples(&dir.join("test.sft"), &examples[..4]).unwrap();
    manifest("chat", DatasetKind::Sft, tok.vocab_size()).save(&dir).unwrap();

    let ctl = TrainControl::new();
    let summary = train(&ws, TrainOptions::new(config(Stage::Sft, "chat", 120)), &ctl, &mut |_| {}).unwrap();
    let losses = step_losses(&summary.run_dir);
    assert!(losses[0].1 > 4.0);
    assert!(
        summary.best_val_loss.unwrap() < 0.5,
        "SFT did not learn the assistant replies: {:?}",
        summary.best_val_loss
    );
    std::fs::remove_dir_all(&ws.root).ok();
}

#[test]
fn gradient_accumulation_matches_large_batch() {
    // batch 8 × accum 1 and batch 4 × accum 2 see the same tokens per step; with plain SGD
    // (no momentum) their first update and hence second-step loss must agree closely.
    let (ws, _) = pretrain_ws("accum");
    let mut a = config(Stage::Pretrain, "tiny", 2);
    a.optimizer.kind = OptimizerKind::Sgd;
    a.optimizer.momentum = 0.0;
    a.optimizer.weight_decay = 0.0;
    a.optimizer.grad_clip = 0.0;
    a.schedule.warmup_steps = 0;
    a.schedule.kind = LrSchedule::Constant;
    a.checkpoint.every = 0;
    let mut b = a.clone();
    b.batch.batch_size = 4;
    b.batch.grad_accum = 2;
    let mut oa = TrainOptions::new(a);
    oa.save_checkpoints = false;
    oa.run_dir = Some(ws.runs_dir().join("acc-a"));
    let mut ob = TrainOptions::new(b);
    ob.save_checkpoints = false;
    ob.run_dir = Some(ws.runs_dir().join("acc-b"));
    let ctl = TrainControl::new();
    train(&ws, oa, &ctl, &mut |_| {}).unwrap();
    train(&ws, ob, &ctl, &mut |_| {}).unwrap();
    let la = step_losses(&ws.runs_dir().join("acc-a"));
    let lb = step_losses(&ws.runs_dir().join("acc-b"));
    // Same RNG stream → the same 8 windows per step, split differently.
    for ((_, x), (_, y)) in la.iter().zip(&lb) {
        assert!((x - y).abs() < 1e-3, "accumulated loss {y} differs from full batch {x}");
    }
    std::fs::remove_dir_all(&ws.root).ok();
}
