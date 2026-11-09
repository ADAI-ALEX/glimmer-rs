//! End-to-end hyperparameter study on the CPU backend with a tiny dataset.

use glimmer_core::Workspace;
use glimmer_data::manifest::{DatasetKind, DatasetManifest, SplitStats};
use glimmer_data::split::SplitSpec;
use glimmer_data::tokens::write_tokens;
use glimmer_optimisation::study::{HalvingConfig, StudyConfig, StudyEvent, run_study};
use glimmer_optimisation::{ParamSpec, SamplerKind};
use glimmer_tokenizer::Tokenizer;
use glimmer_training::TrainControl;
use std::collections::BTreeMap;

fn setup() -> Workspace {
    let root = std::env::temp_dir().join(format!("glimmer-study-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("glimmer.toml"), "").unwrap();
    let ws = Workspace::open(root).unwrap();
    let tok = Tokenizer::bytes_only();
    tok.save(ws.tokenizers_dir().join("bytes.json")).unwrap();
    let ids: Vec<u16> =
        tok.encode(&"no worries mate, she'll be right. ".repeat(150)).into_iter().map(|t| t as u16).collect();
    let dir = ws.datasets_dir().join("tiny");
    write_tokens(&dir.join("train.bin"), &ids).unwrap();
    write_tokens(&dir.join("val.bin"), &ids[..1500]).unwrap();
    write_tokens(&dir.join("test.bin"), &ids[..1500]).unwrap();
    let mut splits = BTreeMap::new();
    splits.insert("train".into(), SplitStats::default());
    DatasetManifest {
        name: "tiny".into(),
        kind: DatasetKind::Pretrain,
        description: String::new(),
        tokenizer: "data/tokenizers/bytes.json".into(),
        tokenizer_sha256: String::new(),
        vocab_size: tok.vocab_size(),
        split: SplitSpec::default(),
        splits,
        sources: vec![],
        preprocessing: vec![],
        created_unix: 0,
        glimmer_version: String::new(),
    }
    .save(&dir)
    .unwrap();
    std::fs::create_dir_all(ws.configs_dir()).unwrap();
    std::fs::write(
        ws.configs_dir().join("base.toml"),
        r#"
        [run]
        name = "base"
        backend = "cpu"
        [model]
        preset = "micro"
        context_length = 32
        [data]
        dataset = "tiny"
        [optimizer]
        lr = 1e-3
        min_lr = 1e-4
        [schedule]
        warmup_steps = 4
        max_steps = 40
        [batch]
        batch_size = 4
        [eval]
        every = 40
        batches = 2
        "#,
    )
    .unwrap();
    ws
}

#[test]
fn successive_halving_promotes_and_resumes() {
    let ws = setup();
    let cfg = StudyConfig {
        name: "sh-test".into(),
        description: "test".into(),
        base_config: "configs/base.toml".into(),
        sampler: SamplerKind::Tpe,
        trials: 4,
        steps: 40,
        successive_halving: Some(HalvingConfig { eta: 2, min_steps: 10 }),
        seed: 5,
        backend: None,
        space: vec![ParamSpec::LogUniform { name: "lr".into(), low: 1e-4, high: 1e-2 }],
    };
    let ctl = TrainControl::new();
    let mut promotions = Vec::new();
    let summary = run_study(&ws, &cfg, &ctl, &mut |e| {
        if let StudyEvent::Promoted { rung, trials } = e {
            promotions.push((rung, trials));
        }
    })
    .unwrap();
    assert_eq!(summary.rungs, vec![10, 20, 40]);
    // 4 trials at rung 0, 2 at rung 1, 1 at rung 2.
    let per_rung = |r: usize| summary.records.iter().filter(|x| x.rung == r).count();
    assert_eq!((per_rung(0), per_rung(1), per_rung(2)), (4, 2, 1));
    assert_eq!(promotions.len(), 2);
    // Promoted trials resumed rather than restarted: the final rung trained to 40 steps total.
    let last = summary.records.iter().find(|r| r.rung == 2).unwrap();
    assert_eq!(last.budget, 40);
    assert_eq!(last.tokens, 40 * 4 * 32);
    assert!(summary.best_val_loss.is_some());
    assert!(summary.dir.join("report.md").is_file());
    assert!(summary.dir.join("trials.jsonl").is_file());
    std::fs::remove_dir_all(&ws.root).ok();
}
