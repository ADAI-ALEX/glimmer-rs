//! The training loop shared by pretraining and supervised fine-tuning.

use crate::checkpoint::{self, CheckpointData, CheckpointKind, CheckpointMeta, TrainerState};
use crate::events::{RunInfo, RunSummary, StopReason, TrainControl, TrainEvent};
use crate::monitor::Monitor;
use crate::optim;
use anyhow::{Context, Result, bail};
use glimmer_core::config::{ModelConfig, Stage, TrainingConfig};
use glimmer_core::metrics::{
    Ema, EvalMetrics, EventKind, EventRecord, MetricRecord, MetricsWriter, SampleRecord, StepMetrics,
};
use glimmer_core::registry::{ModelCard, StageRecord};
use glimmer_core::{Rng, Workspace};
use glimmer_data::batch::{Batch, PretrainBatcher, SftBatcher};
use glimmer_data::manifest::{DatasetKind, DatasetManifest};
use glimmer_data::sft::read_examples;
use glimmer_models::burn::module::Module;
use glimmer_models::burn::optim::{GradientsAccumulator, GradientsParams};
use glimmer_models::burn::tensor::{Device, Int, Tensor, TensorData};
use glimmer_models::loss::weighted_nll_sum;
use glimmer_models::{ComputeDevice, Transformer, backend, store};
use glimmer_tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Generates `(prompt, completion)` samples from the current model at evaluation time.
pub type Sampler = Box<dyn FnMut(&Transformer, &Tokenizer, &Device) -> Vec<(String, String)> + Send>;

pub struct TrainOptions {
    pub config: TrainingConfig,
    /// Resume from a checkpoint directory (continues that run's directory and metrics).
    pub resume: Option<PathBuf>,
    /// Explicit run directory (used for hyperparameter-search trials).
    pub run_dir: Option<PathBuf>,
    /// Pre-opened compute device (avoids re-initialising Metal for every search trial).
    pub device: Option<ComputeDevice>,
    pub sampler: Option<Sampler>,
    /// Write checkpoints. Disable for short search trials.
    pub save_checkpoints: bool,
    /// Pause the run once this many steps are complete: evaluate, save a resumable checkpoint
    /// named `rung`, and return. The schedule still follows `max_steps`, so a later resume
    /// continues exactly (used by successive halving).
    pub stop_after_step: Option<usize>,
}

impl TrainOptions {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            resume: None,
            run_dir: None,
            device: None,
            sampler: None,
            save_checkpoints: true,
            stop_after_step: None,
        }
    }
}

enum Data {
    Pretrain { train: PretrainBatcher, val: Vec<Batch> },
    Sft { train: SftBatcher, val: Vec<Batch> },
}

impl Data {
    fn next(&mut self, batch_size: usize, rng: &mut Rng) -> Batch {
        match self {
            Data::Pretrain { train, .. } => train.sample(batch_size, rng),
            Data::Sft { train, .. } => train.next_batch(batch_size, rng),
        }
    }
    fn val(&self) -> &[Batch] {
        match self {
            Data::Pretrain { val, .. } | Data::Sft { val, .. } => val,
        }
    }
}

struct DeviceBatch {
    inputs: Tensor<2, Int>,
    targets: Tensor<2, Int>,
    weights: Tensor<2>,
    packed: Option<(Tensor<2, Int>, Tensor<2, Int>)>,
}

fn ints(v: &[u32], shape: [usize; 2], device: &Device) -> Tensor<2, Int> {
    let data: Vec<i64> = v.iter().map(|&x| x as i64).collect();
    Tensor::from_data(TensorData::new(data, shape), device)
}

fn upload(b: &Batch, device: &Device) -> DeviceBatch {
    let shape = [b.batch_size, b.seq_len];
    DeviceBatch {
        inputs: ints(&b.inputs, shape, device),
        targets: ints(&b.targets, shape, device),
        weights: Tensor::from_data(TensorData::new(b.weights.clone(), shape), device),
        packed: match (&b.positions, &b.segments) {
            (Some(p), Some(s)) => Some((ints(p, shape, device), ints(s, shape, device))),
            _ => None,
        },
    }
}

fn forward(model: &Transformer, db: &DeviceBatch) -> Tensor<3> {
    match &db.packed {
        Some((pos, seg)) => model.forward_packed(db.inputs.clone(), pos.clone(), seg.clone()),
        None => model.forward(db.inputs.clone()),
    }
}

/// Mean validation loss (per supervised token) over fixed batches.
pub fn evaluate(model: &Transformer, batches: &[Batch], device: &Device) -> f64 {
    let m = model.valid();
    let mut sum = 0.0;
    let mut count = 0.0;
    for b in batches {
        let n = b.supervised_tokens();
        if n == 0 {
            continue;
        }
        let db = upload(b, device);
        let nll = weighted_nll_sum(forward(&m, &db), db.targets, db.weights);
        sum += nll.into_scalar::<f32>() as f64;
        count += n as f64;
    }
    if count == 0.0 { f64::NAN } else { sum / count }
}

/// Resolve `init.from_checkpoint`: a path to a checkpoint or model directory, or
/// `run:<name-prefix>` for the best checkpoint of the most recent run with that prefix.
pub fn resolve_init(ws: &Workspace, reference: &str) -> Result<PathBuf> {
    if let Some(prefix) = reference.strip_prefix("run:") {
        let mut runs: Vec<PathBuf> = std::fs::read_dir(ws.runs_dir())
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&format!("{prefix}-"))))
                    .collect()
            })
            .unwrap_or_default();
        runs.sort();
        for run in runs.iter().rev() {
            for name in ["best", "final"] {
                let p = run.join("checkpoints").join(name);
                if p.join("model.bpk").is_file() {
                    return Ok(p);
                }
            }
        }
        bail!("no completed run named '{prefix}-*' in {}", ws.relative(ws.runs_dir()));
    }
    let p = ws.resolve(reference);
    if p.join("model.bpk").is_file() && p.join("model.json").is_file() {
        return Ok(p);
    }
    let by_name = ws.models_dir().join(reference);
    if by_name.join("model.bpk").is_file() {
        return Ok(by_name);
    }
    bail!("cannot initialise from '{reference}': no model.bpk/model.json found")
}

fn read_lineage(dir: &Path) -> Vec<StageRecord> {
    if let Ok(t) = std::fs::read_to_string(dir.join("lineage.json"))
        && let Ok(v) = serde_json::from_str(&t)
    {
        return v;
    }
    if let Ok(t) = std::fs::read_to_string(dir.join("card.json"))
        && let Ok(card) = serde_json::from_str::<ModelCard>(&t)
    {
        return card.stages;
    }
    Vec::new()
}

struct RunCtx<'a> {
    ws: &'a Workspace,
    cfg: &'a TrainingConfig,
    model_cfg: &'a ModelConfig,
    run_id: &'a str,
    run_dir: &'a Path,
    tokenizer_path: &'a Path,
    backend_label: String,
    lineage: Vec<StageRecord>,
    dataset: String,
    initialised_from: Option<String>,
}

impl RunCtx<'_> {
    fn save(
        &self,
        name: &str,
        kind: CheckpointKind,
        model: &Transformer,
        optimizer: &glimmer_models::burn::optim::ModuleOptimizer,
        state: &TrainerState,
        val_loss: Option<f64>,
    ) -> Result<PathBuf> {
        let dir = self.run_dir.join("checkpoints").join(name);
        let meta = CheckpointMeta {
            run_id: self.run_id.to_string(),
            stage: self.cfg.run.stage,
            kind,
            step: state.step,
            tokens_seen: state.tokens_seen,
            train_loss: state.train_loss_ema.value,
            val_loss,
            backend: self.backend_label.clone(),
            created_unix: glimmer_core::time::now_unix(),
            glimmer_version: glimmer_core::VERSION.to_string(),
        };
        checkpoint::save(
            &dir,
            CheckpointData {
                model,
                optimizer,
                state,
                meta: &meta,
                training: self.cfg,
                model_config: self.model_cfg,
                tokenizer: self.tokenizer_path,
            },
        )?;
        std::fs::write(dir.join("lineage.json"), serde_json::to_string_pretty(&self.lineage)?)?;
        Ok(dir)
    }

    fn stage_record(&self, state: &TrainerState) -> StageRecord {
        StageRecord {
            stage: self.cfg.run.stage.as_str().to_string(),
            run_id: self.run_id.to_string(),
            dataset: self.dataset.clone(),
            steps: state.step,
            tokens: state.tokens_seen,
            final_train_loss: state.train_loss_ema.value,
            best_val_loss: state.best_val_loss,
            backend: self.backend_label.clone(),
            wall_secs: state.wall_secs,
            initialised_from: self.initialised_from.clone(),
        }
    }

    fn export(&self, name: &str, from: &Path, state: &TrainerState) -> Result<PathBuf> {
        let out = self.ws.models_dir().join(name);
        std::fs::create_dir_all(&out)?;
        for f in ["model.bpk", "model.json", "tokenizer.json"] {
            std::fs::copy(from.join(f), out.join(f)).with_context(|| format!("exporting {f}"))?;
        }
        let mut stages = self.lineage.clone();
        stages.push(self.stage_record(state));
        let hw = glimmer_core::hardware::HardwareReport::detect(false);
        let hardware = format!(
            "{} · {:.0} GiB{}",
            hw.cpu_brand.clone().unwrap_or_else(|| hw.arch.clone()),
            hw.total_memory_gib(),
            hw.gpu.as_ref().map(|g| format!(" · {} ({} GPU cores)", g.name, g.cores.unwrap_or(0))).unwrap_or_default()
        );
        let card = ModelCard {
            name: name.to_string(),
            description: if name == "stevie" {
                "STEVIE — a little Aussie AI mate, trained from scratch.".into()
            } else {
                format!("{name} — trained from scratch with GLIMMER.")
            },
            architecture: format!(
                "decoder-only transformer ({}): {} layers, d_model {}, {} heads, d_ff {}, {:?}, {:?}, RoPE",
                self.model_cfg.name,
                self.model_cfg.n_layers,
                self.model_cfg.d_model,
                self.model_cfg.n_heads,
                self.model_cfg.d_ff,
                self.model_cfg.norm,
                self.model_cfg.ffn
            ),
            parameters: self.model_cfg.param_count(),
            context_length: self.model_cfg.context_length,
            vocab_size: self.model_cfg.vocab_size,
            stages,
            system_prompt: (self.cfg.run.stage == Stage::Sft).then(|| glimmer_data::stevie::SYSTEM_PROMPT.to_string()),
            hardware: Some(hardware),
            created_unix: glimmer_core::time::now_unix(),
            glimmer_version: glimmer_core::VERSION.to_string(),
        };
        std::fs::write(out.join("card.json"), serde_json::to_string_pretty(&card)?)?;
        Ok(out)
    }
}

/// Run a training job to completion (or until stopped). Blocks the calling thread.
pub fn train(
    ws: &Workspace,
    mut opts: TrainOptions,
    control: &TrainControl,
    emit: &mut dyn FnMut(TrainEvent),
) -> Result<RunSummary> {
    let cfg = opts.config.clone();
    cfg.validate()?;

    // ── Data ────────────────────────────────────────────────────────────────────────────────
    let data_dir = glimmer_data::build::dataset_dir(ws, &cfg.data.dataset);
    let manifest = DatasetManifest::load(&data_dir).with_context(|| {
        format!("dataset '{}' is not prepared — run `glimmer data build {}`", cfg.data.dataset, cfg.data.dataset)
    })?;
    let expected_kind = match cfg.run.stage {
        Stage::Pretrain => DatasetKind::Pretrain,
        Stage::Sft => DatasetKind::Sft,
    };
    if manifest.kind != expected_kind {
        bail!(
            "stage '{}' needs a {:?} dataset but '{}' is {:?}",
            cfg.run.stage.as_str(),
            expected_kind,
            manifest.name,
            manifest.kind
        );
    }
    let tokenizer_path = ws.resolve(cfg.data.tokenizer.as_deref().unwrap_or(&manifest.tokenizer));
    let tokenizer = Tokenizer::load(&tokenizer_path)?;

    // ── Resume / initialisation ─────────────────────────────────────────────────────────────
    let resume_state = match &opts.resume {
        Some(dir) => Some(checkpoint::load_state(dir)?),
        None => None,
    };
    let init_dir = match (&opts.resume, &cfg.init.from_checkpoint) {
        (None, Some(r)) => Some(resolve_init(ws, r)?),
        _ => None,
    };
    let model_cfg = if let Some(dir) = &opts.resume {
        checkpoint::load_model_config(dir)?
    } else if let Some(dir) = &init_dir {
        // The architecture comes from the checkpoint; settings that do not change the weights
        // (dropout, context length — RoPE has no parameters) may still be overridden.
        let mut c = checkpoint::load_model_config(dir)?;
        if let Some(p) = cfg.model.dropout {
            c.dropout = p;
        }
        if let Some(n) = cfg.model.context_length {
            c.context_length = n;
        }
        c.validate()?;
        c
    } else {
        cfg.model.resolve(tokenizer.vocab_size())?
    };
    if model_cfg.vocab_size != tokenizer.vocab_size() {
        bail!("model vocabulary ({}) does not match tokenizer ({})", model_cfg.vocab_size, tokenizer.vocab_size());
    }

    // ── Device ──────────────────────────────────────────────────────────────────────────────
    let compute = match opts.device.take() {
        Some(d) => d,
        None => backend::select(cfg.run.backend)?,
    };
    backend::seed(&compute.device, cfg.run.seed);
    let eval_device = compute.device.clone();
    let train_device = compute.device.clone().autodiff();

    // ── Run directory ───────────────────────────────────────────────────────────────────────
    let (run_id, run_dir) = if let Some(dir) = &opts.resume {
        let run_dir =
            dir.parent().and_then(|p| p.parent()).context("checkpoint is not inside a run directory")?.to_path_buf();
        (run_dir.file_name().unwrap().to_string_lossy().into_owned(), run_dir)
    } else if let Some(dir) = &opts.run_dir {
        (dir.file_name().unwrap().to_string_lossy().into_owned(), dir.clone())
    } else {
        let id = format!("{}-{}", cfg.run.name, glimmer_core::time::compact_stamp(glimmer_core::time::now_unix()));
        (id.clone(), ws.runs_dir().join(id))
    };
    std::fs::create_dir_all(&run_dir)?;
    std::fs::write(run_dir.join("training.toml"), cfg.to_toml()?)?;
    std::fs::write(run_dir.join("model.json"), serde_json::to_string_pretty(&model_cfg)?)?;
    let mut metrics = MetricsWriter::append(run_dir.join("metrics.jsonl"))?;

    // ── Model and optimiser ─────────────────────────────────────────────────────────────────
    let mut model = if let Some(dir) = &opts.resume {
        store::load_weights(&model_cfg, &train_device, &dir.join("model.bpk"))?
    } else if let Some(dir) = &init_dir {
        store::load_weights(&model_cfg, &train_device, &dir.join("model.bpk"))?
    } else {
        Transformer::init(&model_cfg, &train_device, cfg.run.seed)
    };
    let mut optimizer = optim::build(&cfg.optimizer);
    if let Some(dir) = &opts.resume {
        optimizer =
            optimizer.load(dir.join("optimizer.bpk")).map_err(|e| anyhow::anyhow!("loading optimizer state: {e}"))?;
    }

    let mut state = match &resume_state {
        Some((s, _)) => s.clone(),
        None => TrainerState {
            step: 0,
            tokens_seen: 0,
            rng: Rng::new(cfg.run.seed ^ 0xD47A),
            sft_cursor: None,
            best_val_loss: None,
            best_step: None,
            evals_since_best: 0,
            train_loss_ema: Ema::new(0.05),
            wall_secs: 0.0,
            skipped_steps: 0,
        },
    };

    let ctx_len = model_cfg.context_length;
    let mut data = match cfg.run.stage {
        Stage::Pretrain => {
            let train = PretrainBatcher::open(&data_dir.join("train.bin"), ctx_len)?;
            let val = PretrainBatcher::open(&data_dir.join("val.bin"), ctx_len)?
                .eval_batches(cfg.batch.batch_size, cfg.eval.batches);
            Data::Pretrain { train, val }
        }
        Stage::Sft => {
            let mut init_rng = Rng::new(cfg.run.seed ^ 0x5F7);
            let mut train = SftBatcher::new(read_examples(&data_dir.join("train.sft"))?, ctx_len, &mut init_rng)?;
            if let Some(c) = &state.sft_cursor {
                train.restore(c.clone())?;
            }
            let val_examples = read_examples(&data_dir.join("val.sft"))?;
            let val_batcher = SftBatcher::new(val_examples, ctx_len, &mut Rng::new(0))?;
            let mut val = val_batcher.eval_batches(cfg.batch.batch_size);
            val.truncate(cfg.eval.batches.max(1));
            Data::Sft { train, val }
        }
    };

    let lineage = match (&opts.resume, &init_dir) {
        (Some(dir), _) => read_lineage(dir),
        (None, Some(dir)) => read_lineage(dir),
        _ => Vec::new(),
    };
    let ctx = RunCtx {
        ws,
        cfg: &cfg,
        model_cfg: &model_cfg,
        run_id: &run_id,
        run_dir: &run_dir,
        tokenizer_path: &tokenizer_path,
        backend_label: compute.label(),
        lineage,
        dataset: cfg.data.dataset.clone(),
        initialised_from: init_dir.as_ref().map(|d| ws.relative(d)),
    };

    let tokens_per_step = cfg.tokens_per_step(ctx_len);
    let max_steps = cfg.schedule.max_steps;
    let record = |metrics: &mut MetricsWriter, emit: &mut dyn FnMut(TrainEvent), r: MetricRecord| -> Result<()> {
        metrics.write(&r)?;
        emit(match r {
            MetricRecord::Step(s) => TrainEvent::Step(s),
            MetricRecord::Eval(e) => TrainEvent::Eval(e),
            MetricRecord::Sample(s) => TrainEvent::Sample(s),
            MetricRecord::Event(e) => TrainEvent::Event(e),
        });
        Ok(())
    };
    let event = |step: usize, kind: EventKind, message: String| {
        MetricRecord::Event(EventRecord { step, kind, message, unix: glimmer_core::time::now_unix() })
    };

    let run_info = RunInfo {
        run_id: run_id.clone(),
        run_dir: run_dir.clone(),
        stage: cfg.run.stage,
        model: model_cfg.clone(),
        parameters: model.num_params(),
        backend: compute.kind,
        backend_label: compute.label(),
        fallback_reason: compute.fallback_reason.clone(),
        dataset: cfg.data.dataset.clone(),
        train_tokens: manifest.splits.get("train").map(|s| s.tokens).unwrap_or(0),
        start_step: state.step,
        max_steps,
        tokens_per_step,
        resumed_from: opts.resume.as_ref().map(|d| ws.relative(d)),
        initialised_from: ctx.initialised_from.clone(),
        config: cfg.clone(),
    };
    // Written so other processes (e.g. the terminal UI's watch mode) can follow this run.
    std::fs::write(run_dir.join("run_info.json"), serde_json::to_string_pretty(&run_info)?)?;
    emit(TrainEvent::Started(Box::new(run_info)));
    let origin = match (&opts.resume, &init_dir) {
        (Some(d), _) => format!("resumed from {} at step {}", ws.relative(d), state.step),
        (None, Some(d)) => format!("initialised from {}", ws.relative(d)),
        _ => format!("random initialisation (seed {})", cfg.run.seed),
    };
    record(
        &mut metrics,
        emit,
        event(
            state.step,
            EventKind::Info,
            format!(
                "{} · {} · {} params · {}",
                cfg.run.stage.as_str(),
                compute.label(),
                glimmer_core::time::format_count(model.num_params() as u64),
                origin
            ),
        ),
    )?;
    if let Some(reason) = &compute.fallback_reason {
        record(
            &mut metrics,
            emit,
            event(state.step, EventKind::Warning, format!("Metal unavailable, using CPU: {reason}")),
        )?;
    }

    let mut monitor = Monitor::new(cfg.schedule.warmup_steps);
    let wall_start = Instant::now() - Duration::from_secs_f64(state.wall_secs);
    let mut reason = StopReason::Completed;
    let mut best_checkpoint = run_dir.join("checkpoints").join("best");
    let mut last_checkpoint: Option<PathBuf> = None;
    let mut consecutive_bad = 0usize;
    let mut last_loss: Option<f64> = None;
    let mut last_val_loss: Option<f64> = None;

    while state.step < max_steps {
        // ── Controls ────────────────────────────────────────────────────────────────────────
        if control.is_paused() && !control.should_stop() {
            emit(TrainEvent::Paused(true));
            let paused_at = Instant::now();
            while control.is_paused() && !control.should_stop() {
                std::thread::sleep(Duration::from_millis(50));
                if control.take_checkpoint_request() && opts.save_checkpoints {
                    let p = ctx.save("manual", CheckpointKind::Manual, &model, &optimizer, &state, None)?;
                    record(
                        &mut metrics,
                        emit,
                        event(state.step, EventKind::Checkpoint, format!("Saved {}", ws.relative(&p))),
                    )?;
                }
            }
            // Paused time does not count towards wall-clock training time.
            let _ = paused_at;
            emit(TrainEvent::Paused(false));
        }
        if control.should_stop() {
            reason = StopReason::Cancelled;
            break;
        }

        // ── One optimiser step ──────────────────────────────────────────────────────────────
        let t0 = Instant::now();
        let step = state.step;
        let lr = cfg.schedule.lr_at(step, cfg.optimizer.lr, cfg.optimizer.min_lr);
        let batches: Vec<Batch> =
            (0..cfg.batch.grad_accum).map(|_| data.next(cfg.batch.batch_size, &mut state.rng)).collect();
        let supervised: usize = batches.iter().map(|b| b.supervised_tokens()).sum();
        let processed: usize = batches.iter().map(|b| b.tokens()).sum();
        if supervised == 0 {
            state.step += 1;
            continue;
        }
        let mut accumulator = GradientsAccumulator::<Transformer>::new();
        let mut loss_acc: Option<Tensor<1>> = None;
        for b in &batches {
            let db = upload(b, &train_device);
            let nll = weighted_nll_sum(forward(&model, &db), db.targets.clone(), db.weights.clone())
                .div_scalar(supervised as f64);
            let grads = nll.backward();
            let detached = nll.detach();
            loss_acc = Some(match loss_acc.take() {
                Some(a) => a + detached,
                None => detached,
            });
            accumulator.accumulate(&model, GradientsParams::from_grads(grads, &model));
        }
        let mut grads = accumulator.grads();
        let loss = loss_acc.map(|t| t.into_scalar::<f32>() as f64).unwrap_or(f64::NAN);
        let (grad_norm, clipped) = optim::clip_global_norm(&model, &mut grads, cfg.optimizer.grad_clip);

        if !loss.is_finite() || !grad_norm.is_finite() {
            consecutive_bad += 1;
            state.skipped_steps += 1;
            state.step += 1;
            record(
                &mut metrics,
                emit,
                event(
                    step,
                    EventKind::AutoAction,
                    format!("Skipped step {step}: non-finite loss ({loss}) or gradient norm ({grad_norm})"),
                ),
            )?;
            if consecutive_bad >= 5 {
                record(
                    &mut metrics,
                    emit,
                    event(step, EventKind::Error, "Training diverged: 5 consecutive non-finite steps".into()),
                )?;
                reason = StopReason::Diverged;
                break;
            }
            continue;
        }
        consecutive_bad = 0;
        model = optimizer.step(lr, model, grads);
        state.step += 1;
        state.tokens_seen += processed as u64;
        let smoothed = state.train_loss_ema.update(loss);
        last_loss = Some(loss);
        let dt = t0.elapsed().as_secs_f64();
        state.wall_secs = wall_start.elapsed().as_secs_f64();
        let sm = StepMetrics {
            step: state.step,
            tokens_seen: state.tokens_seen,
            loss,
            lr,
            grad_norm,
            clipped,
            tokens_per_sec: processed as f64 / dt,
            step_time_ms: dt * 1e3,
            elapsed_secs: state.wall_secs,
        };
        for a in monitor.observe_step(&sm) {
            record(&mut metrics, emit, event(state.step, EventKind::Advice, a.message))?;
        }
        record(&mut metrics, emit, MetricRecord::Step(sm))?;
        if let Data::Sft { train, .. } = &data {
            state.sft_cursor = Some(train.cursor().clone());
        }

        // ── Evaluation ──────────────────────────────────────────────────────────────────────
        let is_last = state.step == max_steps;
        let at_rung = opts.stop_after_step == Some(state.step);
        if state.step % cfg.eval.every == 0 || is_last || at_rung {
            let val_loss = evaluate(&model, data.val(), &eval_device);
            last_val_loss = Some(val_loss);
            let improved =
                val_loss.is_finite() && state.best_val_loss.is_none_or(|b| val_loss < b - cfg.early_stopping.min_delta);
            if improved {
                state.best_val_loss = Some(val_loss);
                state.best_step = Some(state.step);
                state.evals_since_best = 0;
            } else {
                state.evals_since_best += 1;
            }
            let em = EvalMetrics {
                step: state.step,
                val_loss,
                val_ppl: val_loss.exp(),
                train_loss: smoothed,
                is_best: improved,
                elapsed_secs: state.wall_secs,
            };
            for a in monitor.observe_eval(&em) {
                record(&mut metrics, emit, event(state.step, EventKind::Advice, a.message))?;
            }
            record(&mut metrics, emit, MetricRecord::Eval(em))?;
            if improved && opts.save_checkpoints {
                best_checkpoint = ctx.save("best", CheckpointKind::Best, &model, &optimizer, &state, Some(val_loss))?;
                record(
                    &mut metrics,
                    emit,
                    event(
                        state.step,
                        EventKind::Checkpoint,
                        format!("New best validation loss {val_loss:.4} → checkpoints/best"),
                    ),
                )?;
            }
            if let Some(sampler) = opts.sampler.as_mut() {
                let valid = model.valid();
                for (prompt, completion) in sampler(&valid, &tokenizer, &eval_device) {
                    record(
                        &mut metrics,
                        emit,
                        MetricRecord::Sample(SampleRecord { step: state.step, prompt, completion }),
                    )?;
                }
            }
            if cfg.early_stopping.patience > 0 && state.evals_since_best >= cfg.early_stopping.patience {
                record(
                    &mut metrics,
                    emit,
                    event(
                        state.step,
                        EventKind::AutoAction,
                        format!(
                            "Early stopping: no validation improvement > {} for {} evaluations (best {:.4} at step {})",
                            cfg.early_stopping.min_delta,
                            state.evals_since_best,
                            state.best_val_loss.unwrap_or(f64::NAN),
                            state.best_step.unwrap_or(0)
                        ),
                    ),
                )?;
                reason = StopReason::EarlyStopped;
                break;
            }
        }

        if at_rung && !is_last {
            reason = StopReason::Paused;
            break;
        }

        // ── Checkpoints ─────────────────────────────────────────────────────────────────────
        let periodic = cfg.checkpoint.every > 0 && state.step % cfg.checkpoint.every == 0 && !is_last;
        if opts.save_checkpoints && (periodic || control.take_checkpoint_request()) {
            let (name, kind) = if periodic {
                (format!("step-{:07}", state.step), CheckpointKind::Periodic)
            } else {
                ("manual".to_string(), CheckpointKind::Manual)
            };
            let p = ctx.save(&name, kind, &model, &optimizer, &state, None)?;
            if periodic {
                checkpoint::prune(&run_dir, cfg.checkpoint.keep_last)?;
            }
            last_checkpoint = Some(p.clone());
            record(&mut metrics, emit, event(state.step, EventKind::Checkpoint, format!("Saved {}", ws.relative(&p))))?;
        }
    }

    state.wall_secs = wall_start.elapsed().as_secs_f64();
    if opts.save_checkpoints && state.step > 0 {
        let (name, kind) = match reason {
            StopReason::Cancelled => ("interrupted", CheckpointKind::Interrupted),
            StopReason::Paused => ("rung", CheckpointKind::Interrupted),
            _ => ("final", CheckpointKind::Final),
        };
        let p = ctx.save(name, kind, &model, &optimizer, &state, None)?;
        record(&mut metrics, emit, event(state.step, EventKind::Checkpoint, format!("Saved {}", ws.relative(&p))))?;
        last_checkpoint = Some(p);
    }

    let have_best = best_checkpoint.join("model.bpk").is_file();
    let exported = match (&cfg.run.export_as, opts.save_checkpoints) {
        (Some(name), true) if !matches!(reason, StopReason::Diverged | StopReason::Paused) => {
            let from = if have_best {
                best_checkpoint.clone()
            } else {
                last_checkpoint.clone().context("no checkpoint to export")?
            };
            let out = ctx.export(name, &from, &state)?;
            record(
                &mut metrics,
                emit,
                event(state.step, EventKind::Info, format!("Exported model '{name}' → {}", ws.relative(&out))),
            )?;
            Some(out)
        }
        _ => None,
    };

    let summary = RunSummary {
        run_id: run_id.clone(),
        run_dir: run_dir.clone(),
        stage: cfg.run.stage,
        reason,
        steps_completed: state.step,
        tokens_seen: state.tokens_seen,
        final_train_loss: state.train_loss_ema.value.or(last_loss),
        best_val_loss: state.best_val_loss,
        best_step: state.best_step,
        last_val_loss,
        best_checkpoint: have_best.then_some(best_checkpoint),
        last_checkpoint,
        wall_secs: state.wall_secs,
        exported_model: exported,
    };
    std::fs::write(run_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    emit(TrainEvent::Finished(Box::new(summary.clone())));
    Ok(summary)
}
