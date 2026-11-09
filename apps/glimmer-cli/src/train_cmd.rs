//! `glimmer train` (headless) and `glimmer runs`.

use crate::style::{self, bold, dim, heading, kv};
use anyhow::{Context, Result, bail};
use clap::Args;
use glimmer_core::config::TrainingConfig;
use glimmer_core::metrics::{EventKind, MetricRecord, read_metrics};
use glimmer_core::time::{format_count, format_duration};
use glimmer_core::{BackendKind, Workspace};
use glimmer_training::{RunSummary, TrainControl, TrainEvent, TrainOptions, checkpoint, train};
use serde::Deserialize;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Training config file (e.g. configs/training/stevie-pretrain.toml).
    #[arg(long, short)]
    pub config: Option<String>,
    /// Model recipe name (configs/models/<name>.toml): runs every stage in order.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Resume from a checkpoint directory or a run directory (latest checkpoint).
    #[arg(long)]
    pub resume: Option<String>,
    /// Override schedule.max_steps.
    #[arg(long)]
    pub max_steps: Option<usize>,
    /// Override the compute backend.
    #[arg(long)]
    pub backend: Option<BackendKind>,
    /// Override the export name.
    #[arg(long)]
    pub export_as: Option<String>,
    /// Start a recipe at this stage index (0-based), e.g. to run only SFT.
    #[arg(long, default_value_t = 0)]
    pub from_stage: usize,
}

/// `configs/models/<name>.toml`: a named model built by a sequence of training stages.
#[derive(Debug, Deserialize)]
pub struct Recipe {
    pub name: String,
    pub description: String,
    pub stages: Vec<String>,
}

fn install_ctrlc(control: Arc<TrainControl>) {
    let _ = ctrlc::set_handler(move || {
        if control.should_stop() {
            eprintln!("\nForce quit.");
            std::process::exit(130);
        }
        eprintln!("\n{} finishing the current step and saving a checkpoint (Ctrl-C again to force)…", style::warn("stopping:"));
        control.stop();
    });
}

pub fn print_event(e: &TrainEvent, max_steps: &mut usize) {
    match e {
        TrainEvent::Started(info) => {
            *max_steps = info.max_steps;
            heading(&format!("training {} — {}", info.stage.as_str(), info.run_id));
            kv("backend", &info.backend_label);
            if let Some(r) = &info.fallback_reason {
                kv("fallback", style::warn(r));
            }
            kv(
                "model",
                format!(
                    "{} · {} params · {} layers · d_model {} · ctx {}",
                    info.model.name,
                    format_count(info.parameters as u64),
                    info.model.n_layers,
                    info.model.d_model,
                    info.model.context_length
                ),
            );
            kv("dataset", format!("{} ({} training tokens)", info.dataset, format_count(info.train_tokens)));
            kv(
                "schedule",
                format!(
                    "steps {}→{} · {} tokens/step · peak lr {:.1e}",
                    info.start_step, info.max_steps, info.tokens_per_step, info.config.optimizer.lr
                ),
            );
            kv("run directory", info.run_dir.display());
            println!();
        }
        TrainEvent::Step(s) => {
            let remaining = max_steps.saturating_sub(s.step) as f64 * s.step_time_ms / 1e3;
            print!(
                "\r  {} {:>6}/{:<6} loss {:>7.4}  lr {:>8.2e}  |g| {:>6.3}{}  {:>7.0} tok/s  {:>6.0} ms  eta {:<8}",
                style::accent("▸"),
                s.step,
                max_steps,
                s.loss,
                s.lr,
                s.grad_norm,
                if s.clipped { "✂" } else { " " },
                s.tokens_per_sec,
                s.step_time_ms,
                format_duration(remaining)
            );
            let _ = std::io::stdout().flush();
        }
        TrainEvent::Eval(m) => {
            println!(
                "\n  {} step {:>6}  val loss {}  ppl {:>8.2}  train {:>7.4}{}",
                style::info("◆"),
                m.step,
                bold(&format!("{:.4}", m.val_loss)),
                m.val_ppl,
                m.train_loss,
                if m.is_best { style::ok("  ★ best") } else { String::new() }
            );
        }
        TrainEvent::Sample(s) => {
            println!("    {} {}", dim("prompt:"), s.prompt.replace('\n', " ⏎ "));
            println!("    {} {}", style::gum("stevie:"), s.completion.replace('\n', " ⏎ "));
        }
        TrainEvent::Event(ev) => {
            let tag = match ev.kind {
                EventKind::Checkpoint => style::dim("checkpoint"),
                EventKind::Advice => style::warn("advice"),
                EventKind::AutoAction => style::info("auto"),
                EventKind::Warning => style::warn("warning"),
                EventKind::Error => style::err("error"),
                EventKind::Info => style::dim("info"),
            };
            println!("\n  {tag} {}", ev.message);
        }
        TrainEvent::Paused(p) => println!("\n  {}", if *p { "paused" } else { "resumed" }),
        TrainEvent::Finished(s) => print_summary(s),
    }
}

fn print_summary(s: &RunSummary) {
    heading(&format!("finished {} ({:?})", s.run_id, s.reason));
    kv("steps", s.steps_completed);
    kv("tokens", format_count(s.tokens_seen));
    kv("wall time", format_duration(s.wall_secs));
    if let Some(l) = s.final_train_loss {
        kv("final train loss", format!("{l:.4}"));
    }
    if let (Some(v), Some(st)) = (s.best_val_loss, s.best_step) {
        kv("best val loss", format!("{v:.4} (ppl {:.2}) at step {st}", v.exp()));
    }
    if let Some(p) = &s.best_checkpoint {
        kv("best checkpoint", p.display());
    }
    if let Some(p) = &s.exported_model {
        kv("exported model", style::ok(&p.display().to_string()));
    }
}

fn resolve_resume(ws: &Workspace, r: &str) -> Result<PathBuf> {
    let p = ws.resolve(r);
    if p.join("state.json").is_file() {
        return Ok(p);
    }
    if p.join("checkpoints").is_dir() {
        return checkpoint::latest(&p).with_context(|| format!("no resumable checkpoint in {}", p.display()));
    }
    let run = ws.runs_dir().join(r);
    if run.join("checkpoints").is_dir() {
        return checkpoint::latest(&run).with_context(|| format!("no resumable checkpoint in {}", run.display()));
    }
    bail!("'{r}' is neither a checkpoint nor a run directory")
}

pub fn run_one(ws: &Workspace, mut cfg: TrainingConfig, args: &TrainArgs, resume: Option<PathBuf>) -> Result<RunSummary> {
    if let Some(n) = args.max_steps {
        cfg.schedule.max_steps = n;
    }
    if let Some(b) = args.backend {
        cfg.run.backend = b;
    }
    if let Some(e) = &args.export_as {
        cfg.run.export_as = Some(e.clone());
    }
    let control = Arc::new(TrainControl::new());
    install_ctrlc(control.clone());
    let mut opts = TrainOptions::new(cfg.clone());
    opts.resume = resume;
    if cfg.eval.sample_prompts {
        opts.sampler = Some(crate::eval_cmd::training_sampler(ws, cfg.run.stage)?);
    }
    let mut max_steps = cfg.schedule.max_steps;
    let summary = train(ws, opts, &control, &mut |e| print_event(&e, &mut max_steps))?;
    Ok(summary)
}

pub fn run(ws: &Workspace, args: TrainArgs) -> Result<()> {
    if let Some(r) = &args.resume {
        let ckpt = resolve_resume(ws, r)?;
        let cfg = TrainingConfig::load(ckpt.join("training.toml"))?;
        println!("{} {}", dim("resuming from"), ws.relative(&ckpt));
        run_one(ws, cfg, &args, Some(ckpt))?;
        return Ok(());
    }
    match (&args.config, &args.model) {
        (Some(c), _) => {
            let path = ws.resolve(c);
            let cfg = TrainingConfig::load(&path)?;
            run_one(ws, cfg, &args, None)?;
        }
        (None, Some(m)) => {
            let path = ws.configs_dir().join("models").join(format!("{m}.toml"));
            let recipe: Recipe = toml::from_str(
                &std::fs::read_to_string(&path).with_context(|| format!("model recipe {} not found", path.display()))?,
            )?;
            heading(&format!("{} — {}", recipe.name, recipe.description));
            for (i, stage) in recipe.stages.iter().enumerate().skip(args.from_stage) {
                println!("  {} stage {}/{}: {}", style::accent("●"), i + 1, recipe.stages.len(), stage);
                let cfg = TrainingConfig::load(ws.resolve(stage))?;
                let s = run_one(ws, cfg, &args, None)?;
                if s.reason == glimmer_training::StopReason::Cancelled || s.reason == glimmer_training::StopReason::Diverged {
                    bail!("stage {} ended with {:?}; not continuing", i + 1, s.reason);
                }
            }
        }
        (None, None) => bail!("pass --config <file>, --model <recipe> or --resume <checkpoint>"),
    }
    Ok(())
}

pub fn list_runs(ws: &Workspace) -> Result<()> {
    heading("training runs");
    let mut runs: Vec<PathBuf> = std::fs::read_dir(ws.runs_dir())
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.join("metrics.jsonl").is_file()).collect())
        .unwrap_or_default();
    runs.sort();
    if runs.is_empty() {
        println!("  {}", dim("no runs yet"));
    }
    for r in runs {
        let name = r.file_name().unwrap().to_string_lossy().into_owned();
        let recs = read_metrics(r.join("metrics.jsonl")).unwrap_or_default();
        let last_step = recs.iter().rev().find_map(|m| match m {
            MetricRecord::Step(s) => Some(s.clone()),
            _ => None,
        });
        let best = recs
            .iter()
            .filter_map(|m| match m {
                MetricRecord::Eval(e) => Some(e.val_loss),
                _ => None,
            })
            .fold(f64::INFINITY, f64::min);
        let status = if r.join("summary.json").is_file() { style::ok("done") } else { style::warn("partial") };
        match last_step {
            Some(s) => println!(
                "  {:<40} {:<8} step {:>6}  loss {:>7.4}  best val {:>7}  {}",
                bold(&name),
                status,
                s.step,
                s.loss,
                if best.is_finite() { format!("{best:.4}") } else { "—".into() },
                dim(&format_duration(s.elapsed_secs))
            ),
            None => println!("  {:<40} {}", bold(&name), dim("no steps")),
        }
    }
    Ok(())
}
