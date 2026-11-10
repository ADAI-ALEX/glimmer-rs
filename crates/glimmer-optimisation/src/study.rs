//! Hyperparameter studies: sequential search with optional successive halving.
//!
//! **Successive halving** (Jamieson & Talwalkar, 2016; the building block of Hyperband):
//! many configurations are trained for a small budget, the best `1/η` are promoted to an `η`×
//! larger budget, and so on. Every trial is configured with the *full* step budget so its
//! learning-rate schedule is identical regardless of where it stops; at each rung it is paused
//! with a resumable checkpoint, and promoted trials continue exactly where they left off.
//!
//! Everything is recorded: the study config, every trial's sampled parameters and full training
//! config (in its run directory), metrics, and per-rung results in `trials.jsonl`.

use crate::sampler::{Observation, SamplerKind, TpeConfig, propose};
use crate::space::{self, ParamSpec};
use anyhow::{Context, Result, bail};
use glimmer_core::config::TrainingConfig;
use glimmer_core::{BackendKind, Rng, Workspace};
use glimmer_models::{ComputeDevice, backend};
use glimmer_training::{StopReason, TrainControl, TrainEvent, TrainOptions, train};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Which validation loss scores a trial.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Validation loss at the end of the trial's budget (or rung).
    #[default]
    Last,
    /// Best validation loss seen so far — matches runs that keep their best checkpoint
    /// (early stopping), which is how fine-tuned models are exported.
    Best,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HalvingConfig {
    /// Promotion factor: keep the best `1/eta` at each rung.
    pub eta: usize,
    /// Budget (steps) of the first rung.
    pub min_steps: usize,
}

/// `configs/tuning/<name>.toml`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StudyConfig {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Training config each trial starts from.
    pub base_config: String,
    pub sampler: SamplerKind,
    /// Number of configurations to try (first-rung trials when halving).
    pub trials: usize,
    /// Full training budget in optimiser steps.
    pub steps: usize,
    #[serde(default)]
    pub successive_halving: Option<HalvingConfig>,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub backend: Option<BackendKind>,
    #[serde(default)]
    pub objective: Objective,
    /// Evaluate every N steps inside trials (default: only at rung boundaries and the end).
    #[serde(default)]
    pub eval_every: Option<usize>,
    pub space: Vec<ParamSpec>,
}

fn default_seed() -> u64 {
    2026
}

impl StudyConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let cfg: Self =
            toml::from_str(&std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?)
                .with_context(|| format!("parsing {}", p.display()))?;
        space::validate(&cfg.space)?;
        if cfg.trials == 0 || cfg.steps == 0 {
            bail!("trials and steps must be positive");
        }
        Ok(cfg)
    }

    /// Step budgets of each rung, ending at the full budget.
    pub fn rungs(&self) -> Vec<usize> {
        match &self.successive_halving {
            None => vec![self.steps],
            Some(h) => {
                // Work down from the full budget so the last rung is exactly `steps`.
                let eta = h.eta.max(2);
                let mut v = vec![self.steps];
                let mut b = self.steps;
                loop {
                    b /= eta;
                    if b < h.min_steps.max(1) {
                        break;
                    }
                    v.push(b);
                }
                v.reverse();
                v
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrialRecord {
    pub trial: usize,
    pub rung: usize,
    pub budget: usize,
    pub params: BTreeMap<String, f64>,
    pub val_loss: Option<f64>,
    pub train_loss: Option<f64>,
    pub tokens: u64,
    pub wall_secs: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudySummary {
    pub name: String,
    pub dir: PathBuf,
    pub best_trial: Option<usize>,
    pub best_params: BTreeMap<String, f64>,
    pub best_val_loss: Option<f64>,
    pub records: Vec<TrialRecord>,
    pub rungs: Vec<usize>,
    pub wall_secs: f64,
}

#[derive(Debug, Clone)]
pub enum StudyEvent {
    TrialStarted { trial: usize, rung: usize, budget: usize, params: BTreeMap<String, f64> },
    Train { trial: usize, event: TrainEvent },
    TrialFinished(TrialRecord),
    Promoted { rung: usize, trials: Vec<usize> },
    Finished(Box<StudySummary>),
}

pub fn study_dir(ws: &Workspace, name: &str) -> PathBuf {
    ws.experiments_dir().join(name)
}

fn trial_dir(dir: &Path, trial: usize) -> PathBuf {
    dir.join("trials").join(format!("t{trial:03}"))
}

fn prune_checkpoints(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir.join("checkpoints"));
}

/// Run a study to completion.
pub fn run_study(
    ws: &Workspace,
    cfg: &StudyConfig,
    control: &TrainControl,
    emit: &mut dyn FnMut(StudyEvent),
) -> Result<StudySummary> {
    let base = TrainingConfig::load(ws.resolve(&cfg.base_config))?;
    let dir = study_dir(ws, &cfg.name);
    if dir.join("trials.jsonl").exists() {
        bail!("study '{}' already exists in {} — choose a new name", cfg.name, ws.relative(&dir));
    }
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("study.toml"), toml::to_string_pretty(cfg)?)?;
    let mut log = std::io::BufWriter::new(std::fs::File::create(dir.join("trials.jsonl"))?);
    let device: ComputeDevice = backend::select(cfg.backend.unwrap_or(base.run.backend))?;
    let rungs = cfg.rungs();
    let mut rng = Rng::new(cfg.seed);
    let started = std::time::Instant::now();
    let mut records: Vec<TrialRecord> = Vec::new();
    let mut history: Vec<Observation> = Vec::new();
    let mut params_of: BTreeMap<usize, BTreeMap<String, f64>> = BTreeMap::new();

    let run_trial = |trial: usize,
                     rung: usize,
                     params: &BTreeMap<String, f64>,
                     records: &mut Vec<TrialRecord>,
                     emit: &mut dyn FnMut(StudyEvent)|
     -> Result<TrialRecord> {
        let budget = rungs[rung];
        let mut tc = space::apply(&base, params);
        tc.run.name = format!("{}-t{trial:03}", cfg.name);
        tc.run.export_as = None;
        tc.run.seed = cfg.seed.wrapping_add(trial as u64);
        tc.schedule.max_steps = cfg.steps;
        tc.schedule.warmup_steps = tc.schedule.warmup_steps.min(cfg.steps / 4);
        // Evaluate at rung boundaries and the end, plus every `eval_every` steps if requested.
        tc.eval.every = cfg.eval_every.unwrap_or(cfg.steps).max(1);
        tc.eval.sample_prompts = false;
        tc.checkpoint.every = 0;
        tc.early_stopping.patience = 0;
        let tdir = trial_dir(&dir, trial);
        let mut opts = TrainOptions::new(tc);
        opts.device = Some(device.clone());
        opts.stop_after_step = (budget < cfg.steps).then_some(budget);
        if rung == 0 {
            opts.run_dir = Some(tdir.clone());
        } else {
            opts.resume = Some(tdir.join("checkpoints").join("rung"));
        }
        emit(StudyEvent::TrialStarted { trial, rung, budget, params: params.clone() });
        let summary = train(ws, opts, control, &mut |e| emit(StudyEvent::Train { trial, event: e }))?;
        let status = match summary.reason {
            StopReason::Paused | StopReason::Completed => "ok",
            StopReason::Cancelled => "cancelled",
            StopReason::Diverged => "diverged",
            StopReason::EarlyStopped => "early_stopped",
        };
        let rec = TrialRecord {
            trial,
            rung,
            budget,
            params: params.clone(),
            val_loss: match cfg.objective {
                Objective::Last => summary.last_val_loss,
                Objective::Best => summary.best_val_loss,
            }
            .filter(|v| v.is_finite()),
            train_loss: summary.final_train_loss,
            tokens: summary.tokens_seen,
            wall_secs: summary.wall_secs,
            status: status.into(),
        };
        records.push(rec.clone());
        emit(StudyEvent::TrialFinished(rec.clone()));
        Ok(rec)
    };

    // Rung 0: propose and evaluate configurations one at a time so TPE learns as it goes.
    let mut alive: Vec<(usize, f64)> = Vec::new();
    for trial in 0..cfg.trials {
        if control.should_stop() {
            break;
        }
        let params = propose(cfg.sampler, &cfg.space, &history, &TpeConfig::default(), &mut rng);
        params_of.insert(trial, params.clone());
        let rec = run_trial(trial, 0, &params, &mut records, emit)?;
        serde_json::to_writer(&mut log, &rec)?;
        log.write_all(b"\n")?;
        log.flush()?;
        let value = rec.val_loss.unwrap_or(f64::INFINITY);
        history.push(Observation { params, value });
        alive.push((trial, value));
        if rungs.len() == 1 {
            prune_checkpoints(&trial_dir(&dir, trial));
        }
    }

    // Later rungs: promote the best 1/eta and resume them.
    if let Some(h) = &cfg.successive_halving {
        for rung in 1..rungs.len() {
            if control.should_stop() {
                break;
            }
            alive.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep = (alive.len() / h.eta.max(2)).max(1);
            for (t, _) in &alive[keep..] {
                prune_checkpoints(&trial_dir(&dir, *t));
            }
            alive.truncate(keep);
            emit(StudyEvent::Promoted { rung, trials: alive.iter().map(|a| a.0).collect() });
            let mut next = Vec::new();
            for (trial, _) in &alive {
                if control.should_stop() {
                    break;
                }
                let params = params_of[trial].clone();
                let rec = run_trial(*trial, rung, &params, &mut records, emit)?;
                serde_json::to_writer(&mut log, &rec)?;
                log.write_all(b"\n")?;
                log.flush()?;
                next.push((*trial, rec.val_loss.unwrap_or(f64::INFINITY)));
            }
            alive = next;
        }
        for (t, _) in &alive {
            prune_checkpoints(&trial_dir(&dir, *t));
        }
    }

    // Best = lowest validation loss at the largest budget any trial reached.
    let max_rung = records.iter().map(|r| r.rung).max().unwrap_or(0);
    let best = records
        .iter()
        .filter(|r| r.rung == max_rung && r.val_loss.is_some())
        .min_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).unwrap());
    let summary = StudySummary {
        name: cfg.name.clone(),
        dir: dir.clone(),
        best_trial: best.map(|b| b.trial),
        best_params: best.map(|b| b.params.clone()).unwrap_or_default(),
        best_val_loss: best.and_then(|b| b.val_loss),
        records: records.clone(),
        rungs: rungs.clone(),
        wall_secs: started.elapsed().as_secs_f64(),
    };
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    std::fs::write(dir.join("report.md"), report(cfg, &summary))?;
    emit(StudyEvent::Finished(Box::new(summary.clone())));
    Ok(summary)
}

/// Markdown report of a study.
pub fn report(cfg: &StudyConfig, s: &StudySummary) -> String {
    let mut out = format!("# Study `{}`\n\n{}\n\n", s.name, cfg.description);
    out.push_str(&format!(
        "* Sampler: `{:?}` · trials: {} · full budget: {} steps · rungs: {:?} · objective: {:?} validation loss\n* Base config: `{}`\n* Wall time: {:.1} min\n\n",
        cfg.sampler,
        cfg.trials,
        cfg.steps,
        s.rungs,
        cfg.objective,
        cfg.base_config,
        s.wall_secs / 60.0
    ));
    let names: Vec<String> = cfg.space.iter().map(|p| p.name().to_string()).collect();
    out.push_str("| trial | rung | steps | ");
    out.push_str(&names.join(" | "));
    out.push_str(" | val loss | status |\n|---|---|---|");
    out.push_str(&"---|".repeat(names.len()));
    out.push_str("---|---|\n");
    let mut recs = s.records.clone();
    recs.sort_by(|a, b| {
        (b.rung, a.val_loss.unwrap_or(f64::INFINITY))
            .partial_cmp(&(a.rung, b.val_loss.unwrap_or(f64::INFINITY)))
            .unwrap()
    });
    for r in &recs {
        let vals: Vec<String> =
            names.iter().map(|n| r.params.get(n).map(|v| format_param(n, *v)).unwrap_or_default()).collect();
        let star = if Some(r.trial) == s.best_trial && r.rung == s.rungs.len() - 1 { " ★" } else { "" };
        out.push_str(&format!(
            "| {}{} | {} | {} | {} | {} | {} |\n",
            r.trial,
            star,
            r.rung,
            r.budget,
            vals.join(" | "),
            r.val_loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into()),
            r.status
        ));
    }
    if let Some(b) = s.best_trial {
        out.push_str(&format!(
            "\n**Best:** trial {b} with validation loss {:.4} — {}\n",
            s.best_val_loss.unwrap_or(f64::NAN),
            s.best_params
                .iter()
                .map(|(k, v)| format!("`{k} = {}`", format_param(k, *v)))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    out
}

pub fn format_param(name: &str, v: f64) -> String {
    match name {
        "warmup_steps" | "batch_size" | "grad_accum" | "context_length" | "d_model" | "n_layers" => {
            format!("{}", v.round())
        }
        _ if v.abs() < 1e-2 && v != 0.0 => format!("{v:.2e}"),
        _ => format!("{v:.3}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rung_budgets() {
        let mut c = StudyConfig {
            name: "s".into(),
            description: String::new(),
            base_config: "x".into(),
            sampler: SamplerKind::Random,
            trials: 9,
            steps: 900,
            successive_halving: Some(HalvingConfig { eta: 3, min_steps: 100 }),
            seed: 1,
            backend: None,
            objective: Objective::Last,
            eval_every: None,
            space: vec![],
        };
        assert_eq!(c.rungs(), vec![100, 300, 900]);
        c.steps = 200;
        c.successive_halving = Some(HalvingConfig { eta: 3, min_steps: 22 });
        assert_eq!(c.rungs(), vec![22, 66, 200]);
        c.successive_halving = None;
        assert_eq!(c.rungs(), vec![200]);
    }
}
