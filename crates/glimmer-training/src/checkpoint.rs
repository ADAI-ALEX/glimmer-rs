//! Training checkpoints.
//!
//! ```text
//! runs/<run-id>/checkpoints/<name>/
//!   model.bpk        model parameters (with persistent parameter ids)
//!   optimizer.bpk    optimiser state (AdamW moments etc., keyed by parameter id)
//!   state.json       step, tokens seen, RNG state, data cursor, best-validation tracking
//!   meta.json        what this checkpoint is (kind, step, losses, backend, version)
//!   model.json       resolved architecture
//!   training.toml    the full training configuration (schedule state is a pure function of
//!                    `state.step` and this config, so it is restored exactly)
//!   tokenizer.json   the tokenizer, so a checkpoint is self-contained
//! ```
//!
//! Checkpoints are written to a temporary directory and renamed into place, so an interrupted
//! save never leaves a half-written checkpoint that looks valid.

use anyhow::{Context, Result};
use glimmer_core::config::{ModelConfig, Stage, TrainingConfig};
use glimmer_core::metrics::Ema;
use glimmer_core::Rng;
use glimmer_data::batch::SftCursor;
use glimmer_models::burn::optim::ModuleOptimizer;
use glimmer_models::store;
use glimmer_models::Transformer;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerState {
    /// Number of optimiser steps completed.
    pub step: usize,
    pub tokens_seen: u64,
    pub rng: Rng,
    pub sft_cursor: Option<SftCursor>,
    pub best_val_loss: Option<f64>,
    pub best_step: Option<usize>,
    pub evals_since_best: usize,
    pub train_loss_ema: Ema,
    pub wall_secs: f64,
    pub skipped_steps: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    Periodic,
    Best,
    Final,
    Interrupted,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub run_id: String,
    pub stage: Stage,
    pub kind: CheckpointKind,
    pub step: usize,
    pub tokens_seen: u64,
    pub train_loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub backend: String,
    pub created_unix: u64,
    pub glimmer_version: String,
}

pub struct CheckpointData<'a> {
    pub model: &'a Transformer,
    pub optimizer: &'a ModuleOptimizer,
    pub state: &'a TrainerState,
    pub meta: &'a CheckpointMeta,
    pub training: &'a TrainingConfig,
    pub model_config: &'a ModelConfig,
    pub tokenizer: &'a Path,
}

pub fn save(dir: &Path, data: CheckpointData) -> Result<()> {
    let parent = dir.parent().context("checkpoint path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".tmp-{}-{}",
        dir.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    std::fs::create_dir_all(&tmp)?;
    store::save_weights(data.model, &tmp.join("model.bpk"))?;
    data.optimizer
        .save(tmp.join("optimizer.bpk"))
        .map_err(|e| anyhow::anyhow!("saving optimizer state: {e}"))?;
    std::fs::write(tmp.join("state.json"), serde_json::to_string_pretty(data.state)?)?;
    std::fs::write(tmp.join("meta.json"), serde_json::to_string_pretty(data.meta)?)?;
    std::fs::write(tmp.join("model.json"), serde_json::to_string_pretty(data.model_config)?)?;
    std::fs::write(tmp.join("training.toml"), data.training.to_toml()?)?;
    std::fs::copy(data.tokenizer, tmp.join("tokenizer.json"))
        .with_context(|| format!("copying tokenizer {}", data.tokenizer.display()))?;
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::rename(&tmp, dir)?;
    Ok(())
}

pub fn load_state(dir: &Path) -> Result<(TrainerState, CheckpointMeta)> {
    let state = serde_json::from_str(
        &std::fs::read_to_string(dir.join("state.json"))
            .with_context(|| format!("{} is not a checkpoint (no state.json)", dir.display()))?,
    )?;
    let meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
    Ok((state, meta))
}

pub fn load_model_config(dir: &Path) -> Result<ModelConfig> {
    Ok(serde_json::from_str(
        &std::fs::read_to_string(dir.join("model.json"))
            .with_context(|| format!("{} has no model.json", dir.display()))?,
    )?)
}

/// Periodic checkpoints in a run, oldest first.
pub fn periodic_checkpoints(run_dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(run_dir.join("checkpoints"))
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("step-"))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// Delete all but the newest `keep` periodic checkpoints.
pub fn prune(run_dir: &Path, keep: usize) -> Result<()> {
    let all = periodic_checkpoints(run_dir);
    if all.len() > keep {
        for p in &all[..all.len() - keep] {
            std::fs::remove_dir_all(p)?;
        }
    }
    Ok(())
}

/// The most recent resumable checkpoint of a run (latest periodic, interrupted or final).
pub fn latest(run_dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(usize, PathBuf)> = None;
    let rd = std::fs::read_dir(run_dir.join("checkpoints")).ok()?;
    for e in rd.flatten() {
        let p = e.path();
        if let Ok((state, meta)) = load_state(&p) {
            if meta.kind == CheckpointKind::Best {
                continue;
            }
            if best.as_ref().is_none_or(|(s, _)| state.step > *s) {
                best = Some((state.step, p));
            }
        }
    }
    best.map(|(_, p)| p)
}
