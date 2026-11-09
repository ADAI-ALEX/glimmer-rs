//! The local model registry: ready-to-use models under `models/<name>/`.
//!
//! ```text
//! models/stevie/
//!   model.bpk        weights (burnpack)
//!   model.json       architecture (ModelConfig)
//!   tokenizer.json   the tokenizer the model was trained with
//!   card.json        provenance: training stages, data, steps, tokens, losses, hardware
//! ```

use crate::config::ModelConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One training stage in a model's lineage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageRecord {
    pub stage: String,
    pub run_id: String,
    pub dataset: String,
    pub steps: usize,
    pub tokens: u64,
    pub final_train_loss: Option<f64>,
    pub best_val_loss: Option<f64>,
    pub backend: String,
    pub wall_secs: f64,
    /// Checkpoint this stage started from (`None` = random initialisation).
    pub initialised_from: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCard {
    pub name: String,
    pub description: String,
    pub architecture: String,
    pub parameters: usize,
    pub context_length: usize,
    pub vocab_size: usize,
    /// Training lineage, oldest first. The first stage always starts from random weights.
    pub stages: Vec<StageRecord>,
    /// System prompt the model was fine-tuned with, if any.
    pub system_prompt: Option<String>,
    pub hardware: Option<String>,
    pub created_unix: u64,
    pub glimmer_version: String,
}

impl ModelCard {
    pub fn total_tokens(&self) -> u64 {
        self.stages.iter().map(|s| s.tokens).sum()
    }
    pub fn total_steps(&self) -> usize {
        self.stages.iter().map(|s| s.steps).sum()
    }
}

/// A model found in the registry.
#[derive(Debug, Clone)]
pub struct RegisteredModel {
    pub name: String,
    pub dir: PathBuf,
    pub config: ModelConfig,
    pub card: Option<ModelCard>,
}

impl RegisteredModel {
    pub fn open(dir: &Path) -> Result<Self> {
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(dir.join("model.json"))
                .with_context(|| format!("{} is not a GLIMMER model directory", dir.display()))?,
        )?;
        let card = std::fs::read_to_string(dir.join("card.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
        Ok(Self {
            name: dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            dir: dir.to_path_buf(),
            config,
            card,
        })
    }

    pub fn weights(&self) -> PathBuf {
        self.dir.join("model.bpk")
    }
    pub fn tokenizer(&self) -> PathBuf {
        self.dir.join("tokenizer.json")
    }
}

/// List models in `models/` (directories containing `model.json` and `model.bpk`).
pub fn list_models(models_dir: &Path) -> Vec<RegisteredModel> {
    let mut out: Vec<RegisteredModel> = std::fs::read_dir(models_dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().join("model.bpk").is_file())
                .filter_map(|e| RegisteredModel::open(&e.path()).ok())
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Resolve a model reference: a registered name, a model directory, or a checkpoint directory.
pub fn resolve_model(models_dir: &Path, root: &Path, reference: &str) -> Result<RegisteredModel> {
    let by_name = models_dir.join(reference);
    if by_name.join("model.json").is_file() {
        return RegisteredModel::open(&by_name);
    }
    let path = if Path::new(reference).is_absolute() { PathBuf::from(reference) } else { root.join(reference) };
    if path.join("model.json").is_file() {
        return RegisteredModel::open(&path);
    }
    anyhow::bail!("model '{reference}' not found (looked in {} and {})", by_name.display(), path.display())
}
