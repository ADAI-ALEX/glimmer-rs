//! Weight persistence.
//!
//! Weights are stored in Burn's burnpack format (`model.bpk`), which records each parameter's
//! module path, persistent parameter id and data. Persisting parameter ids is what allows
//! optimiser state (keyed by id) to be restored exactly when training resumes.

use crate::transformer::Transformer;
use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::store::ModuleRecord;
use burn::tensor::Device;
use glimmer_core::ModelConfig;
use std::path::Path;

pub fn save_weights(model: &Transformer, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    model.clone().save_file(path).map_err(|e| anyhow!("saving weights to {}: {e}", path.display()))
}

/// Build a model from `cfg` and load weights saved by [`save_weights`]. Shapes are validated:
/// a checkpoint from a different architecture is rejected rather than partially loaded.
pub fn load_weights(cfg: &ModelConfig, device: &Device, path: &Path) -> Result<Transformer> {
    let record = ModuleRecord::load(path)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("reading weights {}", path.display()))?;
    Transformer::init(cfg, device, 0)
        .try_load_record(record)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("applying weights {} (architecture mismatch?)", path.display()))
}
