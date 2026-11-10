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
///
/// The file is read completely before the model is built. Burn's file loader is lazy (each tensor
/// is read from disk when first accessed), so a model loaded that way could fail later if its
/// checkpoint were pruned or replaced in the meantime, which training does routinely.
pub fn load_weights(cfg: &ModelConfig, device: &Device, path: &Path) -> Result<Transformer> {
    let bytes = std::fs::read(path).with_context(|| format!("reading weights {}", path.display()))?;
    let record = ModuleRecord::from_bytes(burn::tensor::Bytes::from_bytes_vec(bytes))
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("parsing weights {}", path.display()))?;
    Transformer::init(cfg, device, 0)
        .try_load_record(record)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("applying weights {} (architecture mismatch?)", path.display()))
}

/// Sum of every parameter, read back to the host. Forces all weights to be resident and
/// up to date on the model's device (used to time loading honestly, and as a cheap checksum).
pub fn parameter_checksum(model: &Transformer) -> f64 {
    struct Sum(f64);
    impl burn::module::ModuleVisitor for Sum {
        fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<burn::tensor::Tensor<D>>) {
            self.0 += param.val().sum().into_scalar::<f32>() as f64;
        }
    }
    let mut s = Sum(0.0);
    model.visit(&mut s);
    s.0
}
