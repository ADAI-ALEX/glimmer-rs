//! Search spaces and how sampled values are applied to a training configuration.

use anyhow::{Result, bail};
use glimmer_core::Rng;
use glimmer_core::config::TrainingConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One searchable hyperparameter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParamSpec {
    /// Uniform in log space (learning rates, weight decay).
    LogUniform {
        name: String,
        low: f64,
        high: f64,
    },
    Uniform {
        name: String,
        low: f64,
        high: f64,
    },
    /// One of a fixed set of values (batch sizes, layer counts).
    Choice {
        name: String,
        values: Vec<f64>,
    },
}

impl ParamSpec {
    pub fn name(&self) -> &str {
        match self {
            ParamSpec::LogUniform { name, .. } | ParamSpec::Uniform { name, .. } | ParamSpec::Choice { name, .. } => {
                name
            }
        }
    }

    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            ParamSpec::LogUniform { low, high, .. } => (rng.uniform(low.ln(), high.ln())).exp(),
            ParamSpec::Uniform { low, high, .. } => rng.uniform(*low, *high),
            ParamSpec::Choice { values, .. } => values[rng.below(values.len())],
        }
    }

    /// Map a value to the internal space in which densities are modelled.
    pub fn to_internal(&self, v: f64) -> f64 {
        match self {
            ParamSpec::LogUniform { .. } => v.ln(),
            _ => v,
        }
    }

    pub fn from_internal(&self, x: f64) -> f64 {
        match self {
            ParamSpec::LogUniform { low, high, .. } => x.exp().clamp(*low, *high),
            ParamSpec::Uniform { low, high, .. } => x.clamp(*low, *high),
            ParamSpec::Choice { .. } => x,
        }
    }

    pub fn bounds_internal(&self) -> (f64, f64) {
        match self {
            ParamSpec::LogUniform { low, high, .. } => (low.ln(), high.ln()),
            ParamSpec::Uniform { low, high, .. } => (*low, *high),
            ParamSpec::Choice { values, .. } => (
                values.iter().cloned().fold(f64::INFINITY, f64::min),
                values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ),
        }
    }
}

/// Parameter names understood by [`apply`].
pub const SUPPORTED: [&str; 12] = [
    "lr",
    "min_lr_ratio",
    "weight_decay",
    "warmup_steps",
    "batch_size",
    "grad_accum",
    "dropout",
    "context_length",
    "grad_clip",
    "beta2",
    "d_model",
    "n_layers",
];

pub fn validate(space: &[ParamSpec]) -> Result<()> {
    for p in space {
        if !SUPPORTED.contains(&p.name()) {
            bail!("unsupported search parameter '{}' (supported: {SUPPORTED:?})", p.name());
        }
        match p {
            ParamSpec::LogUniform { low, high, .. } if *low <= 0.0 || low >= high => {
                bail!("log_uniform '{}' needs 0 < low < high", p.name())
            }
            ParamSpec::Uniform { low, high, .. } if low >= high => bail!("uniform '{}' needs low < high", p.name()),
            ParamSpec::Choice { values, .. } if values.is_empty() => bail!("choice '{}' needs values", p.name()),
            _ => {}
        }
    }
    Ok(())
}

/// Apply sampled hyperparameters to a copy of the base configuration. Every change is explicit
/// and recorded with the trial, so each trial is exactly reproducible from its stored config.
pub fn apply(base: &TrainingConfig, params: &BTreeMap<String, f64>) -> TrainingConfig {
    let mut cfg = base.clone();
    let ratio = base.optimizer.min_lr / base.optimizer.lr;
    let ratio = params.get("min_lr_ratio").copied().unwrap_or(ratio);
    for (k, &v) in params {
        match k.as_str() {
            "lr" => cfg.optimizer.lr = v,
            "weight_decay" => cfg.optimizer.weight_decay = v,
            "warmup_steps" => cfg.schedule.warmup_steps = v.round() as usize,
            "batch_size" => cfg.batch.batch_size = v.round().max(1.0) as usize,
            "grad_accum" => cfg.batch.grad_accum = v.round().max(1.0) as usize,
            "dropout" => cfg.model.dropout = Some(v),
            "context_length" => cfg.model.context_length = Some(v.round() as usize),
            "grad_clip" => cfg.optimizer.grad_clip = v,
            "beta2" => cfg.optimizer.beta2 = v,
            "d_model" => cfg.model.d_model = Some(v.round() as usize),
            "n_layers" => cfg.model.n_layers = Some(v.round() as usize),
            _ => {}
        }
    }
    cfg.optimizer.min_lr = cfg.optimizer.lr * ratio.clamp(0.0, 1.0);
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_uniform_samples_in_range_and_log_spread() {
        let p = ParamSpec::LogUniform { name: "lr".into(), low: 1e-4, high: 1e-2 };
        let mut rng = Rng::new(1);
        let xs: Vec<f64> = (0..2000).map(|_| p.sample(&mut rng)).collect();
        assert!(xs.iter().all(|&x| (1e-4..=1e-2).contains(&x)));
        let below_1e3 = xs.iter().filter(|&&x| x < 1e-3).count() as f64 / 2000.0;
        assert!((below_1e3 - 0.5).abs() < 0.05, "log-uniform should split evenly at the geometric mean");
    }

    #[test]
    fn rejects_unknown_parameters() {
        assert!(validate(&[ParamSpec::Uniform { name: "nonsense".into(), low: 0.0, high: 1.0 }]).is_err());
        assert!(validate(&[ParamSpec::LogUniform { name: "lr".into(), low: 0.0, high: 1.0 }]).is_err());
    }
}
