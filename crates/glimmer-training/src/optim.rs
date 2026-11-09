//! Optimiser construction and global gradient-norm clipping.

use glimmer_core::config::{OptimizerConfig, OptimizerKind};
use glimmer_models::burn::module::{Module, ModuleVisitor, Param, ParamGroup};
use glimmer_models::burn::optim::decay::WeightDecayConfig;
use glimmer_models::burn::optim::momentum::MomentumConfig;
use glimmer_models::burn::optim::{AdamWConfig, GradientsParams, LionConfig, ModuleOptimizer, SgdConfig};
use glimmer_models::burn::tensor::Tensor;

/// Parameters excluded from weight decay: normalisation gains and biases. Decaying them pulls
/// RMSNorm gains toward zero, which harms training without regularising anything useful.
pub fn no_decay_group() -> ParamGroup {
    ParamGroup::from_any_predicates(vec!["gamma", "beta", "bias"])
}

/// Build the optimiser: the configured optimiser for weight matrices plus an identical one with
/// zero weight decay for normalisation parameters.
pub fn build(cfg: &OptimizerConfig) -> ModuleOptimizer {
    let (b1, b2, eps, wd) = (cfg.beta1 as f32, cfg.beta2 as f32, cfg.eps as f32, cfg.weight_decay as f32);
    match cfg.kind {
        OptimizerKind::AdamW => {
            let make = |wd: f32| {
                AdamWConfig::new().with_beta_1(b1).with_beta_2(b2).with_epsilon(eps).with_weight_decay(wd).build()
            };
            ModuleOptimizer::from(make(wd)).with_group(no_decay_group(), make(0.0), None)
        }
        OptimizerKind::Lion => {
            let make = |wd: f32| LionConfig::new().with_beta_1(b1).with_beta_2(b2).with_weight_decay(wd).build();
            ModuleOptimizer::from(make(wd)).with_group(no_decay_group(), make(0.0), None)
        }
        OptimizerKind::Sgd => {
            let make = |wd: f32| {
                SgdConfig::new()
                    .with_weight_decay((wd > 0.0).then(|| WeightDecayConfig::new(wd)))
                    .with_momentum(Some(MomentumConfig { momentum: cfg.momentum, dampening: 0.0, nesterov: false }))
                    .build()
            };
            ModuleOptimizer::from(make(wd)).with_group(no_decay_group(), make(0.0), None)
        }
    }
}

struct SquaredNorm<'a> {
    grads: &'a GradientsParams,
    acc: Option<Tensor<1>>,
}

impl ModuleVisitor for SquaredNorm<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(g) = self.grads.get::<D>(param.id) {
            let s = g.powi_scalar(2).sum();
            self.acc = Some(match self.acc.take() {
                Some(a) => a + s,
                None => s,
            });
        }
    }
}

struct Scale<'a> {
    grads: &'a mut GradientsParams,
    factor: f64,
}

impl ModuleVisitor for Scale<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(g) = self.grads.remove::<D>(param.id) {
            self.grads.register::<D>(param.id, g.mul_scalar(self.factor));
        }
    }
}

/// Global L2 norm of all gradients (one device synchronisation).
pub fn grad_norm<M: Module>(model: &M, grads: &GradientsParams) -> f64 {
    let mut v = SquaredNorm { grads, acc: None };
    model.visit(&mut v);
    v.acc.map(|t| (t.into_scalar::<f32>() as f64).sqrt()).unwrap_or(0.0)
}

/// Clip gradients to a maximum global norm. Returns `(norm_before, clipped)`.
pub fn clip_global_norm<M: Module>(model: &M, grads: &mut GradientsParams, max_norm: f64) -> (f64, bool) {
    let norm = grad_norm(model, grads);
    if max_norm > 0.0 && norm.is_finite() && norm > max_norm {
        let mut s = Scale { grads, factor: max_norm / (norm + 1e-6) };
        model.visit(&mut s);
        (norm, true)
    } else {
        (norm, false)
    }
}
