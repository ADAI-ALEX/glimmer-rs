//! Configuration types for models, training runs and data.
//!
//! All configuration is plain serde data so it can be stored in TOML files under `configs/`,
//! embedded in checkpoints, inspected by the terminal UI, and diffed between experiments.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Which compute backend to run on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default, Hash)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// Use Metal when a working Apple GPU is detected, otherwise fall back to the CPU.
    #[default]
    Auto,
    /// Apple GPU through Burn's wgpu runtime with native MSL kernels.
    Metal,
    /// Pure-Rust CPU backend (Burn Flex, multithreaded, SIMD; Apple AMX on macOS).
    Cpu,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Auto => "auto",
            BackendKind::Metal => "metal",
            BackendKind::Cpu => "cpu",
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "metal" | "gpu" => Ok(Self::Metal),
            "cpu" | "flex" => Ok(Self::Cpu),
            other => bail!("unknown backend '{other}' (expected auto, metal or cpu)"),
        }
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Normalisation layer used inside transformer blocks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum NormKind {
    #[default]
    RmsNorm,
    LayerNorm,
}

/// Feed-forward network variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum FfnKind {
    /// Gated SiLU unit (Shazeer, 2020). Three projections.
    #[default]
    SwiGlu,
    /// Classic two-layer GELU MLP.
    Gelu,
}

/// Parameter initialisation strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum InitStrategy {
    /// N(0, init_std) for all weights, with residual output projections scaled by
    /// 1/sqrt(2 * n_layers) (GPT-2 / nanoGPT scheme).
    #[default]
    Scaled,
    /// N(0, init_std) for every weight matrix.
    Normal,
    /// Glorot/Xavier uniform for projections, N(0, init_std) for embeddings.
    Xavier,
}

/// Architecture hyperparameters of a decoder-only transformer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Human-readable architecture name (e.g. the preset it came from).
    #[serde(default = "default_arch_name")]
    pub name: String,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub context_length: usize,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub norm: NormKind,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub ffn: FfnKind,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub init: InitStrategy,
    #[serde(default = "default_init_std")]
    pub init_std: f64,
    #[serde(default = "default_true")]
    pub tie_embeddings: bool,
}

fn default_arch_name() -> String {
    "custom".into()
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_init_std() -> f64 {
    0.02
}
fn default_true() -> bool {
    true
}

/// Names of the built-in model presets.
pub const PRESETS: [&str; 3] = ["micro", "tiny", "small"];

impl ModelConfig {
    /// Built-in presets. `vocab_size` is a placeholder that is replaced by the tokenizer's real
    /// vocabulary size when a training run is resolved.
    pub fn preset(name: &str) -> Option<Self> {
        let base = |name: &str, d, l, h, ff, ctx| ModelConfig {
            name: name.to_string(),
            vocab_size: 8192,
            d_model: d,
            n_layers: l,
            n_heads: h,
            d_ff: ff,
            context_length: ctx,
            dropout: 0.0,
            norm: NormKind::RmsNorm,
            norm_eps: 1e-5,
            ffn: FfnKind::SwiGlu,
            rope_theta: 10_000.0,
            init: InitStrategy::Scaled,
            init_std: 0.02,
            tie_embeddings: true,
        };
        match name {
            // Debugging and correctness tests: compiles and trains in seconds on any backend.
            "micro" => Some(ModelConfig { vocab_size: 512, ..base("micro", 64, 2, 4, 192, 64) }),
            // The local demonstration size for STEVIE on a 16 GB Apple Silicon laptop.
            "tiny" => Some(base("tiny", 384, 6, 6, 1024, 512)),
            // Larger experiments when memory and time permit.
            "small" => Some(base("small", 512, 8, 8, 1408, 512)),
            _ => None,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Exact number of trainable parameters implied by this configuration.
    pub fn param_count(&self) -> usize {
        let d = self.d_model;
        let norm_params = match self.norm {
            NormKind::RmsNorm => d,
            NormKind::LayerNorm => 2 * d,
        };
        let ffn = match self.ffn {
            FfnKind::SwiGlu => 3 * d * self.d_ff,
            FfnKind::Gelu => 2 * d * self.d_ff,
        };
        let per_layer = 2 * norm_params + 4 * d * d + ffn;
        let embeddings = self.vocab_size * d;
        let head = if self.tie_embeddings { 0 } else { self.vocab_size * d };
        embeddings + self.n_layers * per_layer + norm_params + head
    }

    /// Approximate training FLOPs per token (forward + backward), following Kaplan et al.:
    /// 6 * N_non_embedding + attention term 6 * n_layers * context * d_model.
    pub fn train_flops_per_token(&self) -> f64 {
        let non_embedding = (self.param_count() - self.vocab_size * self.d_model) as f64;
        let lm_head = (self.vocab_size * self.d_model) as f64;
        let attn = (self.n_layers * self.context_length * self.d_model) as f64;
        6.0 * (non_embedding + lm_head) + 6.0 * attn
    }

    pub fn validate(&self) -> Result<()> {
        if self.vocab_size < 8 {
            bail!("vocab_size must be at least 8");
        }
        if self.d_model == 0 || self.n_layers == 0 || self.n_heads == 0 || self.d_ff == 0 {
            bail!("model dimensions must be non-zero");
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            bail!("d_model ({}) must be divisible by n_heads ({})", self.d_model, self.n_heads);
        }
        if !self.head_dim().is_multiple_of(2) {
            bail!("head dimension must be even for rotary embeddings");
        }
        if self.context_length < 2 {
            bail!("context_length must be at least 2");
        }
        if !(0.0..1.0).contains(&self.dropout) {
            bail!("dropout must be in [0, 1)");
        }
        Ok(())
    }
}

/// A model reference inside a training config: a preset plus optional field overrides.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub preset: Option<String>,
    pub d_model: Option<usize>,
    pub n_layers: Option<usize>,
    pub n_heads: Option<usize>,
    pub d_ff: Option<usize>,
    pub context_length: Option<usize>,
    pub dropout: Option<f64>,
    pub norm: Option<NormKind>,
    pub ffn: Option<FfnKind>,
    pub rope_theta: Option<f64>,
    pub init: Option<InitStrategy>,
    pub init_std: Option<f64>,
    pub tie_embeddings: Option<bool>,
}

impl ModelSpec {
    /// Resolve into a concrete [`ModelConfig`] using the tokenizer's vocabulary size.
    pub fn resolve(&self, vocab_size: usize) -> Result<ModelConfig> {
        let preset = self.preset.as_deref().unwrap_or("tiny");
        let mut cfg = ModelConfig::preset(preset)
            .with_context(|| format!("unknown model preset '{preset}' (available: {PRESETS:?})"))?;
        cfg.vocab_size = vocab_size;
        macro_rules! apply {
            ($($f:ident),*) => { $( if let Some(v) = self.$f { cfg.$f = v; } )* };
        }
        apply!(
            d_model,
            n_layers,
            n_heads,
            d_ff,
            context_length,
            dropout,
            norm,
            ffn,
            rope_theta,
            init,
            init_std,
            tie_embeddings
        );
        let customised =
            self.d_model.is_some() || self.n_layers.is_some() || self.n_heads.is_some() || self.d_ff.is_some();
        if customised {
            cfg.name = format!("{preset}*");
        }
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Training stage.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    /// Causal language modelling on a raw token stream.
    #[default]
    Pretrain,
    /// Supervised conversational fine-tuning with assistant-only loss masking.
    Sft,
}

impl Stage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Stage::Pretrain => "pretrain",
            Stage::Sft => "sft",
        }
    }
}

/// Optimiser selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerKind {
    #[default]
    AdamW,
    Lion,
    Sgd,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OptimizerConfig {
    #[serde(default)]
    pub kind: OptimizerKind,
    /// Peak learning rate.
    pub lr: f64,
    /// Final learning rate at the end of decay.
    #[serde(default)]
    pub min_lr: f64,
    #[serde(default = "default_wd")]
    pub weight_decay: f64,
    #[serde(default = "default_beta1")]
    pub beta1: f64,
    #[serde(default = "default_beta2")]
    pub beta2: f64,
    #[serde(default = "default_eps")]
    pub eps: f64,
    /// SGD momentum (only used by `sgd`).
    #[serde(default = "default_momentum")]
    pub momentum: f64,
    /// Global gradient-norm clipping threshold. `0` disables clipping.
    #[serde(default = "default_clip")]
    pub grad_clip: f64,
}

fn default_wd() -> f64 {
    0.1
}
fn default_beta1() -> f64 {
    0.9
}
fn default_beta2() -> f64 {
    0.95
}
fn default_eps() -> f64 {
    1e-8
}
fn default_momentum() -> f64 {
    0.9
}
fn default_clip() -> f64 {
    1.0
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            kind: OptimizerKind::AdamW,
            lr: 6e-4,
            min_lr: 6e-5,
            weight_decay: default_wd(),
            beta1: default_beta1(),
            beta2: default_beta2(),
            eps: default_eps(),
            momentum: default_momentum(),
            grad_clip: default_clip(),
        }
    }
}

/// Learning-rate schedule shape. All schedules use linear warmup.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum LrSchedule {
    #[default]
    Cosine,
    Linear,
    Constant,
    /// Warmup-stable-decay: constant after warmup, linear decay over the final `decay_fraction`.
    Wsd,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScheduleConfig {
    #[serde(default)]
    pub kind: LrSchedule,
    #[serde(default)]
    pub warmup_steps: usize,
    pub max_steps: usize,
    /// Fraction of training spent decaying (WSD only).
    #[serde(default = "default_decay_fraction")]
    pub decay_fraction: f64,
}

fn default_decay_fraction() -> f64 {
    0.2
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self { kind: LrSchedule::Cosine, warmup_steps: 100, max_steps: 1000, decay_fraction: default_decay_fraction() }
    }
}

impl ScheduleConfig {
    /// Learning rate at optimiser step `step` (0-based). Pure function of the step, which makes
    /// scheduler state trivially checkpointable and exactly resumable.
    pub fn lr_at(&self, step: usize, peak: f64, min: f64) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return peak * (step + 1) as f64 / self.warmup_steps as f64;
        }
        let total = self.max_steps.max(self.warmup_steps + 1);
        let decay_steps = (total - self.warmup_steps).max(1) as f64;
        let progress = ((step - self.warmup_steps.min(step)) as f64 / decay_steps).clamp(0.0, 1.0);
        match self.kind {
            LrSchedule::Constant => peak,
            LrSchedule::Cosine => min + 0.5 * (peak - min) * (1.0 + (std::f64::consts::PI * progress).cos()),
            LrSchedule::Linear => peak + (min - peak) * progress,
            LrSchedule::Wsd => {
                let start = 1.0 - self.decay_fraction.clamp(0.0, 1.0);
                if progress < start {
                    peak
                } else {
                    let p = ((progress - start) / (1.0 - start).max(1e-9)).clamp(0.0, 1.0);
                    peak + (min - peak) * p
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BatchConfig {
    /// Sequences per micro-batch.
    pub batch_size: usize,
    /// Micro-batches accumulated per optimiser step.
    #[serde(default = "default_one")]
    pub grad_accum: usize,
}

fn default_one() -> usize {
    1
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self { batch_size: 16, grad_accum: 1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvalConfig {
    /// Evaluate every N optimiser steps.
    pub every: usize,
    /// Number of validation batches per evaluation.
    pub batches: usize,
    /// Generate samples from the held-out prompts at each evaluation.
    #[serde(default)]
    pub sample_prompts: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self { every: 200, batches: 20, sample_prompts: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointConfig {
    /// Save a checkpoint every N optimiser steps (0 = only at the end / on request).
    pub every: usize,
    /// Number of periodic checkpoints to keep (the best checkpoint is always kept).
    #[serde(default = "default_keep")]
    pub keep_last: usize,
}

fn default_keep() -> usize {
    2
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self { every: 500, keep_last: 2 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EarlyStoppingConfig {
    /// Stop after this many evaluations without validation improvement. 0 disables.
    pub patience: usize,
    /// Minimum decrease in validation loss that counts as an improvement.
    #[serde(default = "default_min_delta")]
    pub min_delta: f64,
}

fn default_min_delta() -> f64 {
    1e-3
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self { patience: 0, min_delta: default_min_delta() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// Name of a prepared dataset directory under `data/datasets/`.
    pub dataset: String,
    /// Tokenizer file. Defaults to the tokenizer recorded in the dataset manifest.
    pub tokenizer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct InitConfig {
    /// Initialise weights from a checkpoint directory (e.g. a pretrained model for SFT).
    /// When absent the model starts from random initialisation.
    pub from_checkpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Run name prefix; a timestamp is appended to form the run id.
    pub name: String,
    #[serde(default)]
    pub stage: Stage,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub backend: BackendKind,
    /// Optional model name to export the best checkpoint to under `models/` when training ends.
    #[serde(default)]
    pub export_as: Option<String>,
}

fn default_seed() -> u64 {
    1337
}

/// A complete, self-describing training run configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub run: RunConfig,
    #[serde(default)]
    pub model: ModelSpec,
    pub data: DataConfig,
    #[serde(default)]
    pub init: InitConfig,
    #[serde(default)]
    pub optimizer: OptimizerConfig,
    #[serde(default)]
    pub schedule: ScheduleConfig,
    #[serde(default)]
    pub batch: BatchConfig,
    #[serde(default)]
    pub eval: EvalConfig,
    #[serde(default)]
    pub checkpoint: CheckpointConfig,
    #[serde(default)]
    pub early_stopping: EarlyStoppingConfig,
}

impl TrainingConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading training config {}", path.display()))?;
        Self::from_toml(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn from_toml(text: &str) -> Result<Self> {
        let cfg: Self = toml::from_str(text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn validate(&self) -> Result<()> {
        if self.batch.batch_size == 0 || self.batch.grad_accum == 0 {
            bail!("batch_size and grad_accum must be positive");
        }
        if self.schedule.max_steps == 0 {
            bail!("schedule.max_steps must be positive");
        }
        if self.optimizer.lr <= 0.0 || !self.optimizer.lr.is_finite() {
            bail!("optimizer.lr must be a positive finite number");
        }
        if self.optimizer.min_lr < 0.0 || self.optimizer.min_lr > self.optimizer.lr {
            bail!("optimizer.min_lr must be in [0, lr]");
        }
        if self.eval.every == 0 {
            bail!("eval.every must be positive");
        }
        if self.data.dataset.is_empty() {
            bail!("data.dataset must name a prepared dataset");
        }
        Ok(())
    }

    /// Tokens processed per optimiser step for a given context length.
    pub fn tokens_per_step(&self, context_length: usize) -> usize {
        self.batch.batch_size * self.batch.grad_accum * context_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_validate_and_count_params() {
        for name in PRESETS {
            let cfg = ModelConfig::preset(name).unwrap();
            cfg.validate().unwrap();
            assert!(cfg.param_count() > 0);
        }
        // Hand-computed for the micro preset:
        // emb 512*64 + 2 layers * (2*64 + 4*64*64 + 3*64*192) + 64
        let micro = ModelConfig::preset("micro").unwrap();
        let expected = 512 * 64 + 2 * (2 * 64 + 4 * 64 * 64 + 3 * 64 * 192) + 64;
        assert_eq!(micro.param_count(), expected);
    }

    #[test]
    fn spec_overrides_apply() {
        let spec = ModelSpec {
            preset: Some("tiny".into()),
            context_length: Some(256),
            dropout: Some(0.1),
            ..Default::default()
        };
        let cfg = spec.resolve(4000).unwrap();
        assert_eq!(cfg.vocab_size, 4000);
        assert_eq!(cfg.context_length, 256);
        assert_eq!(cfg.dropout, 0.1);
        assert_eq!(cfg.name, "tiny");
    }

    #[test]
    fn invalid_heads_rejected() {
        let spec = ModelSpec { preset: Some("micro".into()), n_heads: Some(3), ..Default::default() };
        assert!(spec.resolve(512).is_err());
    }

    #[test]
    fn cosine_schedule_shape() {
        let s = ScheduleConfig { kind: LrSchedule::Cosine, warmup_steps: 10, max_steps: 110, decay_fraction: 0.2 };
        assert!((s.lr_at(0, 1.0, 0.1) - 0.1).abs() < 1e-12);
        assert!((s.lr_at(9, 1.0, 0.1) - 1.0).abs() < 1e-12);
        assert!((s.lr_at(10, 1.0, 0.1) - 1.0).abs() < 1e-12);
        assert!((s.lr_at(60, 1.0, 0.1) - 0.55).abs() < 1e-9);
        assert!((s.lr_at(110, 1.0, 0.1) - 0.1).abs() < 1e-12);
        assert!((s.lr_at(500, 1.0, 0.1) - 0.1).abs() < 1e-12);
        // Monotonic non-increasing after warmup.
        let mut prev = f64::INFINITY;
        for step in 10..120 {
            let lr = s.lr_at(step, 1.0, 0.1);
            assert!(lr <= prev + 1e-12);
            prev = lr;
        }
    }

    #[test]
    fn wsd_schedule_shape() {
        let s = ScheduleConfig { kind: LrSchedule::Wsd, warmup_steps: 0, max_steps: 100, decay_fraction: 0.2 };
        assert_eq!(s.lr_at(10, 1.0, 0.0), 1.0);
        assert_eq!(s.lr_at(79, 1.0, 0.0), 1.0);
        assert!((s.lr_at(90, 1.0, 0.0) - 0.5).abs() < 1e-9);
        assert!(s.lr_at(100, 1.0, 0.0).abs() < 1e-9);
    }

    #[test]
    fn training_config_roundtrip() {
        let text = r#"
            [run]
            name = "test"
            stage = "sft"
            [model]
            preset = "micro"
            [data]
            dataset = "stevie-sft"
            [optimizer]
            lr = 1e-3
            min_lr = 1e-4
            [schedule]
            max_steps = 10
        "#;
        let cfg = TrainingConfig::from_toml(text).unwrap();
        assert_eq!(cfg.run.stage, Stage::Sft);
        let again = TrainingConfig::from_toml(&cfg.to_toml().unwrap()).unwrap();
        assert_eq!(cfg, again);
    }

    #[test]
    fn unknown_fields_rejected() {
        let text = r#"
            [run]
            name = "test"
            [data]
            dataset = "x"
            [optimizer]
            lr = 1e-3
            learning_rate = 5
        "#;
        assert!(TrainingConfig::from_toml(text).is_err());
    }
}
