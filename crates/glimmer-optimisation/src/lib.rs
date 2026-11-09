//! # glimmer-optimisation
//!
//! Automatic hyperparameter optimisation for GLIMMER training runs.
//!
//! * [`space`] — search spaces (log-uniform, uniform, choice) and explicit application to configs.
//! * [`sampler`] — random search and Tree-structured Parzen Estimator (Bayesian) proposals.
//! * [`study`] — sequential studies with optional successive halving, full experiment tracking
//!   and Markdown reports.
//!
//! The optimiser only ever changes a trial's configuration *before* it starts, and records every
//! value it chose, so each trial is exactly reproducible. Reward-based post-training (RL) is a
//! different technique and deliberately out of scope here.

pub mod sampler;
pub mod space;
pub mod study;

pub use sampler::SamplerKind;
pub use space::ParamSpec;
pub use study::{StudyConfig, StudyEvent, StudySummary, run_study};
