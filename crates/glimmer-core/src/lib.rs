//! # glimmer-core
//!
//! Backend-agnostic building blocks shared by every GLIMMER crate:
//!
//! * [`config`] — model, training and dataset configuration (TOML, serde).
//! * [`hardware`] — real hardware inspection (CPU, memory, Apple GPU, Metal).
//! * [`gpu`] — live GPU counters read from the macOS IORegistry (no fabricated values).
//! * [`metrics`] — training/evaluation metric records shared by the trainer, CLI and TUI.
//! * [`rng`] — a small, fully serialisable PRNG so every run is reproducible and resumable.
//! * [`workspace`] — discovery of the project root and its standard directories.
//!
//! This crate intentionally does not depend on Burn so that tools such as the terminal UI can
//! inspect configurations and metrics without pulling in the tensor runtime.

pub mod config;
pub mod gpu;
pub mod hardware;
pub mod metrics;
pub mod registry;
pub mod rng;
pub mod time;
pub mod workspace;

pub use config::{
    BackendKind, DataConfig, LrSchedule, ModelConfig, OptimizerConfig, Stage, TrainingConfig,
};
pub use rng::Rng;
pub use workspace::Workspace;

/// Version of the GLIMMER framework.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Version of the Rust compiler that built this binary.
pub const RUSTC_VERSION: &str = env!("GLIMMER_RUSTC_VERSION");

/// Cargo profile used for this build (`release` or `debug`).
pub const BUILD_PROFILE: &str = env!("GLIMMER_BUILD_PROFILE");
