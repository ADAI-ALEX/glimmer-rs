//! # glimmer-training
//!
//! The complete training lifecycle for GLIMMER models:
//!
//! * **Pretraining** — next-token prediction on random windows of a token stream.
//! * **Supervised fine-tuning** — packed conversations with assistant-only loss masking,
//!   segment-isolated attention and per-segment positions.
//! * AdamW (default), Lion or SGD; decoupled weight decay (not applied to norm gains);
//!   global gradient-norm clipping; gradient accumulation normalised by supervised tokens;
//!   warmup + cosine/linear/WSD schedules; validation; best-checkpoint selection; early stopping.
//! * **Checkpoints** with model, optimiser, scheduler/step, RNG and data-cursor state for exact
//!   resumption.
//! * A **monitor** that detects spikes, excessive clipping, overfitting and plateaus and logs
//!   advice without silently changing anything.

pub mod checkpoint;
pub mod events;
pub mod monitor;
pub mod optim;
pub mod trainer;

pub use events::{RunInfo, RunSummary, StopReason, TrainControl, TrainEvent};
pub use trainer::{Sampler, TrainOptions, evaluate, resolve_init, train};
