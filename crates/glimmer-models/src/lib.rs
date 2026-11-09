//! # glimmer-models
//!
//! Model definitions and compute-backend management for GLIMMER.
//!
//! * [`transformer`] — the decoder-only transformer (RMSNorm, RoPE, causal attention, SwiGLU,
//!   tied embeddings) with packed-sequence and KV-cached forward passes.
//! * [`loss`] — numerically stable, mask-aware causal language-modelling loss.
//! * [`backend`] — Metal/CPU selection with a numerical self-test (no unverified claims).
//! * [`store`] — saving and loading weights with their configuration.
//!
//! Adding a new architecture means adding a module here that consumes a [`ModelConfig`] and
//! exposes the same forward-pass shape; the trainer, inference engine and TUI are unchanged.

pub mod backend;
pub mod loss;
pub mod store;
pub mod transformer;

pub use backend::{ComputeDevice, select};
pub use glimmer_core::ModelConfig;
pub use transformer::{KvCache, Transformer};

pub use burn;
