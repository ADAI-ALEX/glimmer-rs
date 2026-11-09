//! # glimmer-inference
//!
//! Runs trained GLIMMER models locally — no external APIs.
//!
//! * [`engine`] — loads a model, prefills prompts and decodes token by token with a KV cache,
//!   reusing cached prefixes across chat turns; streams UTF-8-safe text as it is generated.
//! * [`sampling`] — temperature, top-k, top-p and repetition penalty with seeded RNG.
//! * [`chat`] — conversation history, system prompt, context-window management, save/load.

pub mod chat;
pub mod engine;
pub mod sampling;

pub use chat::ChatSession;
pub use engine::{Engine, Generation, GenerationStats, StopReason};
pub use sampling::SamplingConfig;
