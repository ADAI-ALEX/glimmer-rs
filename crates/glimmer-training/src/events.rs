//! Events emitted by the trainer and the control handle used to steer it.
//!
//! The trainer runs on its own thread and reports through a callback; the terminal UI forwards
//! events over a channel to its render loop, the CLI prints them, and hyperparameter search
//! records them. Nothing in the training loop depends on how events are displayed.

use glimmer_core::config::{BackendKind, ModelConfig, Stage, TrainingConfig};
use glimmer_core::metrics::{EvalMetrics, EventRecord, SampleRecord, StepMetrics};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub run_dir: PathBuf,
    pub stage: Stage,
    pub model: ModelConfig,
    pub parameters: usize,
    pub backend: BackendKind,
    pub backend_label: String,
    pub fallback_reason: Option<String>,
    pub dataset: String,
    pub train_tokens: u64,
    pub start_step: usize,
    pub max_steps: usize,
    pub tokens_per_step: usize,
    pub resumed_from: Option<String>,
    pub initialised_from: Option<String>,
    pub config: TrainingConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Completed,
    EarlyStopped,
    Cancelled,
    Diverged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub run_dir: PathBuf,
    pub stage: Stage,
    pub reason: StopReason,
    pub steps_completed: usize,
    pub tokens_seen: u64,
    pub final_train_loss: Option<f64>,
    pub best_val_loss: Option<f64>,
    pub best_step: Option<usize>,
    pub best_checkpoint: Option<PathBuf>,
    pub last_checkpoint: Option<PathBuf>,
    pub wall_secs: f64,
    pub exported_model: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum TrainEvent {
    Started(Box<RunInfo>),
    Step(StepMetrics),
    Eval(EvalMetrics),
    Sample(SampleRecord),
    Event(EventRecord),
    Paused(bool),
    Finished(Box<RunSummary>),
}

/// Shared, lock-free control flags. Clone the `Arc` into the UI thread.
#[derive(Debug, Default)]
pub struct TrainControl {
    paused: AtomicBool,
    stop: AtomicBool,
    checkpoint: AtomicBool,
}

impl TrainControl {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }
    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }
    pub fn toggle_pause(&self) -> bool {
        let now = !self.paused.load(Ordering::SeqCst);
        self.paused.store(now, Ordering::SeqCst);
        now
    }
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }
    /// Request a graceful stop: the trainer finishes the current step, saves a checkpoint and exits.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);
    }
    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
    /// Request a checkpoint at the end of the current step.
    pub fn request_checkpoint(&self) {
        self.checkpoint.store(true, Ordering::SeqCst);
    }
    pub fn take_checkpoint_request(&self) -> bool {
        self.checkpoint.swap(false, Ordering::SeqCst)
    }
}
