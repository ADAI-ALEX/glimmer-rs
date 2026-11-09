//! Sampling hooks used during training (evaluation commands are added alongside).

use anyhow::Result;
use glimmer_core::Workspace;
use glimmer_core::config::Stage;
use glimmer_training::Sampler;

pub fn training_sampler(_ws: &Workspace, stage: Stage) -> Result<Sampler> {
    Ok(glimmer_evaluation::training_sampler(stage))
}
