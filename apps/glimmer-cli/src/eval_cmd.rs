//! Sampling hooks used during training (evaluation commands are added alongside).

use anyhow::Result;
use glimmer_core::config::Stage;
use glimmer_core::Workspace;
use glimmer_inference::SamplingConfig;
use glimmer_inference::engine::generate_once;
use glimmer_tokenizer::chat::{Message, render_prompt};
use glimmer_training::Sampler;

/// Prompts sampled at every evaluation so progress is visible as genuine model output.
pub fn training_sampler(_ws: &Workspace, stage: Stage) -> Result<Sampler> {
    let cfg = SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.15,
        max_new_tokens: 48,
        seed: 7,
        repetition_window: 64,
    };
    Ok(Box::new(move |model, tok, device| {
        let prompts: Vec<(String, Vec<u32>)> = match stage {
            Stage::Pretrain => ["Once upon a time", "The kangaroo is", "In Australia,"]
                .iter()
                .map(|p| (p.to_string(), tok.encode(p)))
                .collect(),
            Stage::Sft => ["How ya going, mate?", "What should I learn first in Rust?"]
                .iter()
                .map(|p| (p.to_string(), render_prompt(tok, &[Message::user(*p)])))
                .collect(),
        };
        prompts
            .into_iter()
            .filter_map(|(label, ids)| generate_once(model, tok, device, &ids, &cfg).ok().map(|(_, text, _)| (label, text)))
            .collect()
    }))
}
