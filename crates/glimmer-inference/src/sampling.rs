//! Token sampling: temperature, top-k, top-p (nucleus) and repetition penalty.

use glimmer_core::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplingConfig {
    /// 0 = greedy (argmax).
    pub temperature: f32,
    /// Keep only the `k` most likely tokens (0 = disabled).
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probability mass reaches `p` (1.0 = disabled).
    pub top_p: f32,
    /// Divide positive logits (multiply negative ones) of recently generated tokens by this
    /// factor (CTRL-style). 1.0 = disabled.
    pub repetition_penalty: f32,
    /// How many recent tokens the repetition penalty considers.
    pub repetition_window: usize,
    pub max_new_tokens: usize,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.92,
            repetition_penalty: 1.15,
            repetition_window: 64,
            max_new_tokens: 160,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        }
    }
}

/// Apply the repetition penalty in place to logits of tokens seen in `recent`.
pub fn apply_repetition_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    for &t in recent {
        if seen.insert(t) {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
            }
        }
    }
}

/// Choose the next token from one position's logits.
pub fn sample(logits: &[f32], recent: &[u32], cfg: &SamplingConfig, rng: &mut Rng, banned: &[u32]) -> u32 {
    let mut logits = logits.to_vec();
    for &b in banned {
        if let Some(l) = logits.get_mut(b as usize) {
            *l = f32::NEG_INFINITY;
        }
    }
    let window = &recent[recent.len().saturating_sub(cfg.repetition_window)..];
    apply_repetition_penalty(&mut logits, window, cfg.repetition_penalty);

    if cfg.temperature <= 0.0 {
        return argmax(&logits);
    }
    let mut idx: Vec<usize> = (0..logits.len()).filter(|&i| logits[i].is_finite()).collect();
    if idx.is_empty() {
        return argmax(&logits);
    }
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    if cfg.top_k > 0 && idx.len() > cfg.top_k {
        idx.truncate(cfg.top_k);
    }
    let max = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - max) / cfg.temperature).exp())
        .collect();
    let total: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= total);
    if cfg.top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= cfg.top_p {
                keep = i + 1;
                break;
            }
        }
        probs.truncate(keep);
        idx.truncate(keep);
        let total: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);
    }
    let r = rng.next_f32();
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return idx[i] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut rng = Rng::new(1);
        assert_eq!(sample(&[0.1, 2.0, 1.0], &[], &SamplingConfig::greedy(), &mut rng, &[]), 1);
    }

    #[test]
    fn banned_tokens_never_sampled() {
        let mut rng = Rng::new(1);
        let cfg = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, ..Default::default() };
        for _ in 0..200 {
            assert_ne!(sample(&[5.0, 0.0, 0.0], &[], &cfg, &mut rng, &[0]), 0);
        }
    }

    #[test]
    fn top_k_restricts_support() {
        let mut rng = Rng::new(2);
        let cfg = SamplingConfig { temperature: 1.0, top_k: 2, top_p: 1.0, repetition_penalty: 1.0, ..Default::default() };
        let logits = [3.0, 2.9, 2.8, 2.7];
        for _ in 0..500 {
            assert!(sample(&logits, &[], &cfg, &mut rng, &[]) < 2);
        }
    }

    #[test]
    fn top_p_restricts_to_nucleus() {
        let mut rng = Rng::new(3);
        let cfg = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 0.5, repetition_penalty: 1.0, ..Default::default() };
        // p ≈ [0.84, 0.11, 0.04, ...]: nucleus at 0.5 is just token 0.
        let logits = [4.0, 2.0, 1.0, 0.0];
        for _ in 0..200 {
            assert_eq!(sample(&logits, &[], &cfg, &mut rng, &[]), 0);
        }
    }

    #[test]
    fn temperature_controls_spread() {
        let logits = [1.0, 0.0];
        let count = |t: f32| {
            let mut rng = Rng::new(4);
            let cfg = SamplingConfig { temperature: t, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, ..Default::default() };
            (0..5000).filter(|_| sample(&logits, &[], &cfg, &mut rng, &[]) == 0).count()
        };
        let cold = count(0.3);
        let warm = count(2.0);
        assert!(cold > warm, "lower temperature should concentrate mass ({cold} vs {warm})");
        // At T=1, p(0) = e/(1+e) ≈ 0.731.
        let t1 = count(1.0) as f64 / 5000.0;
        assert!((t1 - 0.731).abs() < 0.03, "{t1}");
    }

    #[test]
    fn repetition_penalty_discourages_recent_tokens() {
        let mut l = vec![2.0, 2.0, -1.0];
        apply_repetition_penalty(&mut l, &[0, 2, 0], 2.0);
        assert_eq!(l, vec![1.0, 2.0, -2.0]);
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        let cfg = SamplingConfig::default();
        let run = || {
            let mut rng = Rng::new(99);
            (0..20).map(|_| sample(&logits, &[], &cfg, &mut rng, &[])).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
