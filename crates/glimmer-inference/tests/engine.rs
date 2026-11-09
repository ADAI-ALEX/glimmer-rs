//! Inference engine tests on the CPU backend.

use glimmer_core::config::ModelConfig;
use glimmer_inference::{ChatSession, Engine, SamplingConfig, StopReason};
use glimmer_models::burn::tensor::{Int, Tensor, TensorData};
use glimmer_models::{Transformer, backend};
use glimmer_tokenizer::Tokenizer;
use glimmer_tokenizer::chat::Message;

fn engine(seed: u64) -> Engine {
    let tok = Tokenizer::bytes_only();
    let cfg = ModelConfig {
        vocab_size: tok.vocab_size(),
        context_length: 64,
        ..ModelConfig::preset("micro").unwrap()
    };
    let dev = backend::select(glimmer_core::BackendKind::Cpu).unwrap();
    let model = Transformer::init(&cfg, &dev.device, seed);
    Engine::from_parts(model, tok, cfg, dev, "test", std::path::Path::new("."), None)
}

#[test]
fn greedy_generation_matches_full_forward_argmax() {
    // KV-cached greedy decoding must pick exactly the argmax of a full (uncached) forward pass
    // at every step.
    let mut e = engine(5);
    let prompt: Vec<u32> = e.tokenizer.encode("G'day");
    let cfg = SamplingConfig { max_new_tokens: 12, ..SamplingConfig::greedy() };
    let out = e.generate(&prompt, &cfg, |_| true).unwrap();
    let mut seq = prompt.clone();
    for &tok in &out.tokens {
        let data: Vec<i64> = seq.iter().map(|&t| t as i64).collect();
        let x = Tensor::<2, Int>::from_data(TensorData::new(data, [1, seq.len()]), &e.device.device);
        let logits = e.model.forward(x).into_data().try_to_vec::<f32>().unwrap();
        let v = e.config.vocab_size;
        let last = &logits[(seq.len() - 1) * v..];
        let mut masked = last.to_vec();
        for b in [0usize, 6, 7] {
            masked[b] = f32::NEG_INFINITY;
        }
        let expected = glimmer_inference::sampling::argmax(&masked);
        assert_eq!(tok, expected, "cached decode diverged from full forward");
        seq.push(tok);
    }
}

#[test]
fn respects_max_tokens_and_reports_stats() {
    let mut e = engine(6);
    let prompt = e.tokenizer.encode("hello");
    let cfg = SamplingConfig { max_new_tokens: 5, seed: 1, ..Default::default() };
    let out = e.generate(&prompt, &cfg, |_| true).unwrap();
    assert!(out.tokens.len() <= 5);
    if out.stop == StopReason::MaxTokens {
        assert_eq!(out.tokens.len(), 5);
    }
    assert_eq!(out.stats.prompt_tokens, prompt.len());
    assert!(out.stats.total_ms > 0.0);
}

#[test]
fn seeded_generation_is_reproducible() {
    let prompt = Tokenizer::bytes_only().encode("Reckon");
    let cfg = SamplingConfig { max_new_tokens: 10, seed: 42, ..Default::default() };
    let a = engine(7).generate(&prompt, &cfg, |_| true).unwrap();
    let b = engine(7).generate(&prompt, &cfg, |_| true).unwrap();
    assert_eq!(a.tokens, b.tokens);
}

#[test]
fn cancellation_stops_generation() {
    let mut e = engine(8);
    let prompt = e.tokenizer.encode("x");
    let cfg = SamplingConfig { max_new_tokens: 50, seed: 3, temperature: 1.0, ..Default::default() };
    let mut pieces = 0;
    let out = e
        .generate(&prompt, &cfg, |_| {
            pieces += 1;
            pieces < 2
        })
        .unwrap();
    if !out.tokens.is_empty() && out.stop != StopReason::EndOfTurn {
        assert_eq!(out.stop, StopReason::Cancelled);
    }
}

#[test]
fn multi_turn_chat_reuses_kv_cache() {
    let mut e = engine(9);
    let mut s = ChatSession::new("test", Some("SYS".into()), SamplingConfig {
        max_new_tokens: 8,
        seed: 11,
        ..Default::default()
    });
    let (g1, _) = s.respond(&mut e, "first", |_| true).unwrap();
    assert_eq!(g1.stats.reused_tokens, 0);
    let (g2, _) = s.respond(&mut e, "second", |_| true).unwrap();
    // The system prompt and the whole first exchange should come from the cache.
    assert!(g2.stats.reused_tokens >= 3, "expected cache reuse, got {}", g2.stats.reused_tokens);
    assert_eq!(s.messages.len(), 4);
    assert_eq!(s.messages[2], Message::user("second"));
}

#[test]
fn reused_cache_gives_same_result_as_fresh_cache() {
    let prompt_a = Tokenizer::bytes_only().encode("shared prefix then A");
    let prompt_b = Tokenizer::bytes_only().encode("shared prefix then B");
    let cfg = SamplingConfig { max_new_tokens: 8, ..SamplingConfig::greedy() };
    let mut warm = engine(10);
    warm.generate(&prompt_a, &cfg, |_| true).unwrap();
    let reused = warm.generate(&prompt_b, &cfg, |_| true).unwrap();
    assert!(reused.stats.reused_tokens > 0);
    let fresh = engine(10).generate(&prompt_b, &cfg, |_| true).unwrap();
    assert_eq!(reused.tokens, fresh.tokens, "prefix reuse changed the output");
}
