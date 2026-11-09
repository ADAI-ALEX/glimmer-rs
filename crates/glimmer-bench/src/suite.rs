//! The benchmark suite.

use crate::{Measurement, measure};
use anyhow::Result;
use glimmer_core::{ModelConfig, Rng};
use glimmer_inference::{Engine, SamplingConfig};
use glimmer_models::burn::module::Module;
use glimmer_models::burn::optim::GradientsParams;
use glimmer_models::burn::tensor::module::attention;
use glimmer_models::burn::tensor::ops::AttentionModuleOptions;
use glimmer_models::burn::tensor::{Device, Distribution, Int, Tensor, TensorData};
use glimmer_models::loss::weighted_nll_sum;
use glimmer_models::{ComputeDevice, Transformer, store};
use glimmer_tokenizer::Tokenizer;
use std::path::Path;

fn sync1(t: Tensor<1>) {
    let _ = t.into_scalar::<f32>();
}

fn random_tokens(b: usize, t: usize, vocab: usize, device: &Device, seed: u64) -> Tensor<2, Int> {
    let mut rng = Rng::new(seed);
    let v: Vec<i64> = (0..b * t).map(|_| rng.below(vocab) as i64).collect();
    Tensor::from_data(TensorData::new(v, [b, t]), device)
}

pub struct Ctx<'a> {
    pub device: &'a ComputeDevice,
    pub warmup: usize,
    pub iterations: usize,
}

impl Ctx<'_> {
    fn m(&self, name: &str, config: String, stats: (f64, f64, f64), work: Option<(f64, &str)>) -> Measurement {
        Measurement {
            name: name.into(),
            backend: self.device.label(),
            config,
            warmup: self.warmup,
            iterations: self.iterations,
            mean_ms: stats.0,
            std_ms: stats.1,
            min_ms: stats.2,
            work: work.map(|(w, u)| (w, u.to_string())),
        }
    }
}

/// Square matrix multiplication throughput.
pub fn matmul(c: &Ctx, n: usize) -> Measurement {
    let d = &c.device.device;
    let a = Tensor::<2>::random([n, n], Distribution::Default, d);
    let b = Tensor::<2>::random([n, n], Distribution::Default, d);
    let stats = measure(c.warmup, c.iterations, || sync1(a.clone().matmul(b.clone()).sum()));
    c.m("matmul", format!("{n}×{n}·{n}×{n} f32"), stats, Some((2.0 * (n as f64).powi(3) / 1e9, "GFLOP")))
}

/// Causal scaled-dot-product attention (forward, no autodiff).
pub fn attention_fwd(c: &Ctx, b: usize, h: usize, t: usize, dh: usize) -> Measurement {
    let d = &c.device.device;
    let q = Tensor::<4>::random([b, h, t, dh], Distribution::Default, d);
    let k = Tensor::<4>::random([b, h, t, dh], Distribution::Default, d);
    let v = Tensor::<4>::random([b, h, t, dh], Distribution::Default, d);
    let opts = AttentionModuleOptions { scale: None, softcap: None, is_causal: true };
    let stats = measure(c.warmup, c.iterations, || {
        sync1(attention(q.clone(), k.clone(), v.clone(), None, None, opts).sum())
    });
    let flops = 4.0 * (b * h) as f64 * (t * t) as f64 * dh as f64 / 2.0 / 1e9;
    c.m("attention (causal, fwd)", format!("B{b} H{h} T{t} Dh{dh}"), stats, Some((flops, "GFLOP")))
}

/// Model forward pass (inference mode).
pub fn forward(c: &Ctx, cfg: &ModelConfig, batch: usize) -> Measurement {
    let d = &c.device.device;
    let model = Transformer::init(cfg, d, 1);
    let x = random_tokens(batch, cfg.context_length, cfg.vocab_size, d, 2);
    let stats = measure(c.warmup, c.iterations, || sync1(model.forward(x.clone()).sum()));
    let tokens = (batch * cfg.context_length) as f64;
    c.m("forward", format!("{} B{batch} T{}", cfg.name, cfg.context_length), stats, Some((tokens, "tok")))
}

/// Full training step: forward, loss, backward, AdamW update.
pub fn train_step(c: &Ctx, cfg: &ModelConfig, batch: usize) -> Measurement {
    let d = c.device.device.clone().autodiff();
    let mut model = Transformer::init(cfg, &d, 1);
    let mut opt = glimmer_training::optim::build(&glimmer_core::OptimizerConfig::default());
    let t = cfg.context_length;
    let x = random_tokens(batch, t, cfg.vocab_size, &d, 3);
    let y = random_tokens(batch, t, cfg.vocab_size, &d, 4);
    let w = Tensor::<2>::ones([batch, t], &d);
    let mut step = || {
        let loss = weighted_nll_sum(model.forward(x.clone()), y.clone(), w.clone()).div_scalar((batch * t) as f64);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let _ = loss.into_scalar::<f32>();
        model = opt.step(3e-4, model.clone(), grads);
    };
    let stats = measure(c.warmup, c.iterations, &mut step);
    let tokens = (batch * t) as f64;
    let mut m = c.m(
        "train step (fwd+bwd+AdamW)",
        format!("{} {}M params B{batch} T{t}", cfg.name, cfg.param_count() as f64 / 1e6),
        stats,
        Some((tokens, "tok")),
    );
    m.name = "train step (fwd+bwd+AdamW)".into();
    m
}

/// Autoregressive decoding with and without the KV cache.
pub fn decode(c: &Ctx, cfg: &ModelConfig, new_tokens: usize) -> Result<(Measurement, Measurement)> {
    let d = &c.device.device;
    let tok = Tokenizer::bytes_only();
    let cfg = ModelConfig { vocab_size: tok.vocab_size().max(cfg.vocab_size), ..cfg.clone() };
    let model = Transformer::init(&cfg, d, 5);
    let mut engine = Engine::from_parts(model.clone(), tok, cfg.clone(), c.device.clone(), "bench", Path::new("."), None);
    let prompt: Vec<u32> = (0..64).map(|i| 100 + (i % 50)).collect();
    let sampling = SamplingConfig { max_new_tokens: new_tokens, temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 1, repetition_window: 0 };
    // Ban end-of-turn tokens by measuring a fixed number of steps directly.
    let cached = measure(c.warmup.min(2), c.iterations.min(5), || {
        engine.reset_cache();
        let mut cache = model.new_cache();
        let p: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
        let mut x = Tensor::<2, Int>::from_data(TensorData::new(p, [1, prompt.len()]), d);
        for _ in 0..new_tokens {
            let logits = model.forward_cached(x, &mut cache, true);
            let next = logits.argmax(2).reshape([1, 1]);
            x = next;
        }
        let _ = x.into_data();
    });
    let _ = &mut engine;
    let _ = sampling;
    let uncached = measure(c.warmup.min(2), c.iterations.min(5), || {
        let mut seq: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
        for _ in 0..new_tokens {
            let x = Tensor::<2, Int>::from_data(TensorData::new(seq.clone(), [1, seq.len()]), d);
            let logits = model.forward(x);
            let [_, t, _] = logits.dims();
            let next = logits.slice([0..1, t - 1..t]).argmax(2).into_scalar::<i64>();
            seq.push(next);
        }
    });
    let label = format!("{} prompt 64 + {new_tokens} new", cfg.name);
    Ok((
        c.m("decode (KV cache)", label.clone(), cached, Some((new_tokens as f64, "tok"))),
        c.m("decode (no cache)", label, uncached, Some((new_tokens as f64, "tok"))),
    ))
}

/// Host-side batch sampling from a memory-mapped token file.
pub fn data_loading(c: &Ctx, path: &Path, batch: usize, seq: usize) -> Result<Measurement> {
    let b = glimmer_data::PretrainBatcher::open(path, seq)?;
    let mut rng = Rng::new(1);
    let stats = measure(c.warmup, c.iterations * 10, || {
        let batch = b.sample(batch, &mut rng);
        std::hint::black_box(batch);
    });
    let mut m = c.m("data loading (pretrain batch)", format!("B{batch} T{seq} mmap"), stats, Some(((batch * seq) as f64, "tok")));
    m.backend = "host".into();
    Ok(m)
}

/// Checkpoint save and load of model weights.
pub fn checkpoint_io(c: &Ctx, cfg: &ModelConfig, dir: &Path) -> Result<(Measurement, Measurement)> {
    let d = &c.device.device;
    let model = Transformer::init(cfg, d, 6);
    let path = dir.join("bench-model.bpk");
    let save = measure(1, 3, || store::save_weights(&model, &path).unwrap());
    let load = measure(1, 3, || {
        let m = store::load_weights(cfg, d, &path).unwrap();
        std::hint::black_box(m.num_params());
    });
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) as f64 / 1e6;
    std::fs::remove_file(&path).ok();
    let label = format!("{} ({bytes:.0} MB)", cfg.name);
    Ok((
        c.m("checkpoint save (weights)", label.clone(), save, Some((bytes, "MB"))),
        c.m("checkpoint load (weights)", label, load, Some((bytes, "MB"))),
    ))
}

/// Tokenizer encoding throughput on real text.
pub fn tokenizer(c: &Ctx, tok: &Tokenizer, text: &str) -> Measurement {
    let stats = measure(1, c.iterations, || {
        std::hint::black_box(tok.encode(text));
    });
    let mut m = c.m("tokenizer encode", format!("{} KB text, vocab {}", text.len() / 1000, tok.vocab_size()), stats, Some((text.len() as f64 / 1e6, "MB")));
    m.backend = "host".into();
    m
}
