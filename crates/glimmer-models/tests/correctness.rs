//! Neural-network correctness tests for the GLIMMER transformer.
//!
//! These run on the pure-Rust CPU backend so they work everywhere (including CI). The Metal
//! comparison test runs only when a Metal device is available.

use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Device, Int, Tensor, TensorData};
use glimmer_core::config::{FfnKind, InitStrategy, ModelConfig, NormKind};
use glimmer_models::loss::{log_softmax_stable, weighted_nll_sum};
use glimmer_models::store::{load_weights, save_weights};
use glimmer_models::transformer::Transformer;

fn micro() -> ModelConfig {
    ModelConfig {
        vocab_size: 97,
        context_length: 32,
        ..ModelConfig::preset("micro").unwrap()
    }
}

fn cpu() -> Device {
    Device::flex()
}

fn tokens(rows: &[Vec<i64>], device: &Device) -> Tensor<2, Int> {
    let b = rows.len();
    let t = rows[0].len();
    let flat: Vec<i64> = rows.iter().flatten().copied().collect();
    Tensor::from_data(TensorData::new(flat, [b, t]), device)
}

fn to_vec(t: Tensor<3>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn parameter_count_matches_configuration() {
    let device = cpu();
    for (norm, ffn, tie) in [
        (NormKind::RmsNorm, FfnKind::SwiGlu, true),
        (NormKind::LayerNorm, FfnKind::Gelu, false),
        (NormKind::RmsNorm, FfnKind::Gelu, true),
    ] {
        let cfg = ModelConfig {
            norm,
            ffn,
            tie_embeddings: tie,
            ..micro()
        };
        let model = Transformer::init(&cfg, &device, 1);
        assert_eq!(model.num_params(), cfg.param_count(), "{norm:?} {ffn:?} tie={tie}");
    }
}

#[test]
fn initialisation_is_deterministic_and_seed_dependent() {
    let device = cpu();
    let cfg = micro();
    let x = tokens(&[vec![1, 2, 3, 4, 5, 6]], &device);
    let a = to_vec(Transformer::init(&cfg, &device, 7).forward(x.clone()));
    let b = to_vec(Transformer::init(&cfg, &device, 7).forward(x.clone()));
    let c = to_vec(Transformer::init(&cfg, &device, 8).forward(x));
    assert_eq!(a, b);
    assert!(max_diff(&a, &c) > 1e-4);
}

#[test]
fn initial_loss_is_near_uniform_entropy() {
    // A correctly initialised LM predicts roughly uniformly: loss ≈ ln(V).
    let device = cpu();
    let cfg = micro();
    let model = Transformer::init(&cfg, &device, 3);
    let x = tokens(&[(0..16).map(|i| (i * 5 % 97) as i64).collect()], &device);
    let y = tokens(&[(0..16).map(|i| ((i * 5 + 1) % 97) as i64).collect()], &device);
    let w = Tensor::<2>::ones([1, 16], &device);
    let loss = weighted_nll_sum(model.forward(x), y, w).into_scalar::<f32>() / 16.0;
    let uniform = (cfg.vocab_size as f32).ln();
    assert!((loss - uniform).abs() < 0.3, "initial loss {loss}, ln(V) = {uniform}");
}

#[test]
fn causal_mask_hides_future_tokens() {
    let device = cpu();
    let model = Transformer::init(&micro(), &device, 11);
    let base: Vec<i64> = vec![5, 9, 14, 3, 27, 61, 8, 42, 17, 33];
    let mut changed = base.clone();
    changed[6] = 90; // perturb position 6
    let v = model.vocab_size();
    let a = to_vec(model.forward(tokens(&[base], &device)));
    let b = to_vec(model.forward(tokens(&[changed], &device)));
    // Positions 0..6 must be bit-for-bit unaffected; position 6 onwards must change.
    assert!(max_diff(&a[..6 * v], &b[..6 * v]) < 1e-6, "future token leaked into the past");
    assert!(max_diff(&a[6 * v..], &b[6 * v..]) > 1e-4);
}

#[test]
fn packed_segments_are_isolated_and_positions_restart() {
    let device = cpu();
    let model = Transformer::init(&micro(), &device, 13);
    let v = model.vocab_size();
    let s1: Vec<i64> = vec![4, 8, 15, 16, 23];
    let s2: Vec<i64> = vec![42, 7, 9];
    let alone1 = to_vec(model.forward(tokens(&[s1.clone()], &device)));
    let alone2 = to_vec(model.forward(tokens(&[s2.clone()], &device)));
    // Pack [s1 | s2 | pad pad] into one row of length 10.
    let mut row = s1.clone();
    row.extend(&s2);
    row.extend([0, 0]);
    let segs: Vec<i64> = [1, 1, 1, 1, 1, 2, 2, 2, 0, 0].to_vec();
    let pos: Vec<i64> = [0, 1, 2, 3, 4, 0, 1, 2, 0, 1].to_vec();
    let packed = to_vec(model.forward_packed(
        tokens(&[row], &device),
        tokens(&[pos], &device),
        tokens(&[segs], &device),
    ));
    assert!(max_diff(&packed[..5 * v], &alone1) < 1e-4, "segment 1 differs when packed");
    assert!(max_diff(&packed[5 * v..8 * v], &alone2) < 1e-4, "segment 2 differs when packed");
}

#[test]
fn kv_cache_matches_full_forward() {
    let device = cpu();
    let model = Transformer::init(&micro(), &device, 17);
    let seq: Vec<i64> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
    let v = model.vocab_size();
    let full = to_vec(model.forward(tokens(&[seq.clone()], &device)));
    let mut cache = model.new_cache();
    // Prefill 5 tokens, then decode one at a time.
    let mut cached = to_vec(model.forward_cached(tokens(&[seq[..5].to_vec()], &device), &mut cache, false));
    for &tok in &seq[5..] {
        cached.extend(to_vec(model.forward_cached(tokens(&[vec![tok]], &device), &mut cache, true)));
    }
    assert_eq!(cache.len(), seq.len());
    assert_eq!(cached.len(), full.len());
    assert!(max_diff(&cached, &full) < 1e-4, "KV-cached decoding diverges: {}", max_diff(&cached, &full));
    // last_only returns the final position's logits.
    let mut c2 = model.new_cache();
    let last = to_vec(model.forward_cached(tokens(&[seq.clone()], &device), &mut c2, true));
    assert!(max_diff(&last, &full[(seq.len() - 1) * v..]) < 1e-4);
}

#[test]
fn loss_matches_host_reference_and_respects_weights() {
    let device = cpu();
    let logits_v: Vec<f32> = vec![1.0, 2.0, 0.5, -1.0, 0.0, 3.0, 2.0, 2.0, 2.0];
    let logits = Tensor::<3>::from_data(TensorData::new(logits_v.clone(), [1, 3, 3]), &device);
    let targets = tokens(&[vec![1, 2, 0]], &device);
    let weights = Tensor::<2>::from_data(TensorData::new(vec![1.0f32, 0.0, 1.0], [1, 3]), &device);
    let got = weighted_nll_sum(logits, targets, weights).into_scalar::<f32>();
    let nll = |row: &[f32], t: usize| {
        let m = row.iter().cloned().fold(f32::MIN, f32::max);
        let lse = row.iter().map(|x| (x - m).exp()).sum::<f32>().ln() + m;
        lse - row[t]
    };
    let expected = nll(&logits_v[0..3], 1) + nll(&logits_v[6..9], 0);
    assert!((got - expected).abs() < 1e-5, "{got} vs {expected}");
}

#[test]
fn log_softmax_is_stable_for_extreme_logits() {
    let device = cpu();
    let x = Tensor::<2>::from_data(TensorData::new(vec![1000.0f32, -1000.0, 0.0, 999.0], [1, 4]), &device);
    let out = log_softmax_stable(x).into_data().try_to_vec::<f32>().unwrap();
    assert!(out.iter().all(|v| v.is_finite() || *v == f32::NEG_INFINITY), "{out:?}");
    assert!((out[0] - (-(1.0f32 + (-1.0f32).exp()).ln())).abs() < 1e-4);
}

/// Collect every float parameter's gradient magnitude.
struct GradCollector<'a> {
    grads: &'a burn::tensor::Gradients,
    norms: Vec<(String, f32)>,
}

impl ModuleVisitor for GradCollector<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let g = param.val().grad(self.grads);
        let n = g
            .map(|g| g.abs().sum().into_scalar::<f32>())
            .unwrap_or(-1.0);
        self.norms.push((format!("{:?}", param.id), n));
    }
}

#[test]
fn gradients_flow_to_every_parameter() {
    let device = cpu().autodiff();
    let cfg = ModelConfig {
        tie_embeddings: false,
        ..micro()
    };
    let model = Transformer::init(&cfg, &device, 19);
    let x = tokens(&[vec![1, 5, 9, 13, 17, 21, 25, 29]], &device);
    let y = tokens(&[vec![5, 9, 13, 17, 21, 25, 29, 33]], &device);
    let w = Tensor::<2>::ones([1, 8], &device);
    let loss = weighted_nll_sum(model.forward(x), y, w);
    let grads = loss.backward();
    let mut c = GradCollector { grads: &grads, norms: Vec::new() };
    model.visit(&mut c);
    assert!(!c.norms.is_empty());
    for (id, n) in &c.norms {
        assert!(*n > 0.0 && n.is_finite(), "parameter {id} has no gradient ({n})");
    }
}

#[test]
fn autodiff_gradients_match_finite_differences() {
    // Central differences on the token-embedding matrix of a micro model.
    let device = cpu().autodiff();
    let cfg = ModelConfig {
        n_layers: 1,
        ..micro()
    };
    let model = Transformer::init(&cfg, &device, 23);
    let x_rows = vec![vec![3i64, 7, 11, 2, 9, 4]];
    let y_rows = vec![vec![7i64, 11, 2, 9, 4, 1]];
    let loss_of = |m: &Transformer, dev: &Device| -> Tensor<1> {
        let w = Tensor::<2>::ones([1, 6], dev);
        weighted_nll_sum(m.forward(tokens(&x_rows, dev)), tokens(&y_rows, dev), w)
    };
    let loss = loss_of(&model, &device);
    let grads = loss.backward();
    let emb = model.embed.weight.val();
    let analytic = emb.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();
    let base: Vec<f32> = emb.clone().inner().into_data().try_to_vec::<f32>().unwrap();
    let [vocab, d] = emb.dims();

    let plain = Device::flex();
    let eps = 1e-2f32;
    // Check the coordinates with the largest analytic gradients (best signal-to-noise in f32).
    let mut idx: Vec<usize> = (0..analytic.len()).collect();
    idx.sort_by(|&a, &b| analytic[b].abs().partial_cmp(&analytic[a].abs()).unwrap());
    let mut checked = 0;
    for &i in idx.iter().take(12) {
        let eval = |delta: f32| -> f32 {
            let mut w = base.clone();
            w[i] += delta;
            let mut m = Transformer::init(&cfg, &plain, 23);
            m.embed.weight = Param::from_tensor(Tensor::from_data(TensorData::new(w, [vocab, d]), &plain));
            loss_of(&m, &plain).into_scalar::<f32>()
        };
        let numeric = (eval(eps) - eval(-eps)) / (2.0 * eps);
        let rel = (numeric - analytic[i]).abs() / analytic[i].abs().max(1e-3);
        assert!(
            rel < 0.05,
            "coordinate {i}: analytic {} vs numeric {numeric} (rel err {rel})",
            analytic[i]
        );
        checked += 1;
    }
    assert_eq!(checked, 12);
}

#[test]
fn save_and_load_roundtrip_preserves_outputs() {
    let device = cpu();
    let cfg = micro();
    let model = Transformer::init(&cfg, &device, 29);
    let dir = std::env::temp_dir().join(format!("glimmer-store-{}", std::process::id()));
    let path = dir.join("model.bpk");
    save_weights(&model, &path).unwrap();
    let loaded = load_weights(&cfg, &device, &path).unwrap();
    let x = tokens(&[vec![1, 2, 3, 4, 5]], &device);
    assert_eq!(to_vec(model.forward(x.clone())), to_vec(loaded.forward(x)));
    assert_eq!(model.embed.weight.id, loaded.embed.weight.id, "parameter ids must persist");
    // A different architecture must be rejected, not partially loaded.
    let other = ModelConfig { d_model: 32, n_heads: 4, ..cfg };
    assert!(load_weights(&other, &device, &path).is_err());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn init_strategies_produce_finite_outputs() {
    let device = cpu();
    for init in [InitStrategy::Scaled, InitStrategy::Normal, InitStrategy::Xavier] {
        let cfg = ModelConfig { init, ..micro() };
        let out = to_vec(Transformer::init(&cfg, &device, 1).forward(tokens(&[vec![1, 2, 3]], &device)));
        assert!(out.iter().all(|v| v.is_finite()), "{init:?}");
    }
}

/// Cross-backend agreement: identical weights must give (numerically) identical logits on
/// Metal and on the CPU. Skipped when Metal is unavailable (e.g. Linux CI).
#[test]
fn metal_matches_cpu_when_available() {
    let Ok(metal) = glimmer_models::backend::select(glimmer_core::BackendKind::Metal) else {
        eprintln!("Metal unavailable — skipping cross-backend test");
        return;
    };
    let cfg = micro();
    let cpu_model = Transformer::init(&cfg, &cpu(), 31);
    let gpu_model = Transformer::init(&cfg, &metal.device, 31);
    let rows = vec![vec![1i64, 4, 9, 16, 25, 36, 49, 64, 81, 3, 7, 11]];
    let a = to_vec(cpu_model.forward(tokens(&rows, &cpu())));
    let b = to_vec(gpu_model.forward(tokens(&rows, &metal.device)));
    let d = max_diff(&a, &b);
    assert!(d < 1e-3, "Metal and CPU logits differ by {d}");
}
