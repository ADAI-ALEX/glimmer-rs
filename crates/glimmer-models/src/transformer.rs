//! A decoder-only transformer language model.
//!
//! Architecture (pre-norm, Llama-style, sized for small-scale training):
//!
//! ```text
//! tokens ─▶ embedding ─▶ dropout ─▶ [ Block × n_layers ] ─▶ norm ─▶ LM head (tied) ─▶ logits
//!
//! Block:  x = x + dropout(Wo · attention(RoPE(Wq·n(x)), RoPE(Wk·n(x)), Wv·n(x)))
//!         x = x + dropout(W_down · (silu(W_gate·n(x)) ⊙ W_up·n(x)))
//! ```
//!
//! * **RMSNorm** (or LayerNorm) before each sub-layer for stable training without warm-up tricks.
//! * **Rotary position embeddings** (rotate-half form) applied to queries and keys, supporting a
//!   position offset (KV-cached decoding) and explicit per-token positions (packed SFT batches).
//! * **Causal self-attention** through Burn's fused attention op. During training, autodiff uses
//!   Burn's decomposed reference attention, which has a full backward pass.
//! * **SwiGLU** (or GELU) feed-forward network.
//! * **Tied embeddings**: the LM head reuses the token-embedding matrix, saving `vocab × d_model`
//!   parameters — a large fraction of a small model.
//! * Weights are initialised on the host with GLIMMER's seeded RNG, so a seed yields identical
//!   starting weights on every backend.

use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
use burn::tensor::activation::{gelu, silu};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use glimmer_core::config::{FfnKind, InitStrategy, ModelConfig, NormKind};
use glimmer_core::Rng;

/// RMSNorm (`beta == None`) or LayerNorm over the last dimension.
#[derive(Module, Debug)]
pub struct Norm {
    pub gamma: Param<Tensor<1>>,
    pub beta: Option<Param<Tensor<1>>>,
    eps: f64,
}

impl Norm {
    fn new(kind: NormKind, d: usize, eps: f64, device: &Device) -> Self {
        let ones = Tensor::<1>::ones([d], device);
        let beta = match kind {
            NormKind::RmsNorm => None,
            NormKind::LayerNorm => Some(Param::from_tensor(Tensor::<1>::zeros([d], device))),
        };
        Self {
            gamma: Param::from_tensor(ones),
            beta,
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gamma = self.gamma.val().unsqueeze::<3>();
        match &self.beta {
            None => {
                let ms = x.clone().powi_scalar(2).mean_dim(2);
                x / ms.add_scalar(self.eps).sqrt() * gamma
            }
            Some(beta) => {
                let mean = x.clone().mean_dim(2);
                let centred = x - mean;
                let var = centred.clone().powi_scalar(2).mean_dim(2);
                centred / var.add_scalar(self.eps).sqrt() * gamma + beta.val().unsqueeze::<3>()
            }
        }
    }
}

/// Per-layer cached keys and values: `[batch, heads, len, head_dim]`.
#[derive(Debug, Clone)]
pub struct LayerKv {
    pub k: Tensor<4>,
    pub v: Tensor<4>,
}

/// Key/value cache for incremental decoding.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<Option<LayerKv>>,
    len: usize,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: vec![None; n_layers],
            len: 0,
        }
    }

    /// Number of positions currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| *l = None);
        self.len = 0;
    }

    /// Keep only the first `n` cached positions (used to reuse a shared prompt prefix).
    pub fn truncate(&mut self, n: usize) {
        if n >= self.len {
            return;
        }
        if n == 0 {
            self.clear();
            return;
        }
        for kv in self.layers.iter_mut().flatten() {
            let [b, h, _, dh] = kv.k.dims();
            kv.k = kv.k.clone().slice([0..b, 0..h, 0..n, 0..dh]);
            kv.v = kv.v.clone().slice([0..b, 0..h, 0..n, 0..dh]);
        }
        self.len = n;
    }
}

/// Where queries and keys sit in the sequence.
enum Positions {
    /// Contiguous positions `offset .. offset + T` for every row.
    Offset(usize),
    /// Explicit `[B, T]` positions (packed sequences restart at 0 per segment).
    Explicit(Tensor<2, Int>),
}

fn rotate(x: Tensor<4>, cos: &Tensor<4>, sin: &Tensor<4>) -> Tensor<4> {
    let [b, h, t, dh] = x.dims();
    let half = dh / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..t, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..t, half..dh]);
    let r1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let r2 = x1 * sin.clone() + x2 * cos.clone();
    Tensor::cat(vec![r1, r2], 3)
}

#[derive(Module, Debug)]
pub struct Attention {
    pub qkv: Linear,
    pub out: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: Option<Tensor<4, Bool>>,
        cache: Option<&mut Option<LayerKv>>,
    ) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);
        let qkv = self.qkv.forward(x);
        let heads = |i: usize| {
            qkv.clone()
                .slice([0..b, 0..t, i * d..(i + 1) * d])
                .reshape([b, t, h, dh])
                .swap_dims(1, 2)
        };
        let q = rotate(heads(0), cos, sin);
        let k = rotate(heads(1), cos, sin);
        let v = heads(2);
        let (k, v) = match cache {
            Some(slot) => {
                let (k, v) = match slot.take() {
                    Some(prev) => (
                        Tensor::cat(vec![prev.k, k], 2),
                        Tensor::cat(vec![prev.v, v], 2),
                    ),
                    None => (k, v),
                };
                *slot = Some(LayerKv { k: k.clone(), v: v.clone() });
                (k, v)
            }
            None => (k, v),
        };
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            // Causal masking is bottom-right aligned, which is exactly right when the keys
            // include a cached prefix. Packed batches carry their own (causal ∧ same-segment) mask.
            is_causal: mask.is_none(),
        };
        let y = attention(q, k, v, mask, None, options);
        self.out.forward(y.swap_dims(1, 2).reshape([b, t, d]))
    }
}

#[derive(Module, Debug)]
pub struct FeedForward {
    pub gate: Option<Linear>,
    pub up: Linear,
    pub down: Linear,
}

impl FeedForward {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = match &self.gate {
            Some(gate) => silu(gate.forward(x.clone())) * self.up.forward(x),
            None => gelu(self.up.forward(x)),
        };
        self.down.forward(hidden)
    }
}

#[derive(Module, Debug)]
pub struct Block {
    pub norm1: Norm,
    pub attn: Attention,
    pub norm2: Norm,
    pub ffn: FeedForward,
    dropout: Dropout,
}

impl Block {
    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: Option<Tensor<4, Bool>>,
        cache: Option<&mut Option<LayerKv>>,
    ) -> Tensor<3> {
        let a = self.attn.forward(self.norm1.forward(x.clone()), cos, sin, mask, cache);
        let x = x + self.dropout.forward(a);
        let f = self.ffn.forward(self.norm2.forward(x.clone()));
        x + self.dropout.forward(f)
    }
}

#[derive(Module, Debug)]
pub struct Transformer {
    pub embed: Embedding,
    pub blocks: Vec<Block>,
    pub norm: Norm,
    pub head: Option<Linear>,
    dropout: Dropout,
    rope_cos: Tensor<2>,
    rope_sin: Tensor<2>,
    vocab_size: usize,
    context_length: usize,
    n_heads: usize,
}

/// Host-side initialiser that produces identical weights on every backend.
struct Init<'a> {
    rng: Rng,
    device: &'a Device,
}

impl Init<'_> {
    fn normal(&mut self, shape: [usize; 2], std: f64) -> Tensor<2> {
        let n = shape[0] * shape[1];
        let v: Vec<f32> = (0..n).map(|_| (self.rng.normal() * std) as f32).collect();
        Tensor::from_data(TensorData::new(v, shape), self.device)
    }

    fn xavier(&mut self, shape: [usize; 2]) -> Tensor<2> {
        let a = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
        let n = shape[0] * shape[1];
        let v: Vec<f32> = (0..n).map(|_| self.rng.uniform(-a, a) as f32).collect();
        Tensor::from_data(TensorData::new(v, shape), self.device)
    }

    fn linear(&mut self, d_in: usize, d_out: usize, cfg: &ModelConfig, residual: bool) -> Linear {
        let mut lin = LinearConfig::new(d_in, d_out)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(self.device);
        let w = match cfg.init {
            InitStrategy::Xavier => self.xavier([d_in, d_out]),
            InitStrategy::Normal => self.normal([d_in, d_out], cfg.init_std),
            InitStrategy::Scaled => {
                let std = if residual {
                    cfg.init_std / (2.0 * cfg.n_layers as f64).sqrt()
                } else {
                    cfg.init_std
                };
                self.normal([d_in, d_out], std)
            }
        };
        lin.weight = Param::from_tensor(w);
        lin
    }
}

fn rope_tables(cfg: &ModelConfig, device: &Device) -> (Tensor<2>, Tensor<2>) {
    let dh = cfg.head_dim();
    let half = dh / 2;
    let n = cfg.context_length;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for p in 0..n {
        for i in 0..half {
            let inv_freq = 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / dh as f64);
            let angle = p as f64 * inv_freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (
        Tensor::from_data(TensorData::new(cos, [n, half]), device),
        Tensor::from_data(TensorData::new(sin, [n, half]), device),
    )
}

impl Transformer {
    /// Randomly initialise a model. The same `(cfg, seed)` produces the same weights on any device.
    pub fn init(cfg: &ModelConfig, device: &Device, seed: u64) -> Self {
        cfg.validate().expect("invalid model config");
        let mut init = Init {
            rng: Rng::new(seed),
            device,
        };
        let d = cfg.d_model;
        let mut embed = EmbeddingConfig::new(cfg.vocab_size, d)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let emb_std = if cfg.init == InitStrategy::Xavier { cfg.init_std } else { cfg.init_std };
        embed.weight = Param::from_tensor(init.normal([cfg.vocab_size, d], emb_std));
        let blocks = (0..cfg.n_layers)
            .map(|_| Block {
                norm1: Norm::new(cfg.norm, d, cfg.norm_eps, device),
                attn: Attention {
                    qkv: init.linear(d, 3 * d, cfg, false),
                    out: init.linear(d, d, cfg, true),
                    n_heads: cfg.n_heads,
                    head_dim: cfg.head_dim(),
                },
                norm2: Norm::new(cfg.norm, d, cfg.norm_eps, device),
                ffn: FeedForward {
                    gate: match cfg.ffn {
                        FfnKind::SwiGlu => Some(init.linear(d, cfg.d_ff, cfg, false)),
                        FfnKind::Gelu => None,
                    },
                    up: init.linear(d, cfg.d_ff, cfg, false),
                    down: init.linear(cfg.d_ff, d, cfg, true),
                },
                dropout: DropoutConfig::new(cfg.dropout).init(),
            })
            .collect();
        let head = (!cfg.tie_embeddings).then(|| init.linear(d, cfg.vocab_size, cfg, false));
        let (rope_cos, rope_sin) = rope_tables(cfg, device);
        Self {
            embed,
            blocks,
            norm: Norm::new(cfg.norm, d, cfg.norm_eps, device),
            head,
            dropout: DropoutConfig::new(cfg.dropout).init(),
            rope_cos,
            rope_sin,
            vocab_size: cfg.vocab_size,
            context_length: cfg.context_length,
            n_heads: cfg.n_heads,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn context_length(&self) -> usize {
        self.context_length
    }

    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.blocks.len())
    }

    fn rope(&self, positions: &Positions, b: usize, t: usize) -> (Tensor<4>, Tensor<4>) {
        let half = self.rope_cos.dims()[1];
        match positions {
            Positions::Offset(off) => {
                assert!(
                    off + t <= self.context_length,
                    "sequence of {} tokens at offset {off} exceeds context length {}",
                    t,
                    self.context_length
                );
                let cos = self.rope_cos.clone().slice([*off..off + t, 0..half]);
                let sin = self.rope_sin.clone().slice([*off..off + t, 0..half]);
                (cos.unsqueeze::<4>(), sin.unsqueeze::<4>())
            }
            Positions::Explicit(pos) => {
                let flat = pos.clone().reshape([b * t]);
                let cos = self.rope_cos.clone().select(0, flat.clone()).reshape([b, 1, t, half]);
                let sin = self.rope_sin.clone().select(0, flat).reshape([b, 1, t, half]);
                (cos, sin)
            }
        }
    }

    fn hidden(
        &self,
        tokens: Tensor<2, Int>,
        positions: Positions,
        mask: Option<Tensor<4, Bool>>,
        mut cache: Option<&mut KvCache>,
    ) -> Tensor<3> {
        let [b, t] = tokens.dims();
        let (cos, sin) = self.rope(&positions, b, t);
        let mut x = self.dropout.forward(self.embed.forward(tokens));
        for (i, block) in self.blocks.iter().enumerate() {
            let slot = cache.as_deref_mut().map(|c| &mut c.layers[i]);
            x = block.forward(x, &cos, &sin, mask.clone(), slot);
        }
        if let Some(c) = cache {
            c.len += t;
        }
        self.norm.forward(x)
    }

    /// Project hidden states to vocabulary logits.
    fn logits(&self, h: Tensor<3>) -> Tensor<3> {
        let [b, t, d] = h.dims();
        match &self.head {
            Some(head) => head.forward(h),
            None => h
                .reshape([b * t, d])
                .matmul(self.embed.weight.val().transpose())
                .reshape([b, t, self.vocab_size]),
        }
    }

    /// Causal forward pass over `[B, T]` tokens at positions `0..T`. Returns `[B, T, V]` logits.
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let h = self.hidden(tokens, Positions::Offset(0), None, None);
        self.logits(h)
    }

    /// Forward pass over packed sequences: tokens attend only to earlier tokens of the same
    /// segment, and positions restart at 0 for every segment.
    pub fn forward_packed(
        &self,
        tokens: Tensor<2, Int>,
        positions: Tensor<2, Int>,
        segments: Tensor<2, Int>,
    ) -> Tensor<3> {
        let mask = packed_mask(segments, self.n_heads);
        let h = self.hidden(tokens, Positions::Explicit(positions), Some(mask), None);
        self.logits(h)
    }

    /// Incremental forward pass that appends to (and reads from) a KV cache. Only the new tokens
    /// are computed. With `last_only`, returns logits for the final position only (`[B, 1, V]`).
    pub fn forward_cached(&self, tokens: Tensor<2, Int>, cache: &mut KvCache, last_only: bool) -> Tensor<3> {
        let offset = cache.len;
        let h = self.hidden(tokens, Positions::Offset(offset), None, Some(cache));
        let h = if last_only {
            let [b, t, d] = h.dims();
            h.slice([0..b, t - 1..t, 0..d])
        } else {
            h
        };
        self.logits(h)
    }
}

/// Attention mask for packed sequences: `true` (masked) where the key is in the future or in a
/// different segment. Shape `[B, heads, T, T]`.
pub fn packed_mask(segments: Tensor<2, Int>, n_heads: usize) -> Tensor<4, Bool> {
    let [b, t] = segments.dims();
    let device = segments.device();
    let q = segments.clone().reshape([b, t, 1]).expand([b, t, t]);
    let k = segments.reshape([b, 1, t]).expand([b, t, t]);
    let other_segment = q.not_equal(k);
    let rows = Tensor::<1, Int>::arange(0..t as i64, &device).reshape([t, 1]).expand([t, t]);
    let cols = Tensor::<1, Int>::arange(0..t as i64, &device).reshape([1, t]).expand([t, t]);
    let future = cols.greater(rows).reshape([1, t, t]).expand([b, t, t]);
    other_segment
        .bool_or(future)
        .reshape([b, 1, t, t])
        .expand([b, n_heads, t, t])
}
