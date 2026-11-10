use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::activation::{log_softmax, silu};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, DeviceKind, Distribution, Int, Tensor, TensorData};
use std::time::Instant;

#[derive(Module, Debug)]
struct Block {
    n1: RmsNorm,
    qkv: Linear,
    o: Linear,
    n2: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    heads: usize,
}

#[derive(Module, Debug)]
struct Gpt {
    emb: Embedding,
    blocks: Vec<Block>,
    nf: RmsNorm,
    cos: Tensor<2>,
    sin: Tensor<2>,
}

fn lin(i: usize, o: usize, d: &Device) -> Linear {
    LinearConfig::new(i, o)
        .with_bias(false)
        .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 })
        .init(d)
}

fn rope(x: Tensor<4>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
    // x: [B,H,T,Dh]; rotate-half formulation
    let [b, h, t, dh] = x.dims();
    let half = dh / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..t, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..t, half..dh]);
    let c = cos.clone().slice([0..t, 0..half]).unsqueeze::<4>();
    let s = sin.clone().slice([0..t, 0..half]).unsqueeze::<4>();
    let r1 = x1.clone() * c.clone() - x2.clone() * s.clone();
    let r2 = x1 * s + x2 * c;
    Tensor::cat(vec![r1, r2], 3)
}

impl Block {
    fn forward(&self, x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let h = self.heads;
        let dh = d / h;
        let qkv = self.qkv.forward(self.n1.forward(x.clone()));
        let qkv = qkv.reshape([b, t, 3, h, dh]);
        let q = qkv.clone().slice([0..b, 0..t, 0..1, 0..h, 0..dh]).reshape([b, t, h, dh]).swap_dims(1, 2);
        let k = qkv.clone().slice([0..b, 0..t, 1..2, 0..h, 0..dh]).reshape([b, t, h, dh]).swap_dims(1, 2);
        let v = qkv.slice([0..b, 0..t, 2..3, 0..h, 0..dh]).reshape([b, t, h, dh]).swap_dims(1, 2);
        let q = rope(q, cos, sin);
        let k = rope(k, cos, sin);
        let y = attention(q, k, v, None, None, AttentionModuleOptions { scale: None, softcap: None, is_causal: true });
        let y = y.swap_dims(1, 2).reshape([b, t, d]);
        let x = x + self.o.forward(y);
        let hdn = self.n2.forward(x.clone());
        let f = silu(self.gate.forward(hdn.clone())) * self.up.forward(hdn);
        x + self.down.forward(f)
    }
}

impl Gpt {
    fn new(v: usize, d: usize, l: usize, h: usize, ff: usize, ctx: usize, dev: &Device) -> Self {
        let dh = d / h;
        let half = dh / 2;
        let mut cosv = vec![0f32; ctx * half];
        let mut sinv = vec![0f32; ctx * half];
        for p in 0..ctx {
            for i in 0..half {
                let f = 1.0 / 10000f32.powf(2.0 * i as f32 / dh as f32);
                cosv[p * half + i] = (p as f32 * f).cos();
                sinv[p * half + i] = (p as f32 * f).sin();
            }
        }
        let cos = Tensor::<2>::from_data(TensorData::new(cosv, [ctx, half]), dev);
        let sin = Tensor::<2>::from_data(TensorData::new(sinv, [ctx, half]), dev);
        Self {
            emb: EmbeddingConfig::new(v, d).with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 }).init(dev),
            blocks: (0..l)
                .map(|_| Block {
                    n1: RmsNormConfig::new(d).init(dev),
                    qkv: lin(d, 3 * d, dev),
                    o: lin(d, d, dev),
                    n2: RmsNormConfig::new(d).init(dev),
                    gate: lin(d, ff, dev),
                    up: lin(d, ff, dev),
                    down: lin(ff, d, dev),
                    heads: h,
                })
                .collect(),
            nf: RmsNormConfig::new(d).init(dev),
            cos,
            sin,
        }
    }
    fn forward(&self, idx: Tensor<2, Int>) -> Tensor<3> {
        let mut x = self.emb.forward(idx);
        for b in &self.blocks {
            x = b.forward(x, &self.cos, &self.sin);
        }
        let x = self.nf.forward(x);
        // tied head
        let w = self.emb.weight.val(); // [V, D]
        let [b, t, d] = x.dims();
        x.reshape([b * t, d]).matmul(w.transpose()).reshape([b, t, 0usize.max(self.emb.weight.dims()[0])])
    }
}

fn loss_fn(logits: Tensor<3>, targets: Tensor<2, Int>) -> Tensor<1> {
    let [b, t, v] = logits.dims();
    let lp = log_softmax(logits.reshape([b * t, v]), 1);
    let picked = lp.gather(1, targets.reshape([b * t, 1]));
    picked.mean().neg()
}

fn bench_matmul(dev: &Device, n: usize, iters: usize) -> f64 {
    let a = Tensor::<2>::random([n, n], Distribution::Default, dev);
    let b = Tensor::<2>::random([n, n], Distribution::Default, dev);
    for _ in 0..3 {
        let _ = a.clone().matmul(b.clone()).into_data();
    }
    let t0 = Instant::now();
    let mut acc = a.clone().matmul(b.clone());
    for _ in 1..iters {
        acc = acc + a.clone().matmul(b.clone());
    }
    let _ = acc.into_data();
    let s = t0.elapsed().as_secs_f64();
    2.0 * (n as f64).powi(3) * iters as f64 / s / 1e9
}

fn run(name: &str, base: Device, steps: usize) {
    println!("=== {name} ===");
    let t0 = Instant::now();
    println!("matmul 1024: {:.1} GFLOP/s", bench_matmul(&base, 1024, 10));
    println!("matmul 2048: {:.1} GFLOP/s", bench_matmul(&base, 2048, 5));
    let dev = base.clone().autodiff();
    let (v, d, l, h, ff, ctx, bsz) = (4096, 256, 4, 4, 704, 128, 16);
    let mut model = Gpt::new(v, d, l, h, ff, ctx, &dev);
    println!("params: {}", model.num_params());
    let mut opt = AdamWConfig::new().with_weight_decay(0.1).init();
    // fixed batch to check overfitting
    let data: Vec<i64> = (0..bsz * (ctx + 1)).map(|i| ((i * 7919) % 97) as i64 + ((i / 13) % 5) as i64).collect();
    let all = Tensor::<2, Int>::from_data(TensorData::new(data, [bsz, ctx + 1]), &dev);
    let x = all.clone().slice([0..bsz, 0..ctx]);
    let y = all.slice([0..bsz, 1..ctx + 1]);
    let mut first = 0.0;
    let mut timed = Instant::now();
    let warm = 3;
    let mut last = 0.0;
    for s in 0..steps + warm {
        if s == warm {
            timed = Instant::now();
        }
        let logits = model.forward(x.clone());
        let loss = loss_fn(logits, y.clone());
        let lv: f32 = loss.clone().into_scalar();
        if s == 0 {
            first = lv;
        }
        last = lv;
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = opt.step(3e-3, model, gp);
    }
    let el = timed.elapsed().as_secs_f64();
    let tps = (steps * bsz * ctx) as f64 / el;
    println!("loss first {first:.3} last {last:.3}; {:.1} ms/step; {:.0} tok/s; total {:.1}s", el * 1000.0 / steps as f64, tps, t0.elapsed().as_secs_f64());
}

fn main() {
    let which: Vec<String> = std::env::args().skip(1).collect();
    let steps = 30;
    if which.is_empty() || which.iter().any(|w| w == "metal") {
        run("metal", Device::metal(DeviceKind::DefaultDevice), steps);
    }
    if which.iter().any(|w| w == "cpu") {
        eprintln!("cpu backend not compiled");
    }
    if which.iter().any(|w| w == "flex") {
        run("flex", Device::flex(), 10);
    }
}
