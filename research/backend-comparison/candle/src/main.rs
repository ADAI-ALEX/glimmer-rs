use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{AdamW, Embedding, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::time::Instant;

struct Block {
    n1: Tensor,
    n2: Tensor,
    qkv: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    h: usize,
}

fn rms(x: &Tensor, g: &Tensor) -> Result<Tensor> {
    candle_nn::ops::rms_norm_slow(x, g, 1e-5)
}

fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, t, dh) = x.dims4()?;
    let half = dh / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let c = cos.narrow(0, 0, t)?;
    let s = sin.narrow(0, 0, t)?;
    let r1 = (x1.broadcast_mul(&c)? - x2.broadcast_mul(&s)?)?;
    let r2 = (x1.broadcast_mul(&s)? + x2.broadcast_mul(&c)?)?;
    Tensor::cat(&[r1, r2], 3)
}

impl Block {
    fn fwd(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let h = self.h;
        let dh = d / h;
        let qkv = self.qkv.forward(&rms(x, &self.n1)?)?.reshape((b, t, 3, h, dh))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let att = (q.matmul(&k.t()?)? / (dh as f64).sqrt())?;
        let att = att.broadcast_add(mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;
        let x = (x + self.o.forward(&y)?)?;
        let hdn = rms(&x, &self.n2)?;
        let f = (candle_nn::ops::silu(&self.gate.forward(&hdn)?)? * self.up.forward(&hdn)?)?;
        x + self.down.forward(&f)?
    }
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (v, d, l, h, ff, ctx, bsz) = (4096usize, 256usize, 4usize, 4usize, 704usize, 128usize, 16usize);
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let init = candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 };
    let lin = |vb: VarBuilder, i: usize, o: usize| -> Result<Linear> {
        Ok(Linear::new(vb.get_with_hints((o, i), "w", init)?, None))
    };
    let emb_w = vb.get_with_hints((v, d), "emb", init)?;
    let emb = Embedding::new(emb_w.clone(), d);
    let mut blocks = vec![];
    for i in 0..l {
        let vb = vb.pp(format!("b{i}"));
        blocks.push(Block {
            n1: vb.get_with_hints(d, "n1", candle_nn::Init::Const(1.0))?,
            n2: vb.get_with_hints(d, "n2", candle_nn::Init::Const(1.0))?,
            qkv: lin(vb.pp("qkv"), d, 3 * d)?,
            o: lin(vb.pp("o"), d, d)?,
            gate: lin(vb.pp("g"), d, ff)?,
            up: lin(vb.pp("u"), d, ff)?,
            down: lin(vb.pp("dn"), ff, d)?,
            h,
        });
    }
    let nf = vb.get_with_hints(d, "nf", candle_nn::Init::Const(1.0))?;
    let dh = d / h;
    let half = dh / 2;
    let mut cv = vec![0f32; ctx * half];
    let mut sv = vec![0f32; ctx * half];
    for p in 0..ctx {
        for i in 0..half {
            let f = 1.0 / 10000f32.powf(2.0 * i as f32 / dh as f32);
            cv[p * half + i] = (p as f32 * f).cos();
            sv[p * half + i] = (p as f32 * f).sin();
        }
    }
    let cos = Tensor::from_vec(cv, (ctx, half), &dev)?;
    let sin = Tensor::from_vec(sv, (ctx, half), &dev)?;
    let mut m = vec![0f32; ctx * ctx];
    for i in 0..ctx {
        for j in 0..ctx {
            if j > i {
                m[i * ctx + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(m, (ctx, ctx), &dev)?;
    let nparams: usize = vm.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("params {nparams}");
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 3e-3, weight_decay: 0.1, ..Default::default() })?;
    let data: Vec<u32> = (0..bsz * (ctx + 1)).map(|i| ((i * 7919) % 97) as u32 + ((i / 13) % 5) as u32).collect();
    let all = Tensor::from_vec(data, (bsz, ctx + 1), &dev)?;
    let x = all.narrow(1, 0, ctx)?;
    let y = all.narrow(1, 1, ctx)?;
    let steps = 30;
    let warm = 3;
    let mut t0 = Instant::now();
    let start = Instant::now();
    for s in 0..steps + warm {
        if s == warm {
            t0 = Instant::now();
        }
        let mut hdn = emb.forward(&x)?;
        for b in &blocks {
            hdn = b.fwd(&hdn, &cos, &sin, &mask)?;
        }
        let hdn = rms(&hdn, &nf)?;
        let logits = hdn.reshape((bsz * ctx, d))?.matmul(&emb_w.t()?)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &y.flatten_all()?)?;
        opt.backward_step(&loss)?;
        if s == 0 || s == steps + warm - 1 {
            println!("step {s} loss {:.3}", loss.to_scalar::<f32>()?);
        }
    }
    let _ = emb_w.sum_all()?.to_scalar::<f32>()?;
    let el = t0.elapsed().as_secs_f64();
    println!(
        "{:.1} ms/step; {:.0} tok/s; total {:.1}s",
        el * 1000.0 / steps as f64,
        (steps * bsz * ctx) as f64 / el,
        start.elapsed().as_secs_f64()
    );
    let _ = D::Minus1;
    Ok(())
}
