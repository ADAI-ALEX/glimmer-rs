# Backend comparison: Burn vs Candle on Apple Silicon

The experiment that chose GLIMMER's tensor framework. Both programs train the **same** small
transformer — 4 layers, d_model 256, 4 heads, SwiGLU d_ff 704, RMSNorm, RoPE, tied embeddings,
vocab 4096, 4.26M parameters — on a fixed batch (16 × 128 tokens) with AdamW, and report the
steady-state time per training step after 3 warm-up steps (30 timed steps).

Run on a MacBook Pro M1 Pro (8P+2E cores, 16-core GPU, 16 GB), macOS 26.6.2, Rust 1.99.1,
2026-11-09. These are standalone crates, excluded from the main workspace:

```bash
cd research/backend-comparison/burn   && cargo run --release -- metal   # or: flex
cd research/backend-comparison/candle && cargo run --release
```

## Results

| framework / backend | ms per step | tokens / s | notes |
|---|---|---|---|
| **Burn 0.22.1 · Metal** (wgpu + MSL, fusion, autotune) | **49.4** | **41,400** | ~25 s one-off kernel compilation/autotune at start-up |
| Candle 0.11.1 · Metal | 145.0 | 14,100 | fused `rms_norm`/`rope` kernels have no backward pass, so training must use slower decomposed ops; fast start-up (≈ 5 s) |
| Burn 0.22.1 · Flex CPU (rayon + SIMD + Apple AMX) | 271 | 7,550 | pure Rust, no native dependencies; 1.0 TFLOP/s on 2048² matmul via AMX |
| Burn 0.22.1 · Flex CPU (single-threaded default) | 835 | 2,450 | before enabling the `rayon`/`simd`/`apple-amx` features |
| Burn 0.22.1 · CubeCL CPU (LLVM JIT) | — | — | did not build: its downloaded LLVM toolchain links against Homebrew's `libzstd`, which is absent; rejected for portability |

## Decision

**Burn with the Metal backend for training, Burn Flex as the CPU fallback.**

* Burn trained this model **2.9× faster** than Candle on the M1 Pro GPU, thanks to kernel fusion
  and autotuned matmuls, and its autodiff covers every operation the model needs (attention falls
  back to a decomposed implementation with a full backward pass).
* Burn 0.22's runtime `Device` dispatch lets one binary contain Metal and CPU backends with no
  generic plumbing, which keeps the codebase simple.
* Flex needs no native libraries, so CI builds and tests run on stock Linux and macOS runners.
* PyTorch/LibTorch was excluded by design (the project is Rust-native; Burn's own LibTorch backend
  is deprecated as of 0.22).

## Caveats found during the spike

* The spike's quick matmul micro-benchmark under-reported Metal throughput (it timed kernel
  compilation and a fused accumulate). The step-time measurement above is the meaningful one, and
  GLIMMER's real benchmark suite (`glimmer bench`) warms up and synchronises properly.
* After 33 steps, Flex runs ended at a higher loss than Metal runs. That came from Burn drawing
  random initial weights with a backend-specific RNG, not from a numerical difference. GLIMMER
  therefore initialises weights on the host with its own RNG, and its test suite verifies that
  Metal and CPU produce the same logits (max difference < 1e-3) from identical weights.
