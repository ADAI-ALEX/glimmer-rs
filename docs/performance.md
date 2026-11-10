# Performance engineering

GLIMMER optimises only where measurements show a bottleneck, and reports numbers only from
benchmarks that can be rerun with `glimmer bench`.

## Methodology

[`crates/glimmer-bench`](../crates/glimmer-bench):

* **Warm-up** iterations (kernel compilation, autotuning, allocator growth) are excluded.
* Every timed iteration ends with a **device synchronisation** (a scalar is read back), so
  asynchronous GPU queues cannot make work look free.
* Measurements are **repeated** (20 iterations for kernels, 10 for training steps); mean, standard
  deviation and minimum are reported, together with the hardware and model configuration.
* Results are written as JSON to `runs/bench/<timestamp>.json`.
* Memory figures are read from the OS (GPU-resident memory from the IORegistry, process RSS),
  sampled before and after the training-step benchmark.

## Choosing the framework

Before building anything, the same 4.3M-parameter transformer was trained with Burn and with
Candle on the M1 Pro ([research/backend-comparison](../research/backend-comparison)):

| framework / backend | ms per training step | tokens / s |
|---|---|---|
| **Burn · Metal** | **49.4** | **41,400** |
| Candle · Metal | 145.0 | 14,100 |
| Burn · Flex CPU (rayon + SIMD + AMX) | 271 | 7,550 |

Burn was 2.9× faster for training on this GPU (kernel fusion, autotuned matmuls) and has a backward
pass for every operation the model uses.

## Sizing STEVIE: throughput sweep

`glimmer bench --backend metal --preset tiny --sweep-batch 8,16,32 --sweep-context 256,512`
(full training step: forward, loss, backward, AdamW; 13.8M parameters):

| context | batch 8 | batch 16 | batch 32 |
|---|---|---|---|
| 256 | 15,250 tok/s | **16,050 tok/s** | 14,740 tok/s |
| 512 | **13,030 tok/s** | 11,980 tok/s | 12,140 tok/s |

Context 512 costs ~20% throughput relative to 256 (attention is quadratic in context), but chat needs
multi-turn room and RoPE models do not extrapolate well beyond their training length, so STEVIE
pretrains at 512 with batch 8 × 2 accumulation. At ≈ 13k tokens/s that is ≈ 1.3 TFLOP/s of model
arithmetic (6·N per token), roughly a quarter of the M1 Pro's FP32 peak.

## Choosing the inference backend

Generation is a long chain of tiny forward passes (batch 1, one token at a time), each followed by a
device→host copy of the logits. On the GPU every step pays kernel-launch latency, and because the KV
cache grows by one position per token, kernel shapes keep changing — CubeCL compiles and autotunes
each new shape the first time it is seen in a process. During pretraining the same evaluation
prompts recur, so their kernels are cached and an evaluation with three 48-token samples took about
3 s. Interactive chat produces new shapes on every message.

The CPU backend has no JIT and uses Apple's AMX matrix units through Burn Flex. With the
mid-pretraining STEVIE checkpoint and the same seed, both backends produced *identical* text, with
very different speed (measured with `glimmer generate --raw`; the Metal run shared the GPU with
training, so it is a lower bound):

| backend | decode speed |
|---|---|
| CPU (Flex + AMX) | ~450 tokens/s |
| Metal (fresh process, GPU busy with training) | 5–10 tokens/s |

So `auto` selects Metal for training and the CPU for chat, generation and evaluation
(`glimmer_models::backend::select_for_inference`). The uncontended numbers are in the benchmark
table below.

<!-- BENCH_RESULTS -->

## Optimisations applied (and why)

| bottleneck observed | change | evidence |
|---|---|---|
| Flex CPU backend used one core | enabled Burn's `rayon`, `simd` and (macOS) `apple-amx` features | training step 835 → 271 ms; 2048² matmul ≈ 1 TFLOP/s |
| per-tensor gradient clipping distorts updates | one fused global-norm reduction, one sync per step | correctness, not speed; see `optim.rs` |
| recomputing the whole conversation every chat turn | KV cache with prefix reuse across turns | `decode (KV cache)` vs `decode (no cache)` benchmark |
| Metal decode latency for batch-1 chat | workload-aware backend selection (CPU for inference) | table above |
| loading a 274 MB token file into RAM | memory-mapped token files, windows decoded on demand | data loading ≈ 1–2 ms per batch, negligible next to a ~600 ms step |
| SFT padding waste | packing whole conversations into 512-token rows with segment-isolated attention | most of each row is real tokens rather than padding |

Not adopted, with reasons:

* **Mixed precision (f16/bf16)** — Burn supports f16 on Metal, but training it stably needs loss
  scaling and careful testing; at 13.8M parameters the model is not memory-bound, and correctness
  takes priority.
* **Flash attention during training** — Burn's fused attention is used for inference; under autodiff
  Burn falls back to the decomposed implementation, which is what has a backward pass. At context
  512 with 6 heads, the attention matrices are modest.
* **Async/Tokio** — the work is compute-bound; threads and channels are simpler and sufficient.
