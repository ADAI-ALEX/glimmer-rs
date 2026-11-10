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

Repeated later on an idle machine with the finished STEVIE (`glimmer generate --model stevie
--backend <b> --max-tokens 64 --seed 5 "Tell me about wombats."`, two fresh processes each):

| backend | decode speed, fresh process, idle machine |
|---|---|
| CPU (Flex + AMX) | **618 / 609 tokens/s** |
| Metal | **5.9 / 6.1 tokens/s** |

The GPU itself is not slow: once every shape has been compiled, Metal decodes at 437 tokens/s
(benchmark below). The time goes on compiling and autotuning kernels for each new sequence length,
and a chat produces new lengths with every token. So `auto` selects Metal for training and the CPU
for chat, generation and evaluation (`glimmer_models::backend::select_for_inference`).

## Benchmark results

`glimmer bench` (default settings) on an idle 16 GB MacBook Pro: Apple M1 Pro (8 performance + 2
efficiency cores, 16-core GPU), macOS 26.6.2, rustc 1.99.1, release build. Raw results:
[`docs/results/bench-m1pro.json`](results/bench-m1pro.json). Mean of 20 timed iterations for
kernels and 10 for model steps, after warm-up; ± is one standard deviation.

**Kernels**

| operation | Metal | CPU (Flex · 10 threads · AMX) |
|---|---|---|
| matmul 512² f32 | 0.68 ms ± 0.13 (392 GFLOP/s) | **0.46 ms ± 0.13 (585 GFLOP/s)** |
| matmul 1024² f32 | **1.11 ms ± 0.03 (1.94 TFLOP/s)** | 2.20 ms ± 0.09 (0.98 TFLOP/s) |
| matmul 2048² f32 | **6.04 ms ± 0.05 (2.84 TFLOP/s)** | 13.71 ms ± 0.24 (1.25 TFLOP/s) |
| causal attention fwd, B8 H6 T256 Dh64 | **1.79 ms ± 0.02** | 2.34 ms ± 0.09 |
| causal attention fwd, B8 H6 T512 Dh64 | **6.02 ms ± 0.06** | 8.28 ms ± 0.16 |

At small sizes the CPU wins: AMX has no launch overhead. From 1024² up, the GPU is 2–2.3× faster.

**STEVIE (`tiny`, 13.8M parameters)**

| operation | Metal | CPU |
|---|---|---|
| forward pass | B8 T512: 85.5 ms (47.9k tokens/s) | B4 T256: 96.3 ms (10.6k tokens/s) |
| training step (forward + backward + AdamW) | B8 T512: **305 ms (13.4k tokens/s)** | B4 T256: 418 ms (2.45k tokens/s) |
| decode with KV cache (64-token prompt + 64 new) | 147 ms (437 tokens/s) | **109 ms (590 tokens/s)** |
| decode without KV cache | 288 ms (223 tokens/s) | 930 ms (69 tokens/s) |
| KV-cache speed-up | 2.0× | 8.6× |
| checkpoint save (55 MB weights) | 34 ms (1.60 GB/s) | 35 ms (1.59 GB/s) |
| checkpoint load (file → device, all parameters read back) | 259 ms | 222 ms |

The CPU benchmarks use a smaller batch and context to keep them short. Shorter contexts make
attention cheaper per token, which favours the CPU, so the GPU's training advantage (5.5× in
tokens/s here) is if anything understated. Decode numbers are steady-state, with kernels already
compiled; see above for why fresh-process Metal decoding is far slower.

**Transfers, memory and host-side work**

| measurement | result |
|---|---|
| upload a token batch, 8×512 i64 (33 KB), host → Metal | 0.41 ms ± 0.04 |
| read back one row of logits, 8192 f32 (33 KB), Metal → host | 0.34 ms ± 0.06 |
| GPU memory in use, system-wide (IORegistry), before → after the training-step benchmark | 353 MiB → 2.7 GiB |
| process resident memory, Metal run, before → after | 99 → 268 MiB |
| data loading, B16 T512 random windows from the memory-mapped token file | 1.28 ms cold page cache (first run); 0.01 ms warm (second run) |
| tokenizer encoding, 166 KB of text | 1.91 ms (87 MB/s) |

Transfers are latency-bound (~0.3–0.4 ms each, through wgpu staging buffers). Unified memory means
no PCIe copy, but each transfer still pays a submission round-trip. A pretraining step (two
accumulated micro-batches) does two batch loads, two uploads and one scalar read: about 4 ms even
with a cold page cache, under 1% of the ~608 ms step.

In the first run of this suite, the checkpoint-load benchmark exposed a real bug. Burn's burnpack
reader is lazy, so the benchmark (which deleted its temporary file after loading) caused a
background-thread panic when the data was finally read. `load_weights` now reads the file
completely before building the model, the benchmark reads every parameter back inside the timed
region, and `save_and_load_roundtrip_preserves_outputs` deletes the file before using the loaded
model.


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
