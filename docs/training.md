# Training methodology

STEVIE is trained in two stages, both starting points recorded in its model card:

| stage | starts from | data | objective |
|---|---|---|---|
| 1. pretraining | **random initialisation** (seed 1337) | `pretrain-mix`: 137M tokens of general English | next-token prediction on every token |
| 2. supervised fine-tuning | best stage-1 checkpoint | `stevie-sft`: 563 original conversations | next-token prediction on STEVIE's turns only |

No pretrained model, tokenizer or weights from any other project are used at any point.

## The training loop

Implementation: [`crates/glimmer-training/src/trainer.rs`](../crates/glimmer-training/src/trainer.rs).

```text
for step in start..max_steps:
    lr ← schedule(step)                       # pure function of the step: exactly resumable
    batches ← accumulate × sample(rng)        # RNG state is checkpointed
    for each micro-batch:
        loss_mb ← Σ w·nll / supervised_tokens(all micro-batches)
        grads  += ∂loss_mb/∂θ                 # Burn autodiff
    ‖g‖ ← global L2 norm; clip to grad_clip    # one device sync
    skip the step if loss or ‖g‖ is non-finite (logged; abort after 5 in a row)
    θ ← AdamW(θ, g, lr)                        # decoupled weight decay, none on norm gains
    every eval.every: validation loss, best checkpoint, samples, early stopping
    every checkpoint.every: resumable checkpoint (+ pruning)
```

* **Optimiser**: AdamW (β₁ 0.9, β₂ 0.95, ε 1e-8). Lion and SGD+momentum are also available.
  Weight decay applies to weight matrices only; RMSNorm/LayerNorm gains and biases form a
  separate no-decay parameter group.
* **Gradient clipping**: global norm across *all* parameters (Burn's built-in clipping is
  per-tensor, which distorts the update direction), computed on the device in one reduction.
* **Gradient accumulation**: losses are normalised by the total number of supervised tokens across
  all micro-batches, so accumulation is mathematically equivalent to a larger batch. The test
  `gradient_accumulation_matches_large_batch` checks this.
* **Schedules**: linear warmup then cosine, linear, constant, or warmup-stable-decay.
* **Validation**: fixed, evenly spaced windows (pretraining) or every validation conversation
  (SFT), identical across runs so losses are comparable between experiments.
* **Early stopping & best checkpoint**: validation improvements larger than `min_delta` reset the
  patience counter and overwrite `checkpoints/best`; that checkpoint is what gets exported.
* **Mixed precision**: not used. Training runs in f32. Burn can run f16 on Metal, but stable f16
  training needs loss scaling, and correctness takes priority over speed at this scale.

### Checkpoints and resumption

A checkpoint holds model weights, AdamW moments (keyed by persistent parameter ids), the step,
tokens seen, the RNG state, the SFT data cursor, best-validation tracking, the full training config,
the architecture and the tokenizer. Writes are atomic (temporary directory + rename).

`resumed_training_matches_uninterrupted_run` trains 30 steps straight through, then repeats the run
stopping at step 15 and resuming from the checkpoint: the per-step losses after resumption match the
uninterrupted run to 1e-4. Resetting any of optimiser state, schedule position or RNG would fail it.

```bash
glimmer train --resume runs/stevie-pretrain-20261109-231823      # latest checkpoint of a run
glimmer train --resume runs/<run>/checkpoints/step-0008000       # a specific checkpoint
```

### The training monitor

[`monitor.rs`](../crates/glimmer-training/src/monitor.rs) watches the loss, gradient norm and
validation curve, and writes **advice** events: loss spikes, frequent clipping (learning rate
likely too high), very large gradients, rising validation loss with falling training loss
(overfitting), a widening generalisation gap, and plateaus. Advice is never applied
automatically. The only automatic actions are skipping non-finite steps and early stopping, both
configured up front and logged as `auto_action` events, so every run is reproducible from its
config.

## Stage 1: pretraining

Config: [`configs/training/stevie-pretrain.toml`](../configs/training/stevie-pretrain.toml).

| setting | value | why |
|---|---|---|
| model | `tiny`: 6 layers, d_model 384, 6 heads, SwiGLU 1024, context 512, **13.77M parameters** | trains at ~13k tokens/s on the M1 Pro GPU; leaves memory headroom on a 16 GB machine |
| batch | 8 sequences × 512 tokens × 2 accumulation = **8,192 tokens/step** | batch 8 was the most efficient size at context 512 in the throughput sweep |
| steps | 16,000 (**131M tokens** ≈ 0.96 epochs) | one pass over the corpus, no repetition |
| learning rate | peak **1e-3**, 500 warmup steps, cosine to 1e-4 | chosen by the LR range test below |
| regularisation | weight decay 0.1, no dropout, grad clip 1.0 | a single pass over 137M tokens does not overfit |

### Learning-rate range test

Before committing three hours of GPU time, four 300-step probes at constant learning rates (100
warmup steps) were run with the same model and data (`runs/lrprobe-*`):

| peak LR | validation loss @ 300 steps |
|---|---|
| 3e-4 | 4.497 |
| 6e-4 | 4.297 |
| **1e-3** | **4.238** |
| 2e-3 | 4.247 |

Loss improves up to 1e-3 and gets worse by 2e-3, so 1e-3 was chosen as the peak, with cosine decay
providing the low-LR phase that short probes cannot evaluate.

<!-- PRETRAIN_RESULTS -->

## Stage 2: conversational fine-tuning

Config: [`configs/training/stevie-sft.toml`](../configs/training/stevie-sft.toml).

* **Data**: packed rows of 512 tokens built from whole conversations; tokens attend only within
  their own conversation, and RoPE positions restart at 0 for each.
* **Loss masking**: only STEVIE's tokens and the `<|end|>` closing each of its turns are targets.
  System prompts, user turns, role tags and padding contribute nothing. STEVIE learns to answer,
  not to imitate users, and learns when to stop.
* **System prompt**: half the examples include "You are Stevie, a friendly Australian AI mate."
* **Regularisation**: the corpus is small (~41k tokens per epoch), so validation runs every 10
  steps, early stopping has a patience of 8 evaluations, and the best checkpoint is exported.
* **Hyperparameters**: learning rate, weight decay and dropout come from a TPE + successive-halving
  study (see [optimisation.md](optimisation.md)).

<!-- SFT_RESULTS -->

## Reproducing STEVIE

```bash
glimmer data pipeline                      # ≈ 5 min: download, clean, tokenizer, datasets
glimmer tune stevie-sft-search             # optional: re-run the SFT hyperparameter study
glimmer train --model stevie               # pretraining (~3 h on an M1 Pro) then SFT
glimmer evaluate --baseline --model runs/<pretrain-run>/checkpoints/best --model stevie
```

Seeds are fixed for weight initialisation, data sampling, splits and dropout. Bitwise-identical
reruns are guaranteed on the CPU backend; on Metal, autotuning can choose different kernels between
processes, so reruns agree to floating-point rounding rather than bit for bit.
