# How to add another model

GLIMMER separates *what a model is* (configuration + weights) from *how it is trained, run and
displayed*. There are two levels of "new model".

## 1. A new model of the existing architecture (no code)

Most new models are a new size, dataset or persona built from the same transformer.

1. **Write a training config** in `configs/training/`, e.g. `koala-pretrain.toml`:

   ```toml
   [run]
   name = "koala-pretrain"
   stage = "pretrain"
   seed = 42
   backend = "auto"
   export_as = "koala"          # optional: export best checkpoint to models/koala

   [model]
   preset = "small"             # micro | tiny | small, then override any field:
   n_layers = 10
   context_length = 1024

   [data]
   dataset = "pretrain-mix"     # any prepared dataset (see creating-a-dataset.md)

   [optimizer]
   lr = 6e-4
   min_lr = 6e-5

   [schedule]
   warmup_steps = 1000
   max_steps = 40000

   [batch]
   batch_size = 8
   grad_accum = 4
   ```

2. **Optionally add a fine-tuning stage** that starts from the pretrained checkpoint:

   ```toml
   [run]
   name = "koala-sft"
   stage = "sft"
   export_as = "koala"

   [data]
   dataset = "koala-chat"

   [init]
   from_checkpoint = "run:koala-pretrain"   # best checkpoint of the latest koala-pretrain run
   ```

3. **Optionally add a recipe** `configs/models/koala.toml` listing the stages, so
   `glimmer train --model koala` runs them in order.

4. **Train and chat**: `glimmer train --model koala`, then `glimmer chat --model koala`. The model
   appears in the TUI's Models screen with its full lineage card.

Use `glimmer tune` with a study config (see [optimisation.md](optimisation.md)) to choose learning
rate, weight decay, dropout, batch size or even model width/depth for the new model.

## 2. A new architecture (code)

1. **Add the module** to `crates/glimmer-models/src/` — e.g. `mamba.rs` — as a Burn
   `#[derive(Module)]` struct with:
   * `init(cfg: &ModelConfig, device: &Device, seed: u64) -> Self` — initialise on the host from
     `glimmer_core::Rng` so seeds are backend-independent;
   * `forward(tokens: Tensor<2, Int>) -> Tensor<3>` (causal LM logits);
   * `forward_packed(...)` if it should support packed SFT batches;
   * `forward_cached(...)` + a cache type if it should support fast generation.
2. **Extend `ModelConfig`** (`crates/glimmer-core/src/config.rs`) with an `architecture` field and
   any new hyperparameters, keeping `serde(default)` so existing configs and checkpoints still load,
   and implement `param_count()` for it.
3. **Dispatch on the architecture** in the few places that construct models:
   `glimmer_models::store::load_weights`, `Transformer::init` call sites in the trainer, the
   inference engine and the benchmark suite. A small enum wrapper (`enum Model { Transformer(..),
   Mamba(..) }`) implementing the forward methods keeps the rest of the system unchanged.
4. **Add correctness tests** mirroring
   [`crates/glimmer-models/tests/correctness.rs`](../crates/glimmer-models/tests/correctness.rs):
   parameter count, causality, cache equivalence, gradients vs finite differences, save/load.

Because training, evaluation, inference, checkpoints, metrics and the TUI only depend on the
forward-pass signatures and `ModelConfig`, nothing else needs to change.

## Checklist

- [ ] config validates (`glimmer train --config … --max-steps 1`)
- [ ] `glimmer bench --preset …` gives sensible throughput
- [ ] loss decreases on `configs/training/micro-smoke.toml`-style short runs
- [ ] correctness tests pass on CPU (and Metal locally)
- [ ] `glimmer evaluate --model <name> --baseline` produces a report
