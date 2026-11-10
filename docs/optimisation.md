# Optimisation: hyperparameter search and adaptive controls

GLIMMER improves training configurations automatically, but transparently: a search only
changes a trial's configuration *before* the trial starts, records every value it chose, and keeps
each trial's resolved config next to its metrics. Nothing changes silently mid-run.

Implementation: [`crates/glimmer-optimisation`](../crates/glimmer-optimisation).

## Algorithms

| algorithm | what it does | when to use it |
|---|---|---|
| **random search** | samples each parameter independently from its prior | baselines, very small budgets, sanity checks |
| **TPE** (Tree-structured Parzen Estimator) | Bayesian: models `p(x \| good)` and `p(x \| bad)` from completed trials and proposes the candidate maximising their ratio | the default; learns quickly which regions are promising |
| **successive halving** | trains many configurations briefly, promotes the best 1/η to an η× larger budget, repeats | when bad configurations are recognisable early (they usually are for learning rates) |

TPE follows Bergstra et al. (2011) with univariate Parzen estimators as in Hyperopt: completed
trials split at the 25% quantile of the objective; Gaussian kernels with bandwidths from neighbour
distances plus a uniform prior component; 32 candidates drawn from the "good" density; categorical
parameters use Laplace-smoothed counts. The first 5 proposals are random. The unit test
`tpe_beats_random_on_quadratic` checks it concentrates on the optimum far faster than random search
(by more than 2×), and `categorical_tpe_prefers_good_choice` checks the categorical path.

Successive halving (Jamieson & Talwalkar, 2016) is implemented with **exact continuation**: every
trial is configured with the full step budget, so its learning-rate schedule is the same whether it
stops at rung 1 or runs to the end. At a rung boundary the trainer evaluates, saves a resumable
`rung` checkpoint and returns; promoted trials resume from it with optimiser, RNG and data cursor
intact (verified by `successive_halving_promotes_and_resumes`, which checks token counts match an
uninterrupted run). Checkpoints of eliminated trials are deleted as soon as they are eliminated.

TPE and successive halving combine: rung-0 trials are proposed and evaluated one at a time, so TPE
learns from each result before proposing the next.

## Search spaces

```toml
[[space]]
kind = "log_uniform"      # log_uniform | uniform | choice
name = "lr"
low = 3e-5
high = 2e-3
```

Searchable: `lr`, `min_lr_ratio`, `weight_decay`, `warmup_steps`, `batch_size`, `grad_accum`,
`dropout`, `context_length`, `grad_clip`, `beta2`, `d_model`, `n_layers`. Unknown names are
rejected rather than ignored.

## Running a study

```bash
glimmer tune stevie-sft-search          # configs/tuning/stevie-sft-search.toml
```

Outputs in `experiments/<study>/`: `study.toml`, `trials.jsonl` (one record per trial per rung),
`summary.json` and `report.md`. Studies appear in the TUI's Experiments screen. Trials share one
opened compute device, so Metal kernels compile once per study, not once per trial.

The objective is validation loss on the validation split. The test split is never used for
selection; it is reserved for the final evaluation.

<!-- STUDY_RESULTS -->

## Adaptive training controls

During every run the [monitor](../crates/glimmer-training/src/monitor.rs) watches:

| signal | detection | response |
|---|---|---|
| non-finite loss or gradient | NaN/Inf after the backward pass | **automatic**: skip the step (logged); abort as diverged after 5 in a row |
| loss spike | loss > 1.5× its slow EMA and > +0.5 after warmup | advice: lower peak LR or lengthen warmup if spikes recur |
| frequent clipping | > 60% of the last 100 steps clipped | advice: learning rate likely too high |
| exploding gradients | global norm > 50 after warmup | advice |
| overfitting | validation loss rises for two evaluations while training loss falls | advice; best checkpoint is preserved |
| generalisation gap | validation − training loss > 0.5 and growing | advice |
| plateau | < 0.005 validation improvement over four evaluations | advice |
| no improvement | `patience` evaluations without improving by `min_delta` | **automatic** (if configured): early stopping |

Learning-rate scheduling, gradient clipping and early stopping are configured up front. The
automatic actions above are logged as `auto_action` events in `metrics.jsonl` and shown in the TUI,
so every run can be reproduced from its config and its log.

## Reinforcement learning

Hyperparameter search and reinforcement learning are different techniques, and GLIMMER does not
use RL as a substitute for the former. The architecture leaves room for reward-based post-training
(e.g. preference optimisation over STEVIE responses) as a future stage: it would slot in as another
`Stage` with its own data type, reusing the model, checkpointing, evaluation and TUI unchanged.
