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
selection; it is reserved for the final evaluation. Two scoring modes:

| `objective` | trial score | use when |
|---|---|---|
| `last` (default) | validation loss at the end of the trial's budget | the final run trains for a fixed number of steps |
| `best` | lowest validation loss seen so far, evaluated every `eval_every` steps | the final run uses early stopping and exports its best checkpoint |

The score must match how the winning configuration will be used. STEVIE's SFT run exports its best
checkpoint, so its study uses `objective = "best"`; scoring by the last step would penalise
configurations that peak early and then overfit, even though the final run keeps that peak. Each
rung boundary also triggers an evaluation, so a trial is never scored on a stale value.

## Results: tuning STEVIE's fine-tuning

Two studies were run on the M1 Pro GPU. The first was badly designed for this data, and what it
showed shaped the second. Both are kept: [`stevie-sft-search-v1.toml`](../configs/tuning/stevie-sft-search-v1.toml)
and [`stevie-sft-search.toml`](../configs/tuning/stevie-sft-search.toml).

### Study 1: a 200-step budget (superseded)

18 TPE trials, successive halving with η = 3 at [22, 66, 200] steps, scored by best validation
loss. Wall time 5 min 27 s.

The winner (lr 4.35e-4, weight decay 0.093, dropout 0) scored 3.917, but its score was identical at
all three rungs, and so was the runner-up's. Its validation curve explains why:

| step | 11 | 22 | 44 | 66 | 110 | 154 | 200 |
|---|---|---|---|---|---|---|---|
| validation loss | 4.170 | **3.917** | 4.075 | 4.429 | 5.027 | 5.213 | 5.322 |
| training loss | 4.595 | 4.158 | 2.958 | 1.746 | 0.355 | 0.061 | 0.017 |

One epoch of the fine-tuning set is ~11 steps. Every configuration reached its best validation
loss within two epochs, then memorised the 628 training conversations: training loss fell in a
staircase, one step down per epoch, to 0.017. Because the best score was always set inside rung 0,
the later rungs could not change the ranking and successive halving did nothing. The study had
effectively compared configurations on 22 steps, during the 20-step warmup. A test confirmed this
was genuine overfitting rather than a bug, such as dropout leaking into evaluation
(`dropout_is_active_in_training_and_disabled_for_evaluation`).

### Study 2: a budget matched to the data

The budget was moved into the useful window: 54 steps (~4.8 epochs), 5 warmup steps, cosine decay,
evaluation every 3 steps, 24 TPE trials (5 random, then model-guided), successive halving with
η = 2 at [13, 27, 54] steps (~1.2, 2.4 and 4.8 epochs). Wall time 4 min 34 s.

| | lr | weight decay | dropout | best validation loss |
|---|---|---|---|---|
| **winner (trial 10)** | **3.08e-4** | **0.106** | **0.020** | **3.892** |
| runner-up (trial 15) | 3.36e-4 | 0.136 | 0.042 | 3.900 |
| 6th of 6 finalists (trial 20) | 3.62e-4 | 0.143 | 0.035 | 3.907 |
| highest learning rate tried | 3.00e-3 | 0.200 | 0.028 | 4.620 (rung 0) |
| heaviest dropout tried | 5.98e-4 | 0.021 | 0.300 | 4.342 (rung 0) |
| lowest learning rate tried | 1.00e-4 | 0.300 | 0.001 | 4.200 (rung 0) |

What it shows:

* **Successive halving now earns its keep.** Trial 10 ranked 8th after 13 steps (4.013), and only
  pulled ahead once the schedule had decayed (3.900 at 27 steps, 3.892 at 54). A study that only
  ran rung 0 would have picked a different configuration.
* **TPE concentrates quickly.** Trials 0–9 spread across the space, including both ends of the
  learning-rate range. From trial 10 onwards every proposal fell in lr 3.1–3.7e-4, weight decay
  0.11–0.14, dropout 0.02–0.055. Trial 10, the first in that region, turned out to be the winner.
* **The finalists are indistinguishable.** The six finalists are within 0.015 of each other. The
  final run, with the winner's hyperparameters and a different seed, reached 3.898: seed-to-seed
  variation is as large as the gap between finalists. What the study reliably establishes is the
  region (learning rate ≈ 3–4e-4, little or no dropout, moderate weight decay), not a single best
  point.

`configs/training/stevie-sft.toml` uses the winner's values. Each is recorded with its trial, so
the choice can be traced and re-run.

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
