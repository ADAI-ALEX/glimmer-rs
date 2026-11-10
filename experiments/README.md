# experiments/

Hyperparameter studies run by `glimmer tune <config>` write here (ignored by git):

```text
experiments/<study>/
  study.toml        the study configuration
  trials.jsonl      one record per trial and rung: parameters, validation loss, tokens, time
  summary.json      best trial and all records
  report.md         Markdown table of every trial
  trials/t###/      each trial's run directory (metrics, resolved config)
```

Curated results from the STEVIE studies are reproduced in `docs/optimisation.md`.
