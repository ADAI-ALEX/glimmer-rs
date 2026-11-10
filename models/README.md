# models/

Exported, ready-to-chat models live here (ignored by git — weights are large binary files).

```text
models/<name>/
  model.bpk        weights (Burn burnpack)
  model.json       architecture
  tokenizer.json   tokenizer used in training
  card.json        model card: lineage (stages, datasets, steps, tokens, losses), hardware, version
```

Create one by training with `export_as = "<name>"` (e.g. `glimmer train --model stevie`), or copy
a checkpoint directory. List them with `glimmer models list`; inspect with `glimmer models inspect <name>`.
