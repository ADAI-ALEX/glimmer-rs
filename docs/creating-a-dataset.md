# How to create a new training dataset

Datasets are built from declared **sources** by declarative **dataset specs**. Both are plain TOML.

## 1. Declare a source

Create `datasets/sources/<id>.toml`. Every source must state its licence, licence URL, homepage,
provenance and redistribution terms — the build copies these into every dataset manifest.

**Plain text over HTTP** (optionally truncated with a range request):

```toml
id = "my-corpus"
name = "My plain-text corpus"
kind = "http_text"
url = "https://example.org/corpus.txt"
max_bytes = 100_000_000                # optional
document_separator = "<|endoftext|>"   # optional; paragraphs otherwise
license = "CC-BY-4.0"
license_url = "https://creativecommons.org/licenses/by/4.0/"
homepage = "https://example.org"
provenance = "Who wrote it, how it was collected."
redistribution = "What the licence allows; GLIMMER downloads on demand."

[cleaning]
min_chars = 100
max_chars = 50_000
min_alpha_ratio = 0.6
near_dedup = true
near_dedup_threshold = 0.8
blocklist = true
chunk_chars = 0          # split long documents on paragraph boundaries (0 = never)
```

Other kinds: `http_parquet` (one document per row: `url`, `text_column`, optional `title_column`,
`max_rows`), `gutenberg` (a list of `[[books]]` with `id`, `title`, `author`, `died`) and
`conversations` (a local directory of STEVIE-format `.txt` files, see
[`datasets/stevie/README.md`](../datasets/stevie/README.md)).

> Check the licence yourself. "Freely downloadable" is not the same as "licensed for training or
> redistribution".

## 2. Fetch and prepare

```bash
glimmer data fetch --source my-corpus      # → data/raw/my-corpus/ (+ provenance.json with SHA-256)
glimmer data prepare --source my-corpus    # → data/corpora/my-corpus.jsonl (+ .stats.json)
glimmer sources                            # status and statistics of every source
```

## 3. (Optional) train a tokenizer

Reuse `glimmer-8k`, or create `configs/tokenizers/<name>.toml`:

```toml
name = "my-16k"
vocab_size = 16384
min_frequency = 2

[[sources]]
id = "my-corpus"
max_bytes = 80_000_000
```

`glimmer tokenizer train my-16k`. Only training-split documents are used.

## 4. Write a dataset spec

`configs/datasets/<name>.toml`:

```toml
name = "my-mix"
kind = "pretrain"                     # or "sft"
description = "My corpus plus TinyStories"
tokenizer = "data/tokenizers/glimmer-8k.json"

[split]
val_fraction = 0.005
test_fraction = 0.005
seed = 7

[[sources]]
id = "my-corpus"
weight = 2.0                          # upsample ×2 (fractions subsample)

[[sources]]
id = "tinystories-v2"
weight = 0.5
```

For `kind = "sft"`, add:

```toml
[sft]
system_prompt_probability = 0.5
max_tokens = 512
eval_prompts = "datasets/my-bot/eval_prompts.toml"   # leakage check
leakage_threshold = 0.7
```

## 5. Build and inspect

```bash
glimmer data build my-mix
glimmer datasets inspect my-mix
```

The output directory `data/datasets/my-mix/` holds `train/val/test.bin` (pretraining, u16 tokens
separated by `<|endoftext|>`) or `train/val/test.sft` (SFT: tokens + per-token target masks), plus
`manifest.json` recording the tokenizer and its SHA-256, the split seed, per-source document and
token counts per split, licences, provenance and every preprocessing step.

Point a training config at it with `[data] dataset = "my-mix"`.

## Writing a conversational corpus

1. Create `datasets/<bot>/conversations/*.txt` in the `### id | topic` / `U:` / `S:` format.
2. Create `datasets/<bot>/eval_prompts.toml` with held-out prompts — never train on these.
3. Add a `conversations` source pointing at the directory, then an `sft` dataset spec.
4. Aim for diversity over volume: different lengths, structures, follow-up questions, multi-turn
   context, and a consistent character. Near-duplicate detection flags templated variations.
5. Document provenance, especially for AI-written or synthetic material.
