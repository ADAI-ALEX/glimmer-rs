# Evaluation

`glimmer evaluate` measures what a model has learned with numbers *and* with its actual output.
Implementation: [`crates/glimmer-evaluation`](../crates/glimmer-evaluation).

```bash
# Compare an untrained model, the pretrained checkpoint and STEVIE on the same prompts:
glimmer evaluate --baseline \
  --model runs/<pretrain-run>/checkpoints/best --label "after pretraining" \
  --model stevie --label "STEVIE (after SFT)"
```

Reports are written to `runs/eval/<name>.md` (side-by-side Markdown) and `.json` (every metric and
every generated response). They also appear in the TUI's Evaluation screen.

## What is measured

| metric | definition | what it tells you |
|---|---|---|
| conversation test loss / perplexity | mean NLL per supervised token on `stevie-sft/test.sft` (never trained on, never used for selection) | how well the model predicts STEVIE-style replies it has not seen |
| general-text test loss / perplexity | mean NLL on fixed windows of `pretrain-mix/test.bin` | general English modelling; shows whether fine-tuning cost general fluency |
| ends its turn | fraction of replies that end with `<|end|>` rather than hitting the length limit | whether the model learned the conversation format |
| Australian markers | fraction of replies containing Australian vocabulary or spelling (lexicon in [`lexicon.rs`](../crates/glimmer-data/src/lexicon.rs)) | whether STEVIE *sounds* Australian |
| US spellings | fraction of replies with American spellings (color, favorite, …) | dialect consistency |
| topical overlap | reply shares a content word with the prompt | a crude proxy for staying on topic |
| multi-turn continuity | for prompts with earlier context, the reply contains an expected context word (e.g. the user's name) | whether earlier turns influence the reply |
| distinct-2, repeated trigrams | lexical diversity and looping | degenerate repetition |
| length, speed | words per reply; decode tokens per second | practical behaviour |

### Held-out prompts

[`datasets/stevie/eval_prompts.toml`](../datasets/stevie/eval_prompts.toml): 40 prompts across 15
categories (greetings, identity, wildlife, food, Australia, slang, tech, science, feelings,
advice, humour, small talk, general knowledge, opinion, travel) plus 5 multi-turn continuity
probes. They include the four example exchanges from the project brief. The SFT dataset build
excludes any training conversation that overlaps them.

### Sampling

Responses are sampled at temperature 0.7, top-k 40, top-p 0.92, repetition penalty 1.15, up to 96
new tokens, with fixed per-prompt seeds, so a report is reproducible. Every response in a report is
the model's first and only sample for that prompt and seed: nothing is selected, edited or rerun.

## Interpreting results honestly

* The lexical metrics are deliberately simple. They show whether the model has picked up a style
  and a format. They cannot judge whether an answer is *correct* or genuinely coherent; read the
  responses.
* A 14M-parameter model trained for a few hours on a laptop has a little general fluency and a
  learned voice. It does not have reliable knowledge, reasoning or memory. It will state wrong facts
  confidently, mix up details and sometimes drift off topic.
* Lower perplexity on the conversation test set after SFT shows the model learned the
  distribution of STEVIE's replies; it does not show it has become a capable assistant.

<!-- EVAL_RESULTS -->
