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
| ends its turn | fraction of replies closed with `<|end|>`. Stopping on another special token (`<|endoftext|>`, a role tag) or at the length limit does not count | whether the model learned the conversation format |
| Australian markers | fraction of replies containing Australian vocabulary or spelling (lexicon in [`lexicon.rs`](../crates/glimmer-data/src/lexicon.rs)) | whether STEVIE *sounds* Australian |
| US spellings | fraction of replies with American spellings (color, favorite, …) | dialect consistency |
| topical overlap | reply shares a content word with the prompt | a crude proxy for staying on topic |
| multi-turn continuity | for prompts with earlier context, the reply contains an expected context word *as a whole word* (e.g. the user's name; "Tom" does not match "bottom") | whether earlier turns influence the reply (lexically: a reply can mention the right word in the wrong way) |
| distinct-2, repeated trigrams | lexical diversity and looping | degenerate repetition |
| copies training | fraction of replies containing 8+ consecutive words that also appear in a training reply | memorisation: generating new replies vs regurgitating the corpus |
| length, speed | words per reply; decode tokens per second | practical behaviour |

The report also measures the corpus's own **reference replies** (STEVIE's turns in the
`stevie-sft` test split) with the same lexical metrics. The corpus uses Australian vocabulary
sparingly by design, so a model's marker rate should be compared with the reference rate, not
with 100%.

### Held-out prompts

[`datasets/stevie/eval_prompts.toml`](../datasets/stevie/eval_prompts.toml): 40 prompts. 35 are
single-turn prompts across 15 categories (greetings, identity, wildlife, food, Australia, slang,
tech, science, feelings, advice, humour, small talk, general knowledge, opinion, travel); the other
5 are multi-turn continuity probes. They include the four example exchanges from the project brief. The SFT dataset build
excludes any training conversation that overlaps them.

### Sampling

Responses are sampled at temperature 0.7, top-k 40, top-p 0.92, repetition penalty 1.15, up to 96
new tokens, with fixed per-prompt seeds, so a report is reproducible. `--samples N` draws N
responses per prompt with different seeds, and every metric averages over all of them. The
Markdown report shows the first sample for each prompt and the JSON holds all of them. Nothing is
selected, edited or rerun.

## Interpreting results honestly

* The lexical metrics are deliberately simple. They show whether the model has picked up a style
  and a format. They cannot judge whether an answer is *correct* or genuinely coherent; read the
  responses.
* A 14M-parameter model trained for a few hours on a laptop has a little general fluency and a
  learned voice. It does not have reliable knowledge, reasoning or memory. It will state wrong facts
  confidently, mix up details and sometimes drift off topic.
* Lower perplexity on the conversation test set after SFT shows the model learned the
  distribution of STEVIE's replies; it does not show it has become a capable assistant.

## Results: STEVIE

```bash
glimmer evaluate --baseline \
  --model runs/stevie-pretrain-20261109-231823/checkpoints/best --label "after pretraining" \
  --model stevie --label "STEVIE (after SFT)" \
  --model runs/ablation-sft-overtrained-20261110-023457/checkpoints/final --label "ablation: overtrained SFT" \
  --samples 3
```

40 held-out prompts × 3 seeds = 120 responses per model. The full report, with every response,
is in [`docs/results/stevie-eval.md`](results/stevie-eval.md) ([JSON](results/stevie-eval.json)).

| metric | random init | after pretraining | **STEVIE** | ablation: overtrained SFT | corpus reference |
|---|---|---|---|---|---|
| conversation test loss (ppl) | 9.110 (9048) | 4.883 (132.1) | **3.944 (51.6)** | 4.909 (135.6) | |
| general-text test loss (ppl) | 9.047 (8497) | 2.367 (10.7) | 2.571 (13.1) | 3.433 (31.0) | |
| ends its turn with `<|end|>` | 0% | 0% | **100%** | 100% | |
| replies with Australian markers | 6% | 9% | **18%** | 27% | 44% |
| Australian markers per reply | 0.08 | 0.14 | 0.23 | 0.39 | 0.88 |
| replies with US spellings | 27% | 2% | **0%** | 0% | 0% |
| topical overlap with prompt | 0% | 34% | **47%** | 42% | |
| multi-turn continuity (whole word) | 0% | 20% | **40%** | 27% | |
| replies repeating 8+ words of a training reply | 0% | 0% | **0%** | 12% | |
| mean reply length (words) | 44.9 | 65.1 | 18.8 | 22.9 | 27.1 |

The corpus reference column applies the same lexical measures to the 32 STEVIE replies in the
held-out test split.

**What fine-tuning demonstrably taught.**

* **The conversation format.** The pretrained model never closes a turn: it rambles until the
  length limit or a document-end token, often starting mid-word. STEVIE ends 100% of replies with `<|end|>`, at a conversational
  length (19 words on average, against 27 in the reference replies).
* **The distribution of STEVIE's replies.** Test-split loss on unseen STEVIE replies fell from 4.883
  to 3.944 (perplexity 132 → 52).
* **Some of the voice.** Replies with Australian vocabulary or spelling doubled (9% → 18%), and US
  spellings disappeared (2% → 0%). The commonest markers are "Ha," and "Ah," openers, "mate",
  "barbie", "Aussie" and "g'day", plus friendly follow-up questions ("What's on your mind?"). It
  reaches less than half the reference rate (44%), so the voice is recognisable but weaker than in
  the corpus. No response from any model contains "Crikey" or other impersonation markers.
* **Generation, not regurgitation.** None of STEVIE's 120 responses repeats 8 or more consecutive
  words from any of the 858 training replies.

**What it did not learn.** The content of STEVIE's replies is mostly wrong or incoherent. Asked
"What is 7 times 8?", it says "A hundred and sixty-eighty, and five thousand tonnes is an AI."
Asked "Are you a human or a computer?", it says "I can't hear. But I'm a big model, I'm not a
regular model." The 40% continuity score is lexical: in one passing reply STEVIE does use the
name "Tom", but it says "I'm Tom." A 14M-parameter model with ~46k tokens of conversation learns
how STEVIE *sounds* and *when to stop*, not what is true.

**Why the best-validation checkpoint, not the most Australian one.** The overtrained ablation
(same run continued to ~18 epochs without early stopping) scores higher on Australian markers
(27%) because it has partly memorised the corpus: 12% of its replies quote a training reply
verbatim. It is worse on everything that measures generalisation: test loss on unseen
conversations 4.909 vs 3.944, general-text loss 3.433 vs 2.571 (much more forgetting), and
continuity 27% vs 40%. Selecting by validation loss gives a model that generates new replies.

