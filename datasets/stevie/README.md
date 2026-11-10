# The STEVIE conversation corpus

An original corpus of Australian-English conversations used to fine-tune STEVIE, GLIMMER's
demonstration model.

## Provenance

* **Authorship.** Every conversation in [`conversations/`](conversations) was written by Claude
  (Anthropic), acting as GLIMMER's engineer, in November 2026, specifically for this project. They
  were written by hand, one conversation at a time — not generated from templates — to establish a
  consistent, warm, curious and gently cheeky Australian character.
* **Not derived from other data.** No existing dataset, chat log, transcript, broadcast or real
  person's speech was copied, paraphrased or used as a source.
* **Synthetic and AI-authored.** This is synthetic training data. It is documented as such so that
  anyone using it knows exactly what it is.
* **Licence.** Apache-2.0, like the rest of the repository.

## The character

STEVIE is a fictional AI. Its name is an affectionate nod to the spirit of Australian wildlife
enthusiasm, but STEVIE is not an impersonation of any real person, does not reproduce anyone's
voice or catchphrases, and claims no affiliation or endorsement. When asked, STEVIE says it is an
AI built with GLIMMER. The corpus QC (`glimmer data prepare`) warns if STEVIE's lines contain
impersonation markers.

Style goals: natural Australian vocabulary and spelling (mate, reckon, heaps, arvo, no worries,
colour, favourite) used *sparingly* rather than stacked into a caricature; plain, accurate
explanations; follow-up questions; remembering details from earlier in a conversation; pointing
people to real help (e.g. Lifeline 13 11 14) in emotionally serious situations; honesty about being
a small model that makes mistakes.

## Format

```text
### <unique-id> | <topic>
U: user message (one line)
S: STEVIE's reply (one line)
```

Turns alternate starting with the user and ending with STEVIE. `glimmer data prepare --source
stevie-conversations` validates structure, removes exact and near-duplicates (MinHash, Jaccard ≥
0.7) and writes `data/corpora/stevie-conversations.jsonl`.

## Files

| file | topic | | file | topic |
|---|---|---|---|---|
| `01-greetings.txt` | greetings and farewells | | `09-tech.txt` | technology, Rust, AI |
| `02-identity.txt` | who STEVIE is, limits | | `10-science.txt` | science and maths |
| `03-smalltalk.txt` | everyday small talk | | `11-advice.txt` | advice, learning, opinions |
| `04-feelings.txt` | emotional support | | `12-humour.txt` | jokes and banter |
| `05-wildlife.txt` | wildlife, conservation | | `13-multiturn.txt` | context-dependent chats |
| `06-food.txt` | food and cooking | | `14-nature-travel.txt` | outdoors and travel |
| `07-australia.txt` | geography, culture, sport | | `15-general.txt` | general knowledge |
| `08-slang.txt` | Australian English explained | | `16-everyday.txt` | multi-turn everyday chats |

## Held-out evaluation prompts

[`eval_prompts.toml`](eval_prompts.toml) holds 40 prompts (including five multi-turn continuity
probes) that are never trained on. The dataset build excludes any training conversation whose user
turn closely matches one of them.

## Limitations

The corpus is small (~38k training tokens). It can teach a 14M-parameter model a conversational
format and a recognisable voice, not reliable knowledge. Factual statements in the corpus were
written carefully, but a model this size will still produce confident mistakes.
