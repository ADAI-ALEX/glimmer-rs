# Inference and chat

Implementation: [`crates/glimmer-inference`](../crates/glimmer-inference). STEVIE's replies are
produced by its own trained weights; GLIMMER never calls an external model or API.

## Generation loop

```text
prompt tokens ──▶ prefill (one forward over the uncached suffix) ──▶ logits(last)
      ┌──────────────────────────────────────────────────────────────────┘
      ▼
  sample next token ──▶ end-of-turn?  ──yes──▶ stop
      │ no
      ▼
  stream decoded text ──▶ forward_cached([token]) ──▶ logits ──▶ (repeat)
```

* **Prefix reuse.** The engine remembers which tokens are in the KV cache. A new prompt that shares
  a prefix with the cache (the system prompt and earlier turns, in chat) only computes the new
  suffix; the cache is truncated to the shared prefix first. The test
  `reused_cache_gives_same_result_as_fresh_cache` checks that reuse never changes outputs.
* **Stop tokens.** Generation ends on `<|end|>` — which STEVIE learned to emit because SFT
  supervises it — or if the model starts another role tag (`<|user|>`, `<|system|>`,
  `<|assistant|>`, `<|endoftext|>`). Padding and reserved tokens are never sampled.
* **Streaming.** A [`StreamDecoder`](../crates/glimmer-tokenizer/src/stream.rs) buffers incomplete
  UTF-8 sequences, so a half-emitted emoji never flashes as a replacement character.
* **Limits.** `max_new_tokens`, and the context window: generation stops before the cache exceeds
  the model's context length.

## Sampling

| setting | default | effect |
|---|---|---|
| `temperature` | 0.8 | scales logits; 0 = greedy argmax |
| `top_k` | 40 | sample only among the k most likely tokens |
| `top_p` | 0.92 | nucleus: smallest set covering this probability mass |
| `repetition_penalty` | 1.15 | CTRL-style: divides positive / multiplies negative logits of recently generated tokens |
| `repetition_window` | 64 | how many recent tokens the penalty considers |
| `max_new_tokens` | 160 | reply length cap |
| `seed` | 0 | 0 = time-based; any other value makes sampling reproducible |

All of these can be changed live in the TUI (`9 settings`, or `/temperature`, `/topk`, `/topp`).

## Context-window management

[`ChatSession::fit_prompt`](../crates/glimmer-inference/src/chat.rs) assembles
`system prompt + history + latest user message + <|assistant|>`. If it would not leave room for the
reply budget within the context window, the oldest *exchanges* (user + STEVIE pairs) are dropped
until it fits. The system prompt and the latest message are always kept. The chat UI reports how
many messages were trimmed.

## Chat sessions

Conversations save to `conversations/<name>.json` (`/save`, `/load`) with the model name, system
prompt, sampling settings and messages. Restoring a session resets the KV cache; the next reply
re-prefills the restored history.

## Interfaces

```bash
glimmer chat                         # full-screen chat with STEVIE
glimmer chat --plain                 # line-based REPL (works over SSH, in scripts)
glimmer generate "G'day, who are you?"
glimmer generate --raw "Once upon a time"   # plain continuation, no chat template
```

TUI commands: `/help /clear /save [name] /load [name] /model [name] /stats /temperature <t>
/topk <k> /topp <p> /system [text|off] /quit`. `Esc` stops a reply mid-stream.

The welcome line ("How ya going, mate? I'm Stevie. What's on your mind?") is a fixed UI
greeting shown when a chat opens. It is styled as a welcome, is not part of the model's context,
and never substitutes for a generated reply.

## Performance notes

Single-sequence decoding is latency-bound: each token is one small forward pass plus a
device→host copy of the logits for sampling. KV caching turns the per-token cost from
O(prompt length) into O(1) forward work; `glimmer bench` measures both paths (see
[performance.md](performance.md)).
