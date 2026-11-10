# Demo scripts

Input scripts for `GLIMMER_TUI_SCRIPT`, used with `scripts/capture-screenshot.sh` to capture the
README screenshots from the real binary. Each line is a real key event (or a wait) fed through the
same handlers as a person typing; see `crates/glimmer-terminal/src/lib.rs` for the format.

| script | what it does |
|---|---|
| `train-sft.txt` | opens the training lab, selects `stevie-sft`, starts it, waits for the run to finish |
| `chat.txt` | sends four held-out prompts to STEVIE in the chat screen |
| `eval-stevie.txt` | selects STEVIE's report in the evaluation screen |
| `hold.txt`, `hold-long.txt` | keep a screen open long enough to capture it |

How the README screenshots were made:

```bash
# STEVIE's real fine-tuning run, photographed at several moments (60 s and 75 s were used)
scripts/capture-screenshot.sh shots/training.png 30,45,60,75,90,105 scripts/demo/train-sft.txt -- tui --screen train

scripts/capture-screenshot.sh shots/chat.png 18 scripts/demo/chat.txt -- chat
scripts/capture-screenshot.sh shots/evaluation.png 9 scripts/demo/eval-stevie.txt -- tui --screen evaluation
for s in home models datasets experiments settings; do
  scripts/capture-screenshot.sh shots/$s.png 9 scripts/demo/hold.txt -- tui --screen $s
done
# (hardware.png was captured the same way while pretraining was running)

# Remove the Terminal title bar (it shows the local user name) and scale for the README.
xcrun swiftc -O scripts/crop-screenshot.swift -o target/crop-screenshot
target/crop-screenshot shots/chat.png assets/screenshots/chat.png 64 1440
```

Chat replies are sampled live and differ between runs. The chat screenshot is a single take,
used as it came out.
