# Demo scripts

Input scripts for `GLIMMER_TUI_SCRIPT`, used with `scripts/capture-screenshot.sh` to capture the
README screenshots from the real binary. Each line is a real key event (or a wait) fed through the
same handlers as a person typing; see `crates/glimmer-terminal/src/lib.rs` for the format.

```bash
scripts/capture-screenshot.sh assets/screenshots/chat.png 14 scripts/demo/chat.txt -- chat
scripts/capture-screenshot.sh assets/screenshots/hardware.png 10 scripts/demo/hold.txt -- tui --screen hardware
```
