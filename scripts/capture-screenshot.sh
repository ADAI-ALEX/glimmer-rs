#!/usr/bin/env bash
# Capture a genuine screenshot of the GLIMMER terminal UI on macOS.
#
# Opens a dedicated Terminal.app window (via `open`, no automation permissions needed), resizes it
# with an xterm escape sequence, runs a glimmer command in it, captures *only that window* with
# `screencapture -l <window id>`, then lets the window close itself. The UI can be driven with a
# GLIMMER_TUI_SCRIPT file: real key events through the same handlers a person would use.
#
#   scripts/capture-screenshot.sh <out.png> <seconds> [script-file] -- <glimmer args...>
#
# <seconds> may be a comma-separated list (e.g. 40,60,80): the same window is then captured at
# each time, to <out>-<seconds>.png, so a single run (e.g. a real training run) can be
# photographed at several moments.
set -euo pipefail
out="$1"; secs="$2"; shift 2
script_file=""
if [[ "${1:-}" != "--" ]]; then script_file="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift
root="$(cd "$(dirname "$0")/.." && pwd)"
[[ -x "$root/target/window-id" ]] || xcrun swiftc -O "$root/scripts/window-id.swift" -o "$root/target/window-id"
cols="${GLIMMER_SHOT_COLS:-170}"; rows="${GLIMMER_SHOT_ROWS:-46}"
marker="glimmer-shot-$$"
cmdfile="$(mktemp -t glimmer-shot).command"
cat > "$cmdfile" <<CMD
#!/bin/bash
printf '\e]0;$marker\a\e[8;${rows};${cols}t'
cd '$root'
sleep 0.4
${script_file:+GLIMMER_TUI_SCRIPT='$script_file'} ./target/release/glimmer $*
osascript -e 'tell application "Terminal" to close (every window whose name contains "$marker")' >/dev/null 2>&1 &
exit 0
CMD
chmod +x "$cmdfile"
open -a Terminal "$cmdfile"
wid=""
for _ in $(seq 1 40); do
  wid=$("$root/target/window-id" "$marker" 2>/dev/null || true)
  [[ -n "$wid" ]] && break
  sleep 0.25
done
[[ -z "$wid" ]] && { echo "could not find the Terminal window" >&2; exit 1; }
rm_cmd() { rm -f "$cmdfile"; }
trap rm_cmd EXIT
if [[ "$secs" == *,* ]]; then
  elapsed=0
  IFS=',' read -ra times <<< "$secs"
  for t in "${times[@]}"; do
    sleep "$(echo "$t - $elapsed" | bc)"
    elapsed="$t"
    shot="${out%.png}-$t.png"
    screencapture -o -x -l "$wid" "$shot"
    echo "$shot (window $wid)"
  done
else
  sleep "$secs"
  screencapture -o -x -l "$wid" "$out"
  echo "$out (window $wid)"
fi
