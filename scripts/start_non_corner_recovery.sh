#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 /path/to/image-library" >&2
  exit 2
fi

ROOT="$1"
STATE_DIR="${STATE_DIR:-runs/non-corner-recovery/live}"
STOP_FILE="${STOP_FILE:-$STATE_DIR/STOP}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/run.log}"
SESSION_NAME="${SESSION_NAME:-cornercrop-non-corner-recovery}"

mkdir -p "$STATE_DIR"

RUN_CMD="cd \"$(pwd)\" && uv run python -m cornercrop.non_corner_recovery \"$ROOT\" --state-dir \"$STATE_DIR\" --stop-file \"$STOP_FILE\" --resource-profile \"${RESOURCE_PROFILE:-balanced}\" --max-workers \"${MAX_WORKERS:-3}\" --resource-poll-interval \"${RESOURCE_POLL_INTERVAL:-5}\" --heartbeat-interval \"${HEARTBEAT_INTERVAL:-30}\" --progress-every \"${PROGRESS_EVERY:-25}\" --max-removed-area \"${MAX_REMOVED_AREA:-0.30}\" --max-crop \"${MAX_CROP:-0.30}\" >> \"$LOG_FILE\" 2>&1"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME"
    echo "monitor: scripts/watch_non_corner_recovery.py --state-dir \"$STATE_DIR\" --watch --interval 60"
    exit 0
  fi
  tmux new-session -d -s "$SESSION_NAME" "$RUN_CMD"
  tmux display-message -p -t "$SESSION_NAME" "#{pane_pid}" > "$STATE_DIR/pid"
  echo "started tmux_session=$SESSION_NAME pane_pid=$(cat "$STATE_DIR/pid") state_dir=$STATE_DIR log=$LOG_FILE stop_file=$STOP_FILE"
else
  nohup bash -lc "$RUN_CMD" >> "$LOG_FILE" 2>&1 < /dev/null &
  echo "$!" > "$STATE_DIR/pid"
  disown "$(cat "$STATE_DIR/pid")" 2>/dev/null || true
  echo "started pid=$(cat "$STATE_DIR/pid") state_dir=$STATE_DIR log=$LOG_FILE stop_file=$STOP_FILE"
fi
