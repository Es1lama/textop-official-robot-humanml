#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-}"
SESSION="${SESSION:-textop_monitor}"
INTERVAL="${INTERVAL:-300}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "$RUN_DIR" ]; then
  echo "Usage: $0 /path/to/run-dir" >&2
  exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

LOG_FILE="$RUN_DIR/monitor.log"
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && PYTHON_BIN='$PYTHON_BIN' INTERVAL='$INTERVAL' bash scripts/monitor_textop_run.sh '$RUN_DIR' 2>&1 | tee -a '$LOG_FILE'"

echo "started tmux monitor: $SESSION"
echo "log: $LOG_FILE"
