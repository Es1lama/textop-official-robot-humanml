#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
INTERVAL="${INTERVAL:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "$RUN_DIR" ]; then
  echo "Usage: $0 /path/to/run-dir" >&2
  echo "Set INTERVAL=300 to repeat every 5 minutes." >&2
  exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

monitor_once() {
  date '+%F %T'
  echo "RUN_DIR=$RUN_DIR"

  echo
  echo "[checkpoints]"
  find "$RUN_DIR" -maxdepth 1 -name 'ckpt_*.pth' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -n 5 \
    | awk '{print $2}' \
    | xargs -r ls -lh

  echo
  echo "[recent log errors]"
  if [ -f "$RUN_DIR/run.log" ]; then
    rg -n "Traceback|RuntimeError|CUDA out of memory|NaN|nan|Killed|Error|Exception" "$RUN_DIR/run.log" | tail -n 20 || true
  else
    echo "run.log not found"
  fi

  echo
  echo "[tensorboard scalars]"
  "$PYTHON_BIN" - "$RUN_DIR" <<'PY' || true
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:
    print(f"tensorboard unavailable: {exc}")
    raise SystemExit(0)

try:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
except Exception as exc:
    print(f"failed to load events: {exc}")
    raise SystemExit(0)

tags = ea.Tags().get("scalars", [])
keys = [
    "loss/train_total",
    "loss/train_rec",
    "loss/train_latent_rec",
    "extras/lr",
    "extras/stage",
    "extras/grad_norm",
]
for key in keys:
    if key not in tags:
        continue
    sc = ea.Scalars(key)
    if not sc:
        continue
    vals = [item.value for item in sc]
    tail20 = vals[-20:] if len(vals) >= 20 else vals
    tail100 = vals[-100:] if len(vals) >= 100 else vals
    first = vals[0]
    last = vals[-1]
    drop = (first - last) / first * 100 if first else 0.0
    print(
        f"{key}: count={len(sc)} first_step={sc[0].step} first={first:.6g} "
        f"last_step={sc[-1].step} last={last:.6g} min={min(vals):.6g} "
        f"drop={drop:.2f}% tail20_mean={sum(tail20)/len(tail20):.6g} "
        f"tail100_mean={sum(tail100)/len(tail100):.6g}"
    )
PY

  echo
  echo "[gpu summary]"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader || true
}

if [ "$INTERVAL" = "0" ]; then
  monitor_once
else
  while true; do
    monitor_once
    echo
    echo "sleeping ${INTERVAL}s"
    sleep "$INTERVAL"
  done
fi
