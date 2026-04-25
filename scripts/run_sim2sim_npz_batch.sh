#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <npz_dir> <out_dir> [max_count]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPZ_DIR="$(realpath "$1")"
OUT_DIR="$(realpath -m "$2")"
MAX_COUNT="${3:-999999}"
PYTHON_BIN="${PYTHON_BIN:-/data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/sim2sim/.venv/bin/python}"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/annotated" "$OUT_DIR/logs"

count=0
for motion_path in "$NPZ_DIR"/*.npz; do
  [[ -e "$motion_path" ]] || break
  if (( count >= MAX_COUNT )); then
    break
  fi
  motion_name="$(basename "${motion_path%.npz}")"
  raw_video="$OUT_DIR/raw/${motion_name}_sim2sim_raw.mp4"
  annotated_video="$OUT_DIR/annotated/${motion_name}_sim2sim.mp4"
  SIM_LOG="$OUT_DIR/logs/${motion_name}_sim2sim.log" \
  DEPLOY_LOG="$OUT_DIR/logs/${motion_name}_deploy.log" \
    bash "$ROOT/scripts/run_sim2sim_npz_smoke.sh" "$motion_path" "$motion_name" "$raw_video"
  "$PYTHON_BIN" "$ROOT/scripts/overlay_mp4_text.py" \
    --input "$raw_video" \
    --output "$annotated_video" \
    --motion-npz "$motion_path" \
    --title "sim2sim tracker: ${motion_name}"
  echo "[run_sim2sim_npz_batch] saved $annotated_video"
  count=$((count + 1))
done

echo "[run_sim2sim_npz_batch] complete: $count videos under $OUT_DIR/annotated"
