#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <export_dir>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_DIR="$(realpath "$1")"
RENDER_PYTHON="${RENDER_PYTHON:-/data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/sim2sim/.venv/bin/python}"

MUJOCO_GL="${MUJOCO_GL:-egl}" "$RENDER_PYTHON" "$ROOT/scripts/render_npz_reference_mp4.py" \
  "$EXPORT_DIR/gt" \
  --out-dir "$EXPORT_DIR/mp4_gt" \
  --width 640 \
  --height 360 \
  --loops "${LOOPS:-3}" \
  --title-prefix "MVAE ground truth"

MUJOCO_GL="${MUJOCO_GL:-egl}" "$RENDER_PYTHON" "$ROOT/scripts/render_npz_reference_mp4.py" \
  "$EXPORT_DIR/recon" \
  --out-dir "$EXPORT_DIR/mp4_recon" \
  --width 640 \
  --height 360 \
  --loops "${LOOPS:-3}" \
  --title-prefix "MVAE reconstruction"

echo "[render_mvae_recon_batch] GT MP4: $EXPORT_DIR/mp4_gt"
echo "[render_mvae_recon_batch] Recon MP4: $EXPORT_DIR/mp4_recon"
