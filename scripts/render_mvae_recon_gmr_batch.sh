#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <export_dir>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_DIR="$(realpath "$1")"
RENDER_PYTHON="${RENDER_PYTHON:-/data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/sim2sim/.venv/bin/python}"
GMR_XML="${GMR_XML:-/data/haozhe/zzn/VAR_FM/ws/project/_reference/GMR_view/assets/robots/g1/g1.xml}"

MUJOCO_GL="${MUJOCO_GL:-egl}" "$RENDER_PYTHON" "$ROOT/scripts/render_gmr_view_mp4.py" \
  "$EXPORT_DIR/gt" \
  --out-dir "$EXPORT_DIR/mp4_gt_gmr" \
  --xml "$GMR_XML" \
  --width 640 \
  --height 360 \
  --loops "${LOOPS:-3}" \
  --title-prefix "GMR MVAE ground truth"

MUJOCO_GL="${MUJOCO_GL:-egl}" "$RENDER_PYTHON" "$ROOT/scripts/render_gmr_view_mp4.py" \
  "$EXPORT_DIR/recon" \
  --out-dir "$EXPORT_DIR/mp4_recon_gmr" \
  --xml "$GMR_XML" \
  --width 640 \
  --height 360 \
  --loops "${LOOPS:-3}" \
  --title-prefix "GMR MVAE reconstruction"

echo "[render_mvae_recon_gmr_batch] GT MP4: $EXPORT_DIR/mp4_gt_gmr"
echo "[render_mvae_recon_gmr_batch] Recon MP4: $EXPORT_DIR/mp4_recon_gmr"
