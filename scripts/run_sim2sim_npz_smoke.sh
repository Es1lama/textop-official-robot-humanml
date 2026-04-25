#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <motion.npz> [motion_name] [video.mp4]" >&2
  exit 2
fi

MOTION_PATH="$(realpath "$1")"
MOTION_NAME="${2:-$(basename "${MOTION_PATH%.npz}")}"
VIDEO_PATH="${3:-$(pwd)/exports/${MOTION_NAME}_sim2sim.mp4}"

SIM2SIM_ROOT="${SIM2SIM_ROOT:-/data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/sim2sim}"
POLICY_PATH="${POLICY_PATH:-${SIM2SIM_ROOT}/assets/ckpts/G1TRACKING-04-09_12-22/policy.onnx}"
NET="${NET:-lo}"
SIM_TIMEOUT="${SIM_TIMEOUT:-18s}"
DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-12s}"
SIM_LOG="${SIM_LOG:-/tmp/textop_${MOTION_NAME}_sim2sim.log}"
DEPLOY_LOG="${DEPLOY_LOG:-/tmp/textop_${MOTION_NAME}_deploy.log}"

mkdir -p "$(dirname "$VIDEO_PATH")"
rm -f "$VIDEO_PATH"

pushd "$SIM2SIM_ROOT" >/dev/null
PYTHONUNBUFFERED=1 MUJOCO_GL="${MUJOCO_GL:-egl}" timeout -s INT "$SIM_TIMEOUT" \
  .venv/bin/python src/sim2sim.py \
  --net "$NET" \
  --xml_path assets/g1/g1.xml \
  --auto \
  --show-reference-ghost \
  --ghost-motion-path "$MOTION_PATH" \
  --ghost-motion-name "$MOTION_NAME" \
  --video "$VIDEO_PATH" > "$SIM_LOG" 2>&1 &
SIM_PID=$!

sleep "${DEPLOY_DELAY:-4}"

PYTHONUNBUFFERED=1 timeout -s INT "$DEPLOY_TIMEOUT" \
  .venv/bin/python src/deploy.py \
  --net "$NET" \
  --sim2sim \
  --motion-path "$MOTION_PATH" \
  --motion-name "$MOTION_NAME" \
  --checkpoint-path "$POLICY_PATH" > "$DEPLOY_LOG" 2>&1 || true

wait "$SIM_PID" || true
popd >/dev/null

echo "Video: $VIDEO_PATH"
echo "sim2sim log: $SIM_LOG"
echo "deploy log: $DEPLOY_LOG"
