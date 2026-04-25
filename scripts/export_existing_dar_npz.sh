#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/TextOpRobotMDAR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/dataset/HumanML3D_babel_compat}"

GPU="${GPU:-${1:-0}}"
DAR_CKPT="${DAR_CKPT:-${2:-}}"
OUT_DIR="${OUT_DIR:-${3:-$ROOT/exports/dar_npz}}"
MAX_MOTIONS="${MAX_MOTIONS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
PREFIX="${PREFIX:-dar}"

if [ -z "$DAR_CKPT" ]; then
  echo "Usage: $0 [GPU] [DAR_CKPT] [OUT_DIR]" >&2
  exit 1
fi

if [ ! -f "$DAR_CKPT" ]; then
  echo "DAR checkpoint not found: $DAR_CKPT" >&2
  exit 1
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ISAAC_UTILS_PATH="${ISAAC_UTILS_PATH:-$ROOT/deps/isaac_utils}"
LEGACY_ISAAC_UTILS="/data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/_reference/TextOp_official/deps/isaac_utils"
if [ -d "$ISAAC_UTILS_PATH" ]; then
  export PYTHONPATH="$ISAAC_UTILS_PATH:$PYTHONPATH"
elif [ -d "$LEGACY_ISAAC_UTILS" ]; then
  export PYTHONPATH="$LEGACY_ISAAC_UTILS:$PYTHONPATH"
fi

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m robotmdar.cli \
  --config-name=export_dar_npz \
  "data.datadir=$DATA_DIR" \
  "data.batch_size=$BATCH_SIZE" \
  "ckpt.dar=$DAR_CKPT" \
  "guidance_scale=$GUIDANCE_SCALE" \
  "export.out_dir=$OUT_DIR" \
  "export.max_motions=$MAX_MOTIONS" \
  "export.prefix=$PREFIX"
