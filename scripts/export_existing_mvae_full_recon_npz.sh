#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/TextOpRobotMDAR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/dataset/HumanML3D_babel_compat}"

GPU="${GPU:-${1:-0}}"
MVAE_CKPT="${MVAE_CKPT:-${2:-}}"
OUT_DIR="${OUT_DIR:-${3:-$ROOT/exports/mvae_full_recon_npz}}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${4:-}}"
MAX_MOTIONS="${MAX_MOTIONS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"

if [ -z "$MVAE_CKPT" ]; then
  echo "Usage: $0 [GPU] [MVAE_CKPT] [OUT_DIR] [SOURCE_MANIFEST]" >&2
  exit 1
fi

if [ ! -f "$MVAE_CKPT" ]; then
  echo "MVAE checkpoint not found: $MVAE_CKPT" >&2
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

ARGS=(
  --config-name=export_mvae_full_recon_npz
  "data.datadir=$DATA_DIR"
  "data.batch_size=$BATCH_SIZE"
  "ckpt.vae=$MVAE_CKPT"
  "export.out_dir=$OUT_DIR"
  "export.max_motions=$MAX_MOTIONS"
)

if [ -n "$SOURCE_MANIFEST" ]; then
  ARGS+=("export.source_manifest=$SOURCE_MANIFEST")
fi

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m robotmdar.cli "${ARGS[@]}"
