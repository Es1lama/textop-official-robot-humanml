#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/TextOpRobotMDAR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/dataset/HumanML3D_babel_compat}"

GPU="${GPU:-${1:-0}}"
DAR_CKPT="${DAR_CKPT:-${2:-}}"
VAE_CKPT="${VAE_CKPT:-${3:-}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
EXPNAME="${EXPNAME:-OfficialTextOpVisDAR}"

if [ -z "$DAR_CKPT" ]; then
  echo "Usage: $0 [GPU] [DAR_CKPT] [VAE_CKPT]" >&2
  echo "Example: $0 0 /path/to/dar_ckpt.pth /path/to/mvae_ckpt.pth" >&2
  exit 1
fi

if [ ! -f "$DAR_CKPT" ]; then
  echo "DAR checkpoint not found: $DAR_CKPT" >&2
  exit 1
fi

if [ -n "$VAE_CKPT" ] && [ ! -f "$VAE_CKPT" ]; then
  echo "VAE checkpoint not found: $VAE_CKPT" >&2
  exit 1
fi

ARGS=(
  --config-name=vis_dar
  "expname=$EXPNAME"
  "data.datadir=$DATA_DIR"
  "data.batch_size=$BATCH_SIZE"
  "guidance_scale=$GUIDANCE_SCALE"
  "ckpt.dar=$DAR_CKPT"
)

if [ -n "$VAE_CKPT" ]; then
  ARGS+=("ckpt.vae=$VAE_CKPT")
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
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m robotmdar.cli "${ARGS[@]}"
