#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/TextOpRobotMDAR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/dataset/HumanML3D_babel_compat}"

GPU="${GPU:-${1:-0}}"
CKPT="${CKPT:-${2:-}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EXPNAME="${EXPNAME:-OfficialTextOpVisMVAE}"

if [ -z "$CKPT" ]; then
  echo "Usage: $0 [GPU] [MVAE_CKPT]" >&2
  echo "Example: $0 0 /path/to/ckpt_10000.pth" >&2
  exit 1
fi

if [ ! -f "$CKPT" ]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m robotmdar.cli \
  --config-name=vis_mvae \
  expname="$EXPNAME" \
  data.datadir="$DATA_DIR" \
  data.batch_size="$BATCH_SIZE" \
  ckpt.vae="$CKPT"
