#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/TextOpRobotMDAR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/dataset/HumanML3D_babel_compat}"

GPU="${GPU:-${1:-0}}"
EXPNAME="${EXPNAME:-OfficialTextOpMVAE}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_EVERY="${EVAL_EVERY:-1000000}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m robotmdar.cli \
  --config-name=train_mvae \
  expname="$EXPNAME" \
  data.datadir="$DATA_DIR" \
  train.manager.save_every="$SAVE_EVERY" \
  train.manager.eval_every="$EVAL_EVERY"
