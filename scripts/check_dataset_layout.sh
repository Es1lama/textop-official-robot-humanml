#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL_DIR="${LABEL_DIR:-$ROOT/dataset/HumanML3D_babel_compat}"
RAW_DIR="${RAW_DIR:-$ROOT/dataset/robot_humanml_data_v2}"
EXTERNAL_RAW_DIR="/data/haozhe/zzn/VAR_FM/ws/project/dataset/robot_humanml_data_v2"

if [ ! -d "$RAW_DIR/npz" ] && [ -d "$EXTERNAL_RAW_DIR/npz" ]; then
  RAW_DIR="$EXTERNAL_RAW_DIR"
fi

echo "LABEL_DIR=$LABEL_DIR"
echo "RAW_DIR=$RAW_DIR"

test -f "$LABEL_DIR/train/labels.json"
test -f "$LABEL_DIR/val/labels.json"
test -d "$RAW_DIR/npz"
test -d "$RAW_DIR/texts"

python - "$LABEL_DIR" "$RAW_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

label_dir = Path(sys.argv[1])
raw_dir = Path(sys.argv[2])

for split in ["train", "val"]:
    labels = json.loads((label_dir / split / "labels.json").read_text())
    print(f"{split}: {len(labels)} labels")
    missing = []
    for item in labels[:100]:
        name = Path(item["npz_file"]).name
        if not (raw_dir / "npz" / name).exists():
            missing.append(name)
    print(f"{split}: sample missing count (first 100) = {len(missing)}")
    if missing:
        print("missing examples:", missing[:10])
        sys.exit(1)

print("dataset layout looks good")
PY
