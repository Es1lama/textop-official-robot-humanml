# textop-official-robot-humanml

Official TextOp baseline adapted to the `robot_humanml_data_v2 + HumanML3D_babel_compat` dataset layout.

This repo keeps the official TextOp training path, but makes the data side usable with our current raw dataset:

1. official `MVAE -> DAR` two-stage training
2. official `57D` motion representation
3. current raw dataset layout:
   - `dataset/HumanML3D_babel_compat/train/labels.json`
   - `dataset/HumanML3D_babel_compat/val/labels.json`
   - `dataset/robot_humanml_data_v2/npz`
   - `dataset/robot_humanml_data_v2/texts`

## What Is Included

This public repo includes:

1. adapted official `TextOpRobotMDAR` code
2. the label files we actually train on
3. handoff docs
4. start / resume scripts
5. experiment results summary

This public repo does **not** include:

1. raw `robot_humanml_data_v2` motion data
2. large checkpoints
3. local logs
4. unrelated investigation notes

Raw data is assumed to already exist in your lab.
Large checkpoints should be uploaded to Hugging Face instead of GitHub.

## Recommended Repo Name

Recommended public repo name:

`textop-official-robot-humanml`

It is short enough to use, and clear about two things:

1. this is the official TextOp baseline line
2. it is adapted to the Robot HumanML style dataset

## Folder Layout

```text
textop-official-robot-humanml/
├── TextOpRobotMDAR/
├── dataset/
│   └── HumanML3D_babel_compat/
│       ├── train/labels.json
│       └── val/labels.json
├── docs/
│   └── HANDOFF.md
└── scripts/
    ├── check_dataset_layout.sh
    ├── run_official_textop_mvae.sh
    ├── run_official_textop_mvae_resume.sh
    └── run_official_textop_dar.sh
```

## Required Raw Data

Put the raw motion data here:

```text
dataset/robot_humanml_data_v2/
├── npz/
├── texts/
├── train.txt
└── test.txt
```

Only `npz/` and `texts/` are required for the official training path in this repo.

## Why The Labels Work Even If They Contain Old Absolute Paths

The `labels.json` files were copied from the original workspace, so `npz_file` entries may still point to an old machine path.

The loader was changed so that if the old absolute path does not exist, it will automatically fall back to:

```text
dataset/robot_humanml_data_v2/npz/<basename>.npz
```

That means you do **not** need to rewrite every `npz_file` entry before training.

## Quick Start

### 1. Create an environment

Use your own Python environment, then install the package:

```bash
cd TextOpRobotMDAR
python -m pip install -e .
```

### 2. Check dataset layout

```bash
bash scripts/check_dataset_layout.sh
```

### 3. Train MVAE from scratch

```bash
bash scripts/run_official_textop_mvae.sh 0
```

### 4. Resume MVAE

```bash
bash scripts/run_official_textop_mvae_resume.sh 0 /path/to/ckpt_1000.pth
```

### 5. Train DAR

```bash
bash scripts/run_official_textop_dar.sh 0 /path/to/mvae_ckpt.pth
```

## What Was Changed

See [docs/HANDOFF.md](docs/HANDOFF.md).

That file explains:

1. exactly which files were changed
2. what each change does
3. how the data mapping works
4. what experiment result we already verified

## Current Verified Result

We already ran a full-data MVAE verification and confirmed that the baseline is trainable on this dataset.

Observed `loss/train_total`:

1. step 1: `0.459224`
2. step 100: `0.100812`
3. step 500: `0.036759`
4. step 1000: `0.030122`
5. step 1390: `0.026353`

This is a `94.26%` drop from step 1 to step 1390.

## Estimated Training Time

Measured on a single RTX 4090 with official `batch_size=512`:

1. full-data MVAE steady speed: about `0.34 ~ 0.35 s/step`
2. `1` pseudo-epoch is about `336 step`
3. `1` pseudo-epoch is about `1.9 minutes`
4. full official MVAE schedule (`100000 step`) is about `10 hours`
5. DAR is heavier and should be budgeted at about `2.5 ~ 4 days` on one RTX 4090

These are engineering estimates for this dataset and this hardware, not the original paper runtime.

## Checkpoints

Please upload checkpoints to Hugging Face rather than committing them to GitHub.

Suggested Hugging Face layout:

```text
official-textop-robot-humanml/
├── mvae/
│   ├── ckpt_1000.pth
│   └── ...
└── dar/
    ├── ckpt_xxx.pth
    └── ...
```
