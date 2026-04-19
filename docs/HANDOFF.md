# Handoff

This document is meant for the next person who will run or continue this baseline.
It focuses on concrete code locations and concrete logic, not abstract terminology.

For the latest experiment numbers, also read:

`docs/CURRENT_RESULTS.md`

## 1. What This Repo Is

This repo is the official TextOp baseline adapted to our current dataset layout.

What stayed on the official path:

1. official two-stage training order: `MVAE -> DAR`
2. official `history_len=2`
3. official `future_len=8`
4. official `num_primitive=8`
5. official `nfeats=57`
6. official training entrypoints: `train_mvae` and `train_dar`

What changed:

1. the loader can read our `labels.json + npz` data directly
2. the label path resolution is portable
3. a 23DoF robot XML is included because the official release folder did not have it
4. simple launch scripts were added

## 2. The Main Code Change

The main code change is here:

`TextOpRobotMDAR/robotmdar/dataloader/data.py`

### 2.1 What the new loader does

The loader now supports this layout:

```text
dataset/HumanML3D_babel_compat/train/labels.json
dataset/HumanML3D_babel_compat/val/labels.json
dataset/robot_humanml_data_v2/npz/*.npz
```

Before this change, the official code expected:

1. `train.pkl`
2. `val.pkl`
3. `statistics.yaml`

Now it can also read the current raw format directly.

### 2.2 Functions you should read first

Read these functions in order:

1. `_raw_npz_to_motion_dict`
   Purpose:
   Turn one raw IsaacLab-style `.npz` file into the official TextOp motion dictionary.

2. `_resolve_label_npz_path`
   Purpose:
   Make old absolute `npz_file` paths inside `labels.json` still usable on a new machine.

3. `_load_statistics`
   Purpose:
   If `statistics.yaml` is missing, infer `fps` directly from the raw dataset.

4. `_load_raw_labels_npz_dataset`
   Purpose:
   Read all samples from `train/labels.json` or `val/labels.json`, load the linked `.npz`, and store them in official training format.

## 3. How One Label Becomes One Training Sample

Take one record from:

`dataset/HumanML3D_babel_compat/train/labels.json`

Each record contains:

1. `npz_file`
2. `humanml_id`
3. `duration`
4. `length`
5. `fps`
6. `frame_ann`

The loader uses `npz_file` to find the raw motion file.

Then `_raw_npz_to_motion_dict` reads:

1. `joint_pos`
2. `body_pos_w`
3. `body_quat_w`
4. `fps`

Then it builds the official fields:

1. `root_trans_offset`
   Source:
   `body_pos_w[:, 0, :3]`

2. `root_rot`
   Source:
   `body_quat_w[:, 0, :]`
   Extra handling:
   convert quaternion order from `wxyz` to `xyzw`

3. `dof`
   Source:
   `joint_pos`
   Extra handling:
   convert 29DoF raw joints to the official 23DoF subset

4. `contact_mask`
   Source:
   reconstructed from left and right foot trajectories in `body_pos_w`

After that, the code uses the official feature conversion path to build `57D` training features.

## 4. The Path Problem And How It Was Solved

The copied `labels.json` files still contain old absolute paths like:

```text
/data/haozhe/zzn/VAR_FM/ws/project/dataset/robot_humanml_data_v2/npz/000000.npz
```

That is machine-specific and should not be trusted in a public repo.

The fix is in:

`_resolve_label_npz_path`

The logic is:

1. if the original path exists, use it
2. else, if it is a relative path, try `split_dir/npz/...`
3. else, fall back to:
   `dataset/robot_humanml_data_v2/npz/<basename>.npz`

This is why users do not need to rewrite every label entry manually.

## 5. Why The XML File Is Included

File:

`TextOpRobotMDAR/description/g1_23dof_lock_wrist_fitmotionONLY.xml`

Reason:

The official training code initializes FK and geometry-related losses through the robot XML.
The official release folder here did not already contain the expected 23DoF XML, so it was added.

Without this file, the training code would fail before entering stable training.

## 6. Scripts Added For Handoff

Files:

1. `scripts/check_dataset_layout.sh`
2. `scripts/run_official_textop_mvae.sh`
3. `scripts/run_official_textop_mvae_resume.sh`
4. `scripts/run_official_textop_dar.sh`
5. `scripts/run_official_textop_vis_mvae.sh`
6. `scripts/run_official_textop_vis_dar.sh`

### 6.1 `check_dataset_layout.sh`

This checks that:

1. `train/labels.json` exists
2. `val/labels.json` exists
3. raw `npz/` exists
4. raw `texts/` exists
5. at least the first batch of labels can resolve to real motion files

### 6.2 `run_official_textop_mvae.sh`

This starts official MVAE training from scratch.

### 6.3 `run_official_textop_mvae_resume.sh`

This continues MVAE from an existing checkpoint.

### 6.4 `run_official_textop_dar.sh`

This starts or resumes DAR.
It accepts:

1. a trained MVAE checkpoint
2. optionally an existing DAR checkpoint if you want to continue DAR

### 6.5 `run_official_textop_vis_mvae.sh`

This launches the official MVAE reconstruction viewer.
It is the quickest way to show:

1. ground-truth future motion
2. MVAE reconstructed future motion

No extra reconstruction code was written for the handoff.
This is just a thin wrapper over the official `vis_mvae` entrypoint.

### 6.6 `run_official_textop_vis_dar.sh`

This launches the official DAR visualization path.
Use it after DAR training if the next person wants to compare generated motion against the ground truth in the viewer.

## 7. Current Verified Experiment Result

We already verified on full data that this baseline is trainable.

Measured `loss/train_total`:

1. step 1: `0.459224`
2. step 10: `0.306104`
3. step 50: `0.166779`
4. step 100: `0.100812`
5. step 500: `0.036759`
6. step 1000: `0.030122`
7. step 1390: `0.026353`

This corresponds to:

1. `78.05%` drop by step 100
2. `92.00%` drop by step 500
3. `93.44%` drop by step 1000
4. `94.26%` drop by step 1390

That is enough to show the current baseline is not only launchable, but truly trainable.

We later resumed from `ckpt_1000.pth` and stopped the run at step `13896` to free GPU for DAR.
The saved checkpoint history includes:

1. `ckpt_5000.pth`
2. `ckpt_10000.pth`

Recent loss values during that resumed phase:

1. step 5000: `0.011362`
2. step 10000: `0.009063`
3. step 13896: `0.006279`
4. last-20 mean: `0.006592`
5. last-20 min/max: `0.005702 ~ 0.008025`

Interpretation in plain language:

1. the loss is still moving, so this is not a strict final convergence claim
2. but the curve is already much flatter than the early stage
3. for the purpose of switching over to DAR, this is already a reasonable stopping point
4. because the stop happened before the next save point at step `15000`, the practical checkpoint to hand to DAR right now is `ckpt_10000.pth`

We then ran DAR from that MVAE checkpoint and trained it successfully to `step 50000`.

Important DAR numbers:

1. `loss/train_total`: `0.498519 -> 0.100912`
2. `loss/train_rec`: `0.109399 -> 0.021164`
3. `loss/train_latent_rec`: `0.389046 -> 0.079731`
4. recent `loss/train_total` mean over last 20 points: `0.105537`

Plain-language interpretation:

1. DAR is clearly learning
2. this is already a valid checkpoint to hand off
3. but it is still a mid-stage checkpoint, not the end of the full official schedule

## 8. Runtime Estimate

Measured on one RTX 4090, official `batch_size=512`:

1. MVAE steady speed: `0.34 ~ 0.35 s/step`
2. one pseudo-epoch: about `336 step`
3. one pseudo-epoch: about `1.9 minutes`
4. full official MVAE schedule: about `10 hours`
5. DAR should be budgeted at about `2.5 ~ 4 days`

## 9. What Should Be Uploaded Where

### GitHub repo

Put here:

1. code
2. label files
3. handoff docs
4. launch scripts
5. result summary
6. reconstruction / visualization launch scripts
7. a short current-results note such as `docs/CURRENT_RESULTS.md`

### Hugging Face

Put here:

1. `ckpt_*.pth`
2. optional training logs you want to preserve

### Do Not Put In GitHub

1. raw `robot_humanml_data_v2`
2. local `logs/`
3. local virtual environments
4. big intermediate caches
