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
5. raw IsaacLab joint order is reordered into TextOp/GMR grouped joint order before training

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
   reorder IsaacLab 29DoF raw joints into TextOp/GMR grouped order, then keep the official 23 non-wrist joints

   The exact raw indices used are:

   ```python
   [0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8,
    11, 15, 19, 21,
    12, 16, 20, 22]
   ```

   Plain meaning:
   left leg first, right leg second, waist third, left arm fourth, right arm fifth.

4. `contact_mask`
   Source:
   reconstructed from left and right foot trajectories in `body_pos_w`

After that, the code uses the official feature conversion path to build `57D` training features.

Important:

Any checkpoint trained before the joint-order fix learned the wrong channel semantics.
Do not use old MVAE/DAR checkpoints as paper-facing results after this correction.

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
7. `scripts/export_existing_dar_npz.sh`
8. `scripts/run_sim2sim_npz_smoke.sh`
9. `scripts/render_npz_reference_mp4.py`
10. `scripts/run_sim2sim_npz_batch.sh`
11. `scripts/overlay_mp4_text.py`
12. `scripts/export_existing_mvae_recon_npz.sh`
13. `scripts/render_mvae_recon_batch.sh`
14. `scripts/render_gmr_view_mp4.py`
15. `scripts/render_mvae_recon_gmr_batch.sh`

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

This is just a thin wrapper over the official `vis_mvae` entrypoint.
For batch MP4 export, use `export_existing_mvae_recon_npz.sh` and `render_mvae_recon_batch.sh` below.

### 6.6 `run_official_textop_vis_dar.sh`

This launches the official DAR visualization path.
Use it after DAR training if the next person wants to compare generated motion against the ground truth in the viewer.

### 6.7 `export_existing_dar_npz.sh`

This exports an existing DAR checkpoint into plain `.npz` motion files.

It writes:

1. `sim2sim/*.npz`
2. `sim2sim_gt/*.npz`
3. `tracker/*/motion.npz`
4. `tracker_gt/*/motion.npz`

The sim2sim `.npz` contains:

1. `fps`
2. `root_pos`
3. `root_rot`
4. `dof_pos`
5. `joint_names`
6. `body_names`
7. `local_body_pos`

The first four fields are for tracking.
`body_names` and `local_body_pos` are for drawing the reference ghost in sim2sim.

Batch export:

```bash
MAX_MOTIONS=32 BATCH_SIZE=8 bash scripts/export_existing_dar_npz.sh <GPU> <DAR_CKPT> <OUT_DIR>
```

### 6.8 `run_sim2sim_npz_smoke.sh`

This launches the local sim2sim tracker against one exported `.npz` and records an MP4.

Example:

```bash
bash scripts/run_sim2sim_npz_smoke.sh \
  exports/smoke_dar_npz/sim2sim/dar_0000.npz \
  dar_0000 \
  exports/smoke_dar_npz/dar_0000_sim2sim.mp4
```

It uses the existing sim2sim checkout by default:

```text
/data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/sim2sim
```

Override with `SIM2SIM_ROOT=/path/to/sim2sim` if needed.

### 6.9 `render_npz_reference_mp4.py`

This renders exported `.npz` target motions directly to MP4.

This is the fast official/reference visualization path:

1. it does not run a controller
2. it does not test whether the tracker can follow the motion
3. it shows what DAR generated as the target motion

Example:

```bash
MUJOCO_GL=egl /data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/sim2sim/.venv/bin/python \
  scripts/render_npz_reference_mp4.py \
  exports/batch_dar50000_mp4/sim2sim \
  --out-dir exports/batch_dar50000_mp4/official_mp4 \
  --width 640 \
  --height 360 \
  --loops 3
```

### 6.10 `run_sim2sim_npz_batch.sh`

This runs a directory of exported `.npz` files through sim2sim one by one.

Example:

```bash
SIM_TIMEOUT=18s DEPLOY_TIMEOUT=12s bash scripts/run_sim2sim_npz_batch.sh \
  exports/batch_dar50000_mp4/sim2sim \
  exports/batch_dar50000_mp4/sim2sim_mp4 \
  4
```

It writes:

1. raw sim2sim videos under `raw/`
2. text-annotated videos under `annotated/`
3. sim2sim and deploy logs under `logs/`

### 6.11 `overlay_mp4_text.py`

This reads `primary_text` or `texts` from the motion `.npz` and burns it into an MP4.

It is used by `run_sim2sim_npz_batch.sh`.

### 6.12 `export_existing_mvae_recon_npz.sh`

This exports MVAE reconstruction samples from an existing MVAE checkpoint.

It writes:

1. `gt/*.npz`
2. `recon/*.npz`
3. `manifest.jsonl`

Meaning:

1. `gt` is the original validation motion
2. `recon` is the MVAE encode/decode result
3. this is an MVAE reconstruction check, not text-conditioned generation

Example:

```bash
MAX_MOTIONS=4 BATCH_SIZE=4 PYTHON_BIN=/data/haozhe/miniconda3/envs/motiongpt/bin/python \
  bash scripts/export_existing_mvae_recon_npz.sh \
  4 \
  /path/to/mvae/ckpt_10000.pth \
  exports/batch_mvae_recon_10000_mp4
```

### 6.13 `render_mvae_recon_batch.sh`

This renders both sides of an MVAE reconstruction batch:

1. `gt/*.npz` to `mp4_gt/*.mp4`
2. `recon/*.npz` to `mp4_recon/*.mp4`

Example:

```bash
LOOPS=3 bash scripts/render_mvae_recon_batch.sh exports/batch_mvae_recon_10000_mp4
```

The output is meant for side-by-side human inspection:

1. if `recon` visually follows `gt`, MVAE reconstruction is working
2. if `recon` freezes, jitters badly, or loses the main action, MVAE is not good enough for DAR

### 6.14 `render_gmr_view_mp4.py`

This renders `.npz` files with the same joint-order semantics used by the local `GMR_view` reference repo.

Use this for paper-facing or evaluator-facing visualization.

Important behavior:

1. GMR/MT files use `dof_pos`, `root_pos`, `root_rot`, and `joint_names` or `body_names`
2. GMR/MT `root_rot` is treated as `xyzw`
3. GMR/MT `dof_pos` is treated as grouped MT order
4. wrist joints are removed before rendering through the GMR 23DoF MuJoCo XML
5. Isaac-style files use `joint_pos` in Isaac left/right-interleaved order
6. video FPS defaults to the motion file's own `fps`

This fixes the old quick renderer issue where a `50 FPS` motion was rendered at `30 FPS`, making clips look slower and harder to compare with tracker/evaluation output.

Example:

```bash
MUJOCO_GL=egl /data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/sim2sim/.venv/bin/python \
  scripts/render_gmr_view_mp4.py \
  exports/batch_mvae_recon_100000_mp4/gt \
  --out-dir exports/batch_mvae_recon_100000_mp4/mp4_gt_gmr
```

### 6.15 `render_mvae_recon_gmr_batch.sh`

This renders both sides of an MVAE reconstruction batch through the GMR-compatible path:

1. `gt/*.npz` to `mp4_gt_gmr/*.mp4`
2. `recon/*.npz` to `mp4_recon_gmr/*.mp4`

Example:

```bash
LOOPS=3 bash scripts/render_mvae_recon_gmr_batch.sh exports/batch_mvae_recon_100000_mp4
```

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
