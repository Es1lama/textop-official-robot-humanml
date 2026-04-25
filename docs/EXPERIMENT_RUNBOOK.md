# Experiment Runbook

This runbook defines how to run, monitor, stop, and hand off the official TextOp baseline.

Assumption:

1. "Phase 1" means MVAE training.
2. "Phase 2" means DAR training from a selected MVAE checkpoint.
3. If "Phase 2" instead means the later "our latent, no official VAE" comparison, do not start it until this official baseline has passed MVAE reconstruction, DAR visualization, and tracker smoke checks.

## 1. Preflight

Before starting a training run:

1. Close unrelated tmux sessions owned by the current user.
2. Confirm data layout:
   `bash scripts/check_dataset_layout.sh`
3. Pick a GPU with enough free memory and stable utilization.
4. Record the selected GPU, command, run directory, and checkpoint source.
5. Start a monitor:
   `bash scripts/start_textop_monitor_tmux.sh /path/to/run-dir`

## 2. Phase 1: MVAE

Purpose:

1. Learn a motion representation.
2. Prove that real robot motion can be compressed and reconstructed.
3. Produce a VAE checkpoint for DAR.

Command:

```bash
bash scripts/run_official_textop_mvae.sh <GPU>
```

Resume command:

```bash
bash scripts/run_official_textop_mvae_resume.sh <GPU> /path/to/mvae_ckpt.pth
```

## 3. MVAE Stop Criteria

Minimum stop criteria:

1. `loss/train_total` has dropped at least `90%` from early training.
2. A checkpoint has been saved after the drop.
3. No `NaN`, `CUDA out of memory`, traceback, or repeated restart appears in `run.log`.
4. A reconstruction sample passes the numerical check below.

Preferred stop criteria:

1. `loss/train_total <= 0.01`, or the last 100 logged values improve by less than about `10%` compared with the previous window.
2. Reconstruction feature-space mean MSE is roughly `<= 0.01`.
3. Reconstruction qpos-space MAE is roughly `<= 0.01`.

Current verified MVAE state:

1. `loss/train_total`: `0.459224 -> 0.006279`
2. recommended checkpoint: `ckpt_10000.pth`
3. reconstruction feature-space mean MSE: `0.004995`
4. reconstruction qpos-space MAE: `0.00351`

This satisfies the practical stop criteria for moving to DAR.

## 4. Switch From MVAE To DAR

Switch to DAR when:

1. MVAE has a saved checkpoint that satisfies the stop criteria.
2. MVAE reconstruction has been checked.
3. The MVAE checkpoint path is recorded.
4. Enough GPU memory is available for DAR.

Do not switch if:

1. MVAE reconstruction is clearly broken.
2. The chosen checkpoint cannot be loaded by `vis_mvae`.
3. Training logs contain unresolved runtime errors.

## 5. Phase 2: DAR

Purpose:

1. Learn text/history-conditioned latent prediction.
2. Produce a checkpoint that can generate motions for visualization and tracker testing.

Command:

```bash
bash scripts/run_official_textop_dar.sh <GPU> /path/to/mvae_ckpt.pth
```

Resume command:

```bash
DAR_CKPT=/path/to/dar_ckpt.pth bash scripts/run_official_textop_dar.sh <GPU> /path/to/mvae_ckpt.pth
```

## 6. DAR Stop Criteria

Minimum checkpoint for qualitative testing:

1. `step >= 50000`
2. `loss/train_total` has dropped at least `70%` from early training.
3. `loss/train_rec` and `loss/train_latent_rec` are also lower than early values.
4. A checkpoint has been saved.

Preferred checkpoint for paper-facing comparison:

1. test checkpoints at `50000`, `100000`, `150000`, and later if compute allows
2. choose by visualization and tracker behavior, not only by train loss
3. keep the config and run log with the selected checkpoint

Current verified DAR state:

1. checkpoint: `ckpt_50000.pth`
2. `loss/train_total`: `0.498519 -> 0.100912`
3. `loss/train_rec`: `0.109399 -> 0.021164`
4. `loss/train_latent_rec`: `0.389046 -> 0.079731`

This is a valid mid-stage checkpoint for visualization and continuation.

## 7. Visualization Before Tracker

Before running tracker:

1. Run MVAE reconstruction visualization:
   `bash scripts/run_official_textop_vis_mvae.sh <GPU> /path/to/mvae_ckpt.pth`
2. Run DAR generation visualization:
   `bash scripts/run_official_textop_vis_dar.sh <GPU> /path/to/dar_ckpt.pth /path/to/mvae_ckpt.pth`
3. Check that generated motion is not static, exploding, or obviously broken.
4. Check a few prompt types:
   `stand`, `walk`, `turn`, `raise/wave arm`.

Visualization pass criteria:

1. Body remains physically plausible.
2. No immediate collapse or severe joint jitter.
3. Motion changes over time.
4. Prompt category is roughly recognizable.

## 8. Tracker Check

Only run tracker after visualization passes.

Tracker pass criteria:

1. The tracker loads the generated target without shape/key errors.
2. The simulated robot enters the control loop.
3. The robot stays upright for at least `20s`.
4. Control frequency is stable enough for the existing sim2sim setup.
5. Save logs and, if possible, a short screen recording.

If tracker fails:

1. First inspect generated motion shape, joint order, and root representation.
2. Then inspect tracker input conversion.
3. Only tune tracker settings after the motion file is known to be valid.

## 9. Monitoring

Run a one-shot status check:

```bash
bash scripts/monitor_textop_run.sh /path/to/run-dir
```

Run periodic monitoring in tmux:

```bash
bash scripts/start_textop_monitor_tmux.sh /path/to/run-dir
```

The monitor checks:

1. latest TensorBoard scalar values
2. latest checkpoint
3. recent log errors
4. GPU memory/utilization summary

Manual intervention is required if:

1. loss becomes `NaN`
2. loss increases for a long window and does not recover
3. no checkpoint is written after the expected interval
4. GPU process disappears
5. `run.log` contains traceback, OOM, or killed process messages
