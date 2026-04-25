# Current Results

This file records the latest verified training status after the original public handoff commit.

## 1. MVAE Status

MVAE was first verified on full data, then resumed and stopped later to free GPU for DAR.

Key numbers:

1. `loss/train_total`: `0.459224 -> 0.026353` by step `1390`
2. resumed late-stage checkpoint: `ckpt_10000.pth`
3. resumed tail value near stop: `0.006279` at step `13896`
4. recent tail mean: `0.006592`

Interpretation:

1. this is not a strict claim of final mathematical convergence
2. but the run is clearly in a much flatter late regime
3. `ckpt_10000.pth` is a reasonable handoff checkpoint

## 2. DAR Status

DAR was started from MVAE `ckpt_10000.pth` and trained successfully to `step 50000`.

Key numbers:

1. `loss/train_total`: `0.498519 -> 0.100912`
2. `loss/train_rec`: `0.109399 -> 0.021164`
3. `loss/train_latent_rec`: `0.389046 -> 0.079731`
4. recent `loss/train_total` mean over last 20 points: `0.105537`
5. recent `loss/train_total` mean over last 100 points: `0.107501`

Interpretation:

1. DAR is not fully finished relative to the whole official schedule
2. but it is clearly training in the correct direction
3. `ckpt_50000.pth` is already usable as a mid-stage handoff checkpoint

## 3. Recommended Handoff Files

For a teammate who wants to continue from the current state:

1. MVAE checkpoint: `ckpt_10000.pth`
2. DAR checkpoint: `ckpt_50000.pth`
3. DAR config: saved Hydra `cfg.yaml`
4. training logs: `run.log`

## 4. What Reconstruction Means

Reconstruction is an MVAE-only check.

It means:

1. take a real motion from the validation set
2. encode it into latent
3. decode it back
4. compare reconstructed motion against the original motion

It answers:

1. did MVAE learn a useful motion representation?
2. can it preserve the main motion content when compressing and decoding?

It does **not** answer:

1. whether text-driven generation is already good
2. whether DAR has learned prompt-conditioned control well

## 5. Exported Reconstruction Sample

One reconstruction sample was exported locally during verification.

Measured values:

1. feature-space mean MSE: `0.004995`
2. feature-space mean MAE: `0.039733`
3. qpos-space MSE: `3.14e-05`
4. qpos-space MAE: `0.00351`
5. qpos-space max absolute error: `0.02525`

These numbers are enough to say the MVAE reconstruction is already reasonable.

## 6. MVAE Reconstruction MP4 Batch

A 4-sample MVAE reconstruction batch was exported on 2026-04-25.

Output directory:

```text
exports/batch_mvae_recon_10000_mp4
```

Files:

1. manifest: `exports/batch_mvae_recon_10000_mp4/manifest.jsonl`
2. ground-truth MP4s: `exports/batch_mvae_recon_10000_mp4/mp4_gt/*.mp4`
3. VAE reconstruction MP4s: `exports/batch_mvae_recon_10000_mp4/mp4_recon/*.mp4`
4. batch summary: `exports/batch_mvae_recon_10000_mp4/BATCH_SUMMARY.md`

Checkpoint:

```text
/data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/_reference/TextOp_official/TextOpRobotMDAR/logs/RobotMDAR/OfficialFullTrainabilityResume/train-mvae-20260418_160348/ckpt_10000.pth
```

What it means:

1. `mp4_gt` shows the original validation-set motion
2. `mp4_recon` shows the motion after MVAE encode/decode
3. this checks MVAE reconstruction quality
4. this is not text generation, not DAR generation, and not sim2sim tracking

Validation:

1. 4 ground-truth MP4s were generated
2. 4 VAE reconstruction MP4s were generated
3. each MP4 is `6.6s`, `30 FPS`, `198` frames

Batch text prompts:

1. `mvae_0000`: `a person walks in a curved angle then stops. | a person walks in a curved line. | a man walks and turns to the right.`
2. `mvae_0001`: `a person walks forward, swaying their hips. | a person spins to their right, appears to look around, walks quickly up a hill, then turns back around to their left. | the person is doing a casual quick walk.`
3. `mvae_0002`: `he stomps his right feet | a person stomps with their right leg. | a person stomps their right foot`
4. `mvae_0003`: `a person  slowly walked forward | a person walks forward at medium speed. | walking forward and then stopping.`

## 7. DAR Export And sim2sim Smoke Check

DAR checkpoint `ckpt_50000.pth` was exported to tracker-readable motion files on
2026-04-25.

Export command:

```bash
MAX_MOTIONS=2 BATCH_SIZE=2 PYTHON_BIN=/data/haozhe/miniconda3/envs/motiongpt/bin/python \
  bash scripts/export_existing_dar_npz.sh \
  4 \
  /data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/_reference/TextOp_official/TextOpRobotMDAR/logs/RobotMDAR/OfficialFullDARFromMVAE10000/train-dar-20260418_181357/ckpt_50000.pth \
  /data/haozhe/zzn/VAR_FM/ws/project/textop-official-robot-humanml/exports/smoke_dar_npz
```

Exported files:

1. sim2sim motions: `exports/smoke_dar_npz/sim2sim/dar_0000.npz`, `exports/smoke_dar_npz/sim2sim/dar_0001.npz`
2. ground-truth comparison motions: `exports/smoke_dar_npz/sim2sim_gt/*`
3. best-effort TextOpTracker-style files: `exports/smoke_dar_npz/tracker/*/motion.npz`
4. videos: `exports/smoke_dar_npz/dar_0000_sim2sim.mp4`, `exports/smoke_dar_npz/dar_0001_sim2sim.mp4`

Motion shape:

1. `fps`: `50`
2. generated clip length: `66` frames
3. generated clip duration: `1.32s`
4. `dof_pos`: `66 x 29`
5. `root_pos`: `66 x 3`
6. `root_rot`: `66 x 4`

sim2sim tracker smoke result:

1. tracker loaded the exported `.npz`
2. joint/action mapping was `29/29`
3. ONNX policy inference was about `0.5 ms`
4. high-level controller loop ran around `50 Hz`
5. MuJoCo video recording succeeded

Important limitation:

1. this proves the export and tracker runtime path are connected
2. this does not prove the generated motion is already paper-quality
3. with the current mid-stage `ckpt_50000.pth`, the robot can fall after several seconds in sim2sim
4. use this as a smoke test, not as a final tracker-quality result

## 8. Batch MP4 Export

A 4-sample batch was exported on 2026-04-25.

Output directory:

```text
exports/batch_dar50000_mp4
```

Files:

1. manifest: `exports/batch_dar50000_mp4/manifest.jsonl`
2. official/reference MP4s: `exports/batch_dar50000_mp4/official_mp4/*.mp4`
3. sim2sim tracker MP4s: `exports/batch_dar50000_mp4/sim2sim_mp4/annotated/*.mp4`
4. sim2sim logs: `exports/batch_dar50000_mp4/sim2sim_mp4/logs/*.log`

The official/reference MP4s are direct kinematic playback of TextOp/DAR generated target motions.
The sim2sim MP4s are closed-loop tracker runs where the controller tries to follow the exported target.

Batch text prompts:

1. `dar_0000`: `a person walks in a curved angle then stops. | a person walks in a curved line. | a man walks and turns to the right.`
2. `dar_0001`: `a person walks forward, swaying their hips. | a person spins to their right, appears to look around, walks quickly up a hill, then turns back around to their left. | the person is doing a casual quick walk.`
3. `dar_0002`: `he stomps his right feet | a person stomps with their right leg. | a person stomps their right foot`
4. `dar_0003`: `a person  slowly walked forward | a person walks forward at medium speed. | walking forward and then stopping.`

Validation:

1. all 4 official/reference MP4s were generated
2. all 4 sim2sim MP4s were generated
3. all 4 sim2sim deploy logs entered `Running high level...`
4. all 4 sim2sim deploy logs reported `Action mapping: 29/29 mapped`
5. all 4 sim2sim control loops ran near `50 Hz`

Limitation:

1. current mid-stage DAR motions still cause the simulated robot to fall after roughly `8s`
2. therefore this batch proves export/render/tracker connectivity, not final motion quality
3. IsaacLab TextOpTracker was not run because the available conda environments do not currently import `isaaclab`
