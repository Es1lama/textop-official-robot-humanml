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
