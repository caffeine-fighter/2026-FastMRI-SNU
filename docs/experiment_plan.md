# Experiment plan

## Phase 0: VESSL recovery

- Wait for the workspace quota issue to be resolved.
- Do not create duplicate workspaces.
- Do not delete volumes.
- Keep the local branch ready for immediate pull.

## Phase 1: execution sanity

### EXP000

Smoke training, 1 epoch.

Goal:

- verify /root/Data mount
- verify train.py runs
- verify result/checkpoints
- verify reconstructions_val
- verify val_loss_log.npy

### EXP001

Official-like baseline, 5 epochs.

Goal:

- establish reference validation loss
- run evaluate_val.py
- produce loss plot

## Phase 2: capacity sweep

- EXP010: cascade 2, chans 9, sens_chans 4
- EXP011: cascade 4, chans 9, sens_chans 4
- EXP012: cascade 4, chans 12, sens_chans 4
- EXP013: cascade 4, chans 12, sens_chans 8

## Phase 3: bbox-aware improvements

- foreground-weighted image loss
- bbox-weighted loss
- SSIM + L1 or Charbonnier hybrid
- acc4/acc8 separate analysis

## Metrics to record

- val_loss
- SSIM_full
- SSIM_bbox
- acc4 SSIM_full / SSIM_bbox
- acc8 SSIM_full / SSIM_bbox
- seed
- branch
- commit
- command
