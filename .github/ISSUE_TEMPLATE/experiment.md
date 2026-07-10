---
name: Experiment
description: Track a VESSL EXP### or desktop LOCAL_ experiment
title: "[Experiment] <EXP_ID> <short description>"
labels: ["type:experiment", "status:needs-review"]
assignees: []
---

## Experiment ID

- `exp_id`:

## Config

- Architecture / variant:
- `cascade`:
- `chans`:
- `sens_chans`:
- `epochs`:
- `seed`:
- Machine: VESSL / Desktop WSL

## Command

```bash
# paste exact command here
```

## Expected result

- Hypothesis:
- Expected quality/risk:
- Expected runtime impact:

## Metrics

- `val_loss`:
- `SSIM_full`:
- `SSIM_bbox`:
- `quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox`:
- `acc4 quality`:
- `acc8 quality`:
- `skipped.json` status:
- checkpoint path:

## Risks

- OOM risk:
- Timing risk:
- Rule/compliance risk:

## Acceptance criteria

- [ ] Training/evaluation command recorded exactly.
- [ ] Metrics recorded in `experiments/experiment_log.csv` when applicable.
- [ ] `skipped.json` reviewed.
- [ ] No forbidden files staged or committed.
- [ ] Decision/rationale documented.

## Safety constraints

- Do not commit `Data/`, `data/`, `*.h5`, checkpoints, `*.pt`, `*.pth`, `*.ckpt`, result folders, `.env`, or secrets.
- Do not modify `recon_eval.py`.
- Do not run training or official eval without explicit approval.
