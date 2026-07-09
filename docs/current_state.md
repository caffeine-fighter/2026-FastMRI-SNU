# Current State — Phase 2 Handoff

_Last updated: 2026-07-09 KST from desktop WSL workspace `Cafe-Pastelica`._

## Machine roles

- **VESSL**: official `EXP###` training and official `recon_eval` / leaderboard evaluation.
- **Desktop WSL**: local probes, automation, documentation, and Phase 2 wrapper preparation.
- **Laptop**: control plane / VS Code SSH client / assistant interface.

## VESSL running job

`EXP030_varnet_c4_ch12_s8_e20` is currently training on VESSL.

While this job is active, do not run VESSL commands that can alter the workspace, change git state, consume the training GPU, or start evaluation.

## Current best completed validation candidate

Current completed VESSL validation reference:

| Experiment | Config | val_loss | SSIM_full | SSIM_bbox | quality_score |
|---|---|---:|---:|---:|---:|
| `EXP012_varnet_c4_ch12_s4_e10` | cascade=4, chans=12, sens_chans=4, epochs=10 | 3.2876096990602717 | 0.8994141339351495 | 0.9187541341189271 | 0.9090841340270383 |

Quality formula:

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
```

`EXP012` is the current completed validation candidate until `EXP030` finishes and is evaluated.

## Desktop LOCAL probe summary

All `LOCAL_` results are exploratory desktop probes only. They are not official VESSL scores, must not be submitted, and must not be treated as final candidates.

| Probe | Config | Epochs | SSIM_full | SSIM_bbox | quality |
|---|---|---:|---:|---:|---:|
| `LOCAL_EXP012_varnet_c4_ch12_s4_e1` | c4/ch12/s4 | 1 | 0.8794828387 | 0.8889114335 | 0.8841971361 |
| `LOCAL_EXP013_varnet_c4_ch12_s8_e1` | c4/ch12/s8 | 1 | 0.8807929440 | 0.8890808084 | 0.8849368762 |
| `LOCAL_EXP014_varnet_c6_ch12_s8_e1` | c6/ch12/s8 | 1 | 0.8811104767 | 0.8903269135 | 0.8857186951 |
| `LOCAL_EXP015_varnet_c6_ch12_s4_e1` | c6/ch12/s4 | 1 | 0.8747640465 | 0.8805270891 | 0.8776455678 |
| `LOCAL_EXP016_varnet_c3_ch12_s8_e1` | c3/ch12/s8 | 1 | 0.8775208604 | 0.8807431730 | 0.8791320167 |

Local-probe interpretation:

- The best 1-epoch desktop probe by local quality is `LOCAL_EXP014` (`c6/ch12/s8`), but this is not a final candidate.
- `LOCAL_EXP013` is close behind and tests the `ch12/s8` direction with lower cascade cost.
- `LOCAL_EXP015` suggests `c6/ch12/s4` underperformed `c6/ch12/s8` locally.
- Promising `LOCAL_` directions must be reproduced on VESSL as `EXP###` before any final decision.

## Next steps after `EXP030` finishes

1. Confirm the `EXP030` training process has fully exited on VESSL.
2. Check `EXP030` artifacts and `best_model.pt` path.
3. Run validation evaluation with `scripts/evaluate_val.py`.
4. Plot validation loss with `scripts/plot_loss.py`.
5. Compare `EXP030` against `EXP012` using:
   - `val_loss`
   - `SSIM_full`
   - `SSIM_bbox`
   - `quality_score = 0.5 * full + 0.5 * bbox`
   - Phase 2 runtime expectations
6. Append an `EXP030` row to `experiments/experiment_log.csv` with exact metrics.
7. Commit and push the EXP030 metrics/log/docs update only after reviewing forbidden files.
8. Only after EXP030 metrics are recorded, merge/update `phase2/eval-wrapper` as needed.
9. Set Phase 2 candidates for `EXP012` and `EXP030` with `scripts/set_phase2_candidate.sh`.
10. Run official Phase 2 wrapper evaluation only after the user approves and preflight passes.

See `docs/vessl_after_exp030_runbook.md` for command templates.

## Do-not-do list while `EXP030` is running

Do not run these on VESSL while `EXP030` is training:

- `git pull`
- `git checkout`
- `git reset`
- `recon_eval.sh`
- any training command
- any Phase 2 leaderboard/evaluation command
- any command that modifies mounted `Data` directories

Do not modify without explicit approval:

- `recon_eval.py`
- `train.py`
- `reconstruct.py`
- model code
- loss code
- transforms/data pipeline code
- official metric functions

Never stage or commit:

- `Data/`, `data/`
- `*.h5`
- `result/`, `results/`, `runs/`, `checkpoints/`, `checkpoints_phase2/`
- `*.pt`, `*.pth`, `*.ckpt`
- `.env`, `.env.local`
- secrets or credentials
