# Current State — Phase 2 Handoff

_Last updated: 2026-07-09 KST from desktop WSL workspace `Cafe-Pastelica`._

## Machine roles

- **VESSL**: official `EXP###` training and official `recon_eval` / leaderboard evaluation.
- **Desktop WSL**: local probes, automation, documentation, and Phase 2 wrapper preparation.
- **Laptop**: control plane / VS Code SSH client / assistant interface.

## VESSL running job

`EXP030_varnet_c4_ch12_s8_e20` is currently **running on VESSL**.

While this job is active, do not run VESSL commands that can alter the workspace, change git state, consume the training GPU, or start evaluation.

## Current best completed validation candidate

`EXP012_varnet_c4_ch12_s4_e10` is the current best completed validation candidate until `EXP030` finishes and is evaluated.

| Experiment | Config | val_loss | SSIM_full | SSIM_bbox | quality_score |
|---|---|---:|---:|---:|---:|
| `EXP012_varnet_c4_ch12_s4_e10` | cascade=4, chans=12, sens_chans=4, epochs=10 | 3.2876096990602717 | 0.8994141339351495 | 0.9187541341189271 | 0.9090841340270383 |

Quality formula:

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
```

## Desktop LOCAL probes completed

All `LOCAL_` results are exploratory desktop probes only. They are not official VESSL scores, must not be submitted, and must not be treated as final candidates.

Completed local probes:

- `LOCAL_EXP012_varnet_c4_ch12_s4_e1`
- `LOCAL_EXP013_varnet_c4_ch12_s8_e1`
- `LOCAL_EXP014_varnet_c6_ch12_s8_e1`
- `LOCAL_EXP015_varnet_c6_ch12_s4_e1`
- `LOCAL_EXP016_varnet_c3_ch12_s8_e1`

### Local probe ranking by overall quality

| Rank | Probe | Config | SSIM_full | SSIM_bbox | quality_score | acc4 quality | acc8 quality | skipped |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `LOCAL_EXP014_varnet_c6_ch12_s8_e1` | c6/ch12/s8/e1 | 0.8811104767 | 0.8903269135 | 0.8857186951 | 0.9017071749 | 0.8631502224 | `[]` |
| 2 | `LOCAL_EXP013_varnet_c4_ch12_s8_e1` | c4/ch12/s8/e1 | 0.8807929440 | 0.8890808084 | 0.8849368762 | 0.9012535192 | 0.8616591192 | `[]` |
| 3 | `LOCAL_EXP012_varnet_c4_ch12_s4_e1` | c4/ch12/s4/e1 | 0.8794828387 | 0.8889114335 | 0.8841971361 | 0.8985789049 | 0.8644184298 | `[]` |
| 4 | `LOCAL_EXP016_varnet_c3_ch12_s8_e1` | c3/ch12/s8/e1 | 0.8775208604 | 0.8807431730 | 0.8791320167 | 0.8943828902 | 0.8578022018 | `[]` |
| 5 | `LOCAL_EXP015_varnet_c6_ch12_s4_e1` | c6/ch12/s4/e1 | 0.8747640465 | 0.8805270891 | 0.8776455678 | 0.8930437597 | 0.8563069000 | `[]` |

Interpretation:

- `LOCAL_EXP014` (`c6/ch12/s8`) is the local 1-epoch quality leader.
- `LOCAL_EXP014` beats `LOCAL_EXP013` by only about `0.00078` quality, so Phase 2 timing may matter.
- `LOCAL_EXP016` (`c3/ch12/s8`) is quality-lower and unlikely to recover with timing alone.
- `LOCAL_EXP015` (`c6/ch12/s4`) is clearly weaker than `LOCAL_EXP014` (`c6/ch12/s8`).
- Do not prioritize `c6` on VESSL until `EXP030` official/validation results are known.

See `reports/local_comparisons/local_probe_summary.md` for the generated local-probe summary.

## Next action after `EXP030` finishes

After `EXP030` finishes training on VESSL:

1. Confirm no `EXP030` training process is still running.
2. Check `EXP030` artifacts and `best_model.pt` path.
3. Run validation evaluation with `scripts/evaluate_val.py`.
4. Plot validation loss with `scripts/plot_loss.py`.
5. Print `metrics.csv` and `skipped.json`.
6. Compare `EXP030` against the completed `EXP012` reference by `val_loss`, `SSIM_full`, `SSIM_bbox`, and `quality_score`.
7. Record `EXP030` in `experiments/experiment_log.csv` with exact metrics.
8. Commit and push the EXP030 validation metrics/docs update after reviewing forbidden files.
9. Only then merge/update `phase2/eval-wrapper` as needed.
10. Only after candidate selection, preflight, user approval, and available mounted leaderboard data, run official Phase 2 `bash recon_eval.sh` through the wrapper scripts.

See `docs/vessl_after_exp030_runbook.md` for copy-paste command templates.

## Do-not-do list while VESSL training is running

Do not run these on VESSL while `EXP030` is training:

- `git pull`
- `git checkout`
- `git reset`
- `git switch`
- `git merge`
- `recon_eval.sh`
- `scripts/run_recon_eval_once.sh`
- `scripts/repeat_recon_eval.sh`
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
