# Current State — Phase 2 Handoff

_Last updated: 2026-07-10 KST from VESSL workspace after the official EXP030 30-repeat evaluation._

## Machine roles

- **VESSL**: official `EXP###` training and official `recon_eval` / leaderboard evaluation.
- **Desktop WSL**: local probes, automation, documentation, and Phase 2 wrapper preparation.
- **Laptop**: control plane / VS Code SSH client / assistant interface.

## VESSL training status

`EXP030_varnet_c4_ch12_s8_e20` is complete. No training process should be running before Phase 2 evaluation.

## Current best validation candidate

`EXP030_varnet_c4_ch12_s8_e20` is the current best validation candidate.

| Experiment | Config | val_loss | SSIM_full | SSIM_bbox | quality_score |
|---|---|---:|---:|---:|---:|
| `EXP012_varnet_c4_ch12_s4_e10` | cascade=4, chans=12, sens_chans=4, epochs=10 | 3.2876096990602717 | 0.8994141339351495 | 0.9187541341189271 | 0.9090841340270383 |
| `EXP030_varnet_c4_ch12_s8_e20` | cascade=4, chans=12, sens_chans=8, epochs=20 | 3.202955630212294 | 0.90337035478141 | 0.9259878171156652 | 0.9146790859485376 |

EXP030 beats EXP012 by `+0.0055949519214994` validation quality.

Quality formula:

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
```

EXP030 validation details:

- best_epoch: 19
- acc4 SSIM_full: 0.918414056886912
- acc4 SSIM_bbox: 0.9352672735107279
- acc8 SSIM_full: 0.8874255976018807
- acc8 SSIM_bbox: 0.9076007461106336
- skipped validation files: []

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
- EXP030 is the official VESSL validation leader and should be evaluated before considering new VESSL training.

See `reports/local_comparisons/local_probe_summary.md` for the generated local-probe summary.

## Official Phase 2 result

The approved official 30-repeat evaluation completed successfully for
`EXP030_official`. All 30 runs reported the same quality metrics, and run 01
provided both the minimum valid timing and maximum total score.

| Metric | Final value |
|---|---:|
| completed runs | 30 |
| best run | `EXP030_official_run01` |
| SSIM_full | 0.9178 |
| SSIM_bbox | 0.9108 |
| quality_score | 0.9143 |
| minimum time_ms_per_slice | 173.4 |
| time_score | 0.0009513541666666666 |
| total_score | 0.9152513541666667 |

EXP030 remains the final candidate. Its official one-shot total-score advantage
over EXP012 was `0.0062859895833333`, despite EXP030's slower reconstruction.

## Next action

Finalize the GitHub repository and submit the required items separately:

1. GitHub repository with the `bash recon_eval.sh` reproduction instructions.
2. `reports/figures/EXP030_varnet_c4_ch12_s8_e20_val_loss.png`.
3. `/root/result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt` as a separate model-weight artifact; never commit it to Git.
4. `reports/phase2/EXP030_model_description.pptx`.

See `docs/final_submission_checklist.md` and
`reports/phase2/final_score_summary.md` for the final handoff.

## Safety rules

Do not run these unless explicitly approved for the current step:

- `recon_eval.sh`
- `scripts/run_recon_eval_once.sh`
- `scripts/repeat_recon_eval.sh`
- any training command
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
