# 2026-FastMRI-SNU

This repository tracks work for the **2026 SNU FastMRI Challenge**, focused on knee MRI reconstruction. VESSL is the official environment for `EXP###` training, validation, and official leaderboard evaluation, while the desktop WSL machine with RTX 4070 Ti SUPER is used only for local probes, automation, documentation, and command preparation.

## Current status dashboard

| Item | Status |
|---|---|
| VESSL training | `EXP031` c4/ch12/s8 training is complete on status commit `bc13bcc`; final experiment-log row and file-backed handoff pending |
| Current official candidate | `EXP030_varnet_c4_ch12_s8_e20` remains authoritative until EXP031 validation completes |
| Official SSIM_full | `0.9178` |
| Official SSIM_bbox | `0.9108` |
| Official quality_score | `0.9143` |
| Selected 30-repeat timing | `173.4 ms/slice` (minimum of the required 30-run cohort) |
| Final total_score | `0.9152513541666667` |
| Desktop LOCAL status | 17 one-epoch probes plus the five-run adaptive e5/seed/e10 campaign are complete; 22/22 LOCAL runs succeeded |
| GitHub status | Default and VESSL branches were verified at `bc13bcc`; LOCAL work remains isolated on `local/probe-sweep-20260710-desktop4070ti` |
| Phase 2 status | EXP031 training-complete telemetry is available; LOCAL c4/ch16 promotion gate failed seed robustness; EXP031 handoff and organizer upload pending |

The existing official 30-repeat report used the **20-epoch EXP030 checkpoint**. It is not a 30-epoch result. No EXP031 metric is treated as official until the VESSL validation artifacts and checkpoint identity are verified.

## Scoring

```text
Final Score = 0.5 * SSIM_full + 0.5 * SSIM_bbox + Tiebreaker
```

The tiebreaker is based on reconstruction speed in `ms/slice`:

```text
<= 80 ms/slice    -> +0.001
>= 2000 ms/slice  -> +0
80..2000          -> linear interpolation
```

Small quality gaps can be timing-sensitive: `ΔSSIM < 0.001` may be overturned by the speed tiebreaker. Official Phase 2 values must come from the official entrypoint:

```bash
bash recon_eval.sh
```

## Machine roles

| Machine | Role |
|---|---|
| VESSL | Official `EXP###` training, validation, and official `recon_eval` |
| Desktop WSL / RTX 4070 Ti SUPER | `LOCAL_` probes only; automation, documentation, and command generation |
| Laptop | Control plane / VS Code SSH / Hermes interface |

`LOCAL_` results are exploratory only. They are not official scores and must not be submitted as final candidates.

## Best validation results so far

| exp_id | config | epochs | SSIM_full | SSIM_bbox | quality_score | val_loss | status |
|---|---|---:|---:|---:|---:|---:|---|
| `EXP010` | `EXP010_varnet_c2_ch9_s4_e10` | 10 | 0.8946281794350307 | 0.9089210673889018 | 0.9017746234119662 | 3.3869223552422363 | done |
| `EXP012` | `EXP012_varnet_c4_ch12_s4_e10` | 10 | 0.8994141339351495 | 0.9187541341189271 | 0.9090841340270383 | 3.2876096990602717 | completed comparison candidate |
| `EXP030` | `EXP030_varnet_c4_ch12_s8_e20` | 20 | 0.90337035478141 | 0.9259878171156652 | 0.9146790859485376 | 3.202955630212294 | current official candidate |
| `EXP031` | c4/ch12/s8 continuation | 30 | pending | pending | pending | 3.1818922822 (telemetry) | training-complete telemetry; final file-backed validation pending |

## Local probe summary

Desktop one-epoch `LOCAL_` probes completed so far: **17** (`LOCAL_EXP012` through `LOCAL_EXP028`). All are exploratory and use the desktop validation protocol.

| Rank | Probe | Config | SSIM_full | SSIM_bbox | Local quality |
|---:|---|---|---:|---:|---:|
| 1 | `LOCAL_EXP018_varnet_c4_ch16_s8_e1` | c4/ch16/s8/e1 | 0.8854211528 | 0.9004961319 | 0.8929586423 |
| 2 | `LOCAL_EXP019_varnet_c4_ch12_s12_e1` | c4/ch12/s12/e1 | 0.8844853268 | 0.8948144457 | 0.8896498863 |
| 3 | `LOCAL_EXP028_varnet_c7_ch12_s8_e1` | c7/ch12/s8/e1 | 0.8843235364 | 0.8948626322 | 0.8895930843 |
| 4 | `LOCAL_EXP021_varnet_c6_ch12_s12_e1` | c6/ch12/s12/e1 | 0.8844415618 | 0.8939631755 | 0.8892023687 |
| 5 | `LOCAL_EXP027_varnet_c6_ch14_s8_e1` | c6/ch14/s8/e1 | 0.8841284133 | 0.8931268794 | 0.8886276464 |
| 6 | `LOCAL_EXP025_varnet_c4_ch12_s10_e1` | c4/ch12/s10/e1 | 0.8834250094 | 0.8925935345 | 0.8880092719 |
| 7 | `LOCAL_EXP022_varnet_c4_ch16_s12_e1` | c4/ch16/s12/e1 | 0.8831212078 | 0.8911024973 | 0.8871118525 |
| 8 | `LOCAL_EXP020_varnet_c4_ch18_s8_e1` | c4/ch18/s8/e1 | 0.8817238236 | 0.8924220486 | 0.8870729361 |
| 9 | `LOCAL_EXP014_varnet_c6_ch12_s8_e1` | c6/ch12/s8/e1 | 0.8811104767 | 0.8903269135 | 0.8857186951 |
| 10 | `LOCAL_EXP013_varnet_c4_ch12_s8_e1` | c4/ch12/s8/e1 | 0.8807929440 | 0.8890808084 | 0.8849368762 |

Round 2 (`LOCAL_EXP024`–`LOCAL_EXP028`) completed 5/5 with no failed runs or skipped validation files. `LOCAL_EXP028 c7/ch12/s8/e1` led that round, but `LOCAL_EXP018 c4/ch16/s8/e1` remains the overall one-epoch LOCAL leader.

The adaptive matched-comparison campaign completed 5/5 without failures or skipped validation files:

| Probe | Config | Seed | SSIM_full | SSIM_bbox | Local quality |
|---|---|---:|---:|---:|---:|
| `LOCAL_EXP029` | c4/ch12/s8/e5 | 430 | 0.8957743251 | 0.9116992114 | 0.9037367682 |
| `LOCAL_EXP032` | c4/ch16/s8/e5 | 430 | 0.8986326562 | 0.9185488328 | 0.9085907445 |
| `LOCAL_EXP033` | c4/ch12/s8/e1 | 431 | 0.8823435134 | 0.8932403356 | 0.8877919245 |
| `LOCAL_EXP034` | c4/ch16/s8/e1 | 431 | 0.8808135834 | 0.8923592767 | 0.8865864301 |
| `LOCAL_EXP035` | c4/ch16/s8/e10 | 430 | 0.9021009122 | 0.9229051363 | 0.9125030243 |

The c4/ch16/s8 candidate won the matched seed-430 five-epoch comparison by `+0.0048539763` quality and improved from e5 to e10 by `+0.0039122798`. However, it lost the seed-431 replication by `-0.0012054944`; its full-image component regressed by `-0.0015299300`, beyond the allowed `-0.001` floor. The predefined LOCAL promotion gate therefore **fails**. Do not promote c4/ch16/s8 from this desktop evidence.

`LOCAL_` checkpoints and timing are never official. See:

- [`reports/local_comparisons/local_probe_summary.md`](reports/local_comparisons/local_probe_summary.md)
- [`reports/local_comparisons/local_probe_sweep_20260711_round2.md`](reports/local_comparisons/local_probe_sweep_20260711_round2.md)
- [`reports/local_comparisons/local_probe_adaptive_followup_20260711_final.md`](reports/local_comparisons/local_probe_adaptive_followup_20260711_final.md)
- [`reports/local_comparisons/local_probe_adaptive_followup_20260711.json`](reports/local_comparisons/local_probe_adaptive_followup_20260711.json)
- [`reports/local_comparisons/local_probe_adaptive_followup_plan_20260711.json`](reports/local_comparisons/local_probe_adaptive_followup_plan_20260711.json)
- [`docs/exp031_post_training_handoff.md`](docs/exp031_post_training_handoff.md)

## Phase 2 workflow

Hard rules:

- Do **not** modify `recon_eval.py`.
- Use `utils/learning/test_part.py` for custom model I/O support.
- Official leaderboard submission path: `bash recon_eval.sh`.
- Run official `recon_eval` only when the GPU is idle and the user has approved evaluation.
- Repeat the final candidate 30 times for timing; final timing uses the minimum valid `ms/slice` result.
- Do **not** use image fields, bbox annotations, or given GRAPPA during inference.
- Do **not** modify mounted `Data` directories.

Wrapper helpers:

- `scripts/set_phase2_candidate.sh`
- `scripts/phase2_preflight.sh`
- `scripts/run_recon_eval_once.sh`
- `scripts/repeat_recon_eval.sh`
- `scripts/phase2_score.py`

## Final-candidate reproduction

The submitted model is a four-cascade VarNet with 12 cascade U-Net channels and
eight sensitivity-map U-Net channels. Place the separately submitted model
weight at the path expected by the official entrypoint:

```bash
mkdir -p ../result/test_Varnet/checkpoints
cp /path/to/submitted/best_model.pt ../result/test_Varnet/checkpoints/best_model.pt
bash recon_eval.sh
```

`recon_eval.sh` defaults to the final `EXP030` architecture (`cascade=4`,
`chans=12`, `sens_chans=8`) and invokes the fixed `recon_eval.py` harness. The
mounted leaderboard data is read from `/root/Data/leaderboard` and must remain
read-only.

For an internal VESSL verification using the original experiment path:

```bash
bash scripts/set_phase2_candidate.sh \
  EXP030 \
  /root/result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt \
  4 12 8 \
  'Final candidate selected after official comparison'
bash scripts/phase2_preflight.sh
bash scripts/run_recon_eval_once.sh EXP030_official
```

The official 30-repeat evaluation has already completed. Do not repeat it for
submission; see [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md).

## Safety rules

Never commit:

- `Data/`
- `data/`
- `*.h5`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `result/`
- `results/`
- `checkpoints_phase2/`
- `.env`
- `.env.local`
- secrets or credentials

## Links

- [`docs/current_state.md`](docs/current_state.md)
- [`docs/exp031_post_training_handoff.md`](docs/exp031_post_training_handoff.md)
- [`docs/final_submission_checklist.md`](docs/final_submission_checklist.md)
- [`docs/phase2_plan.md`](docs/phase2_plan.md)
- [`docs/vessl_after_exp030_runbook.md`](docs/vessl_after_exp030_runbook.md)
- [`experiments/experiment_log.csv`](experiments/experiment_log.csv)
- [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md)
- [`reports/phase2/EXP030_model_description.pptx`](reports/phase2/EXP030_model_description.pptx)
- [`reports/local_comparisons/local_probe_summary.md`](reports/local_comparisons/local_probe_summary.md)
- [`reports/local_comparisons/local_probe_summary.txt`](reports/local_comparisons/local_probe_summary.txt)
