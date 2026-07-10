# 2026-FastMRI-SNU

This repository tracks work for the **2026 SNU FastMRI Challenge**, focused on knee MRI reconstruction. VESSL is the official environment for `EXP###` training, validation, and official leaderboard evaluation, while the desktop WSL machine with RTX 4070 Ti SUPER is used only for local probes, automation, documentation, and command preparation.

## Current status dashboard

| Item | Status |
|---|---|
| VESSL training | Complete; no active training job |
| Final candidate | `EXP030_varnet_c4_ch12_s8_e20` |
| Official SSIM_full | `0.9178` |
| Official SSIM_bbox | `0.9108` |
| Official quality_score | `0.9143` |
| Selected 30-repeat timing | `173.4 ms/slice` (minimum of the required 30-run cohort) |
| Final total_score | `0.9152513541666667` |
| Phase 2 status | Official 30-repeat evaluation complete; final package ready |

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
| `EXP030` | `EXP030_varnet_c4_ch12_s8_e20` | 20 | 0.90337035478141 | 0.9259878171156652 | 0.9146790859485376 | 3.202955630212294 | final candidate |

## Local probe summary

Desktop 1-epoch `LOCAL_` probes completed so far:

| Rank | Probe | Config | Local quality |
|---:|---|---|---:|
| 1 | `LOCAL_EXP014_varnet_c6_ch12_s8_e1` | c6/ch12/s8/e1 | 0.8857186951 |
| 2 | `LOCAL_EXP013_varnet_c4_ch12_s8_e1` | c4/ch12/s8/e1 | 0.8849368762 |
| 3 | `LOCAL_EXP012_varnet_c4_ch12_s4_e1` | c4/ch12/s4/e1 | 0.8841971361 |
| 4 | `LOCAL_EXP016_varnet_c3_ch12_s8_e1` | c3/ch12/s8/e1 | 0.8791320167 |
| 5 | `LOCAL_EXP015_varnet_c6_ch12_s4_e1` | c6/ch12/s4/e1 | 0.8776455678 |

`LOCAL_` probes are exploratory only and are not official VESSL results. See:

- [`reports/local_comparisons/local_probe_summary.md`](reports/local_comparisons/local_probe_summary.md)
- [`reports/local_comparisons/local_probe_summary.txt`](reports/local_comparisons/local_probe_summary.txt)

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
- [`docs/final_submission_checklist.md`](docs/final_submission_checklist.md)
- [`docs/phase2_plan.md`](docs/phase2_plan.md)
- [`docs/vessl_after_exp030_runbook.md`](docs/vessl_after_exp030_runbook.md)
- [`experiments/experiment_log.csv`](experiments/experiment_log.csv)
- [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md)
- [`reports/phase2/EXP030_model_description.pptx`](reports/phase2/EXP030_model_description.pptx)
- [`reports/local_comparisons/local_probe_summary.md`](reports/local_comparisons/local_probe_summary.md)
- [`reports/local_comparisons/local_probe_summary.txt`](reports/local_comparisons/local_probe_summary.txt)
