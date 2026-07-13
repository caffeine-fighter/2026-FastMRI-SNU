# 2026 FastMRI SNU

VarNet experiments and Phase 2 submission tooling for the 2026 SNU FastMRI Challenge. VESSL is the source of truth for all `EXP###` runs and official evaluation.

<!-- EXP031_STATUS_START -->
## Live VESSL status

_Last update: 2026-07-14 07:54 KST (2026-07-13 22:54 UTC)_

| Check | Value |
|---|---|
| Run | `EXP035_varnet_c8_ch12_s8_e30` (running) |
| Change | `capacity-only comparison versus EXP032: cascades 6 to 8; same chans/sens/LR/epochs/seed; worst-shape training uses 7741/8192 MiB` |
| Progress | epoch `8/30`, iteration `4030/4651`, `29.6%` |
| ETA | `33.39 hours`; finish `2026-07-15 17:17 KST (2026-07-15 08:17 UTC)` |
| Best validation loss | epoch `7`: `3.2314329978710994` |
| Validation snapshot | pending final validation |
| Health | `0` error matches; checkpoints `present` |

`EXP033R_epoch32` remains the one-shot official leader while this one-variable experiment trains. No official evaluation starts automatically.
<!-- EXP031_STATUS_END -->

## Official result

| Candidate | Evidence | SSIM_full | SSIM_bbox | Quality | Time | Total score |
|---|---|---:|---:|---:|---:|---:|
| `EXP030_varnet_c4_ch12_s8_e20` | 30-run minimum | 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | 0.9152513541666667 |
| `EXP031_varnet_c4_ch12_s8_e30` | one official run | 0.9191 | 0.9114 | 0.91525 | **173.1 ms/slice** | 0.9162015104166666 |
| `EXP032_varnet_c6_ch12_s8_e30` | one official run | 0.9197 | **0.9120** | 0.91585 | 212.2 ms/slice | 0.9167811458333334 |
| `EXP033R_varnet_c4_ch12_s8_lr3e4_e33` epoch 32 | one official run | **0.9199** | **0.9120** | **0.91595** | 173.6 ms/slice | **0.91690125** |
| `EXP034_varnet_c4_ch12_s8_lr3e4_scorealigned_e33` | one official run | 0.9181 | 0.9106 | 0.91435 | 173.7 ms/slice | 0.9153011979166666 |

EXP033R epoch 32 is the current one-shot official leader. Versus EXP032, it improves quality by `+0.00010`, is `38.6 ms/slice` faster, and improves total score by `+0.0001201041666666347`. EXP031, EXP032, and EXP030 remain protected references; the final 30-run timing cohort is deferred until model freeze.

Local promotion uses leaderboard-faithful equal-acceleration validation quality. EXP031's reference is `0.9154137446412757`; EXP032 scored `0.9149434297189161` locally, so the five-epoch lower-LR continuation uses EXP031 as preregistered. Pooled diagnostics are not promotion thresholds.

## Latest objective result

EXP034 tested the opt-in score-aligned objective as a one-variable comparison against EXP033R. It preserved the same EXP031 source checkpoint, c4/ch12/s8 architecture, LR `0.0003`, seed `430`, sampler order, and five-epoch budget. The run and all five retained epochs completed with zero error matches and strict `30 volumes / 791 slices / 161 boxes / 0 skips` coverage. Epoch 33 ranked first locally at full `0.9032527316484851`, bbox `0.9266163614092091`, and quality `0.9149345465288471`, trailing EXP033R's `0.9156824558941089` by `0.0007479093652618118`. The authorized official one-shot confirmed rejection: total `0.9153011979166666`, which trails EXP032 by `0.001479947916666724`.

## Common commands

```bash
# Check repository safety
python scripts/check_submission.py

# Evaluate one experiment on validation data
python scripts/evaluate_val.py \
  --exp-name <EXP_NAME> \
  --target-dir /root/Data/val/image \
  --recon-dir ../result/<EXP_NAME>/reconstructions_val \
  --out-dir ../result/<EXP_NAME>/metrics

# Check the official wrapper before an approved run
bash scripts/phase2_preflight.sh
```

Official leaderboard entrypoint:

```bash
mkdir -p ../result/test_Varnet/checkpoints
cp /path/to/submitted/best_model.pt ../result/test_Varnet/checkpoints/best_model.pt
bash recon_eval.sh
```

`recon_eval.sh` defaults to the EXP030 architecture: cascade 4, 12 channels, and 8 sensitivity-map channels.

## Repository map

| Path | Purpose |
|---|---|
| `train.py` | VESSL training entrypoint |
| `reconstruct.py` | Reconstruction entrypoint |
| `recon_eval.sh` | Official Phase 2 entrypoint |
| `scripts/` | Validation, preflight, scoring, and reporting helpers |
| `experiments/experiment_log.csv` | Experiment registry |
| `reports/phase2/` | Official score and submission reports |
| `docs/` | Workflow, status, rules, and decision history |

## Non-negotiable rules

- Do not modify `recon_eval.py`.
- Do not run official evaluation while training is active or without approval.
- Treat mounted `Data` directories as read-only.
- Never commit data, H5 files, checkpoints, result directories, `.env` files, or credentials.
- Keep `LOCAL_` probes exploratory; only VESSL `EXP###` runs can become official candidates.

## Documentation

Start with [`docs/README.md`](docs/README.md). The most useful pages are:

- [`docs/current_state.md`](docs/current_state.md): current candidate, active work, and next actions
- [`docs/score_optimization_40_day_roadmap.md`](docs/score_optimization_40_day_roadmap.md): experiment portfolio through final freeze
- [`docs/vessl_workflow.md`](docs/vessl_workflow.md): training and validation commands
- [`docs/phase2_plan.md`](docs/phase2_plan.md): scoring and official-evaluation rules
- [`docs/final_submission_checklist.md`](docs/final_submission_checklist.md): external upload checklist
- [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md): verified EXP030 official result
- [`reports/phase2/EXP031_validation_summary.md`](reports/phase2/EXP031_validation_summary.md): final EXP031 validation result
