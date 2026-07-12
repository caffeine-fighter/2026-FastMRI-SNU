# 2026 FastMRI SNU

VarNet experiments and Phase 2 submission tooling for the 2026 SNU FastMRI Challenge. VESSL is the source of truth for all `EXP###` runs and official evaluation.

<!-- EXP031_STATUS_START -->
## Live VESSL status

_Last update: 2026-07-13 06:11 KST (2026-07-12 21:11 UTC)_

| Check | Value |
|---|---|
| Run | `EXP033R_varnet_c4_ch12_s8_lr3e4_e33` (running) |
| Change | `relaunch after retained-path fix; five-epoch continuation from EXP031 best; LR 0.0003` |
| Progress | epoch `29/33`, iteration `1770/4651`, `89.0%` |
| ETA | `3.91 hours`; finish `2026-07-13 10:06 KST (2026-07-13 01:06 UTC)` |
| Best validation loss | epoch `28`: `3.1712609686302593` |
| Validation snapshot | pending final validation |
| Health | `0` error matches; checkpoints `present` |

`EXP032` remains the one-shot official leader while this one-variable experiment trains. No official evaluation starts automatically.
<!-- EXP031_STATUS_END -->

## Official result

| Candidate | Evidence | SSIM_full | SSIM_bbox | Quality | Time | Total score |
|---|---|---:|---:|---:|---:|---:|
| `EXP030_varnet_c4_ch12_s8_e20` | 30-run minimum | 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | 0.9152513541666667 |
| `EXP031_varnet_c4_ch12_s8_e30` | one official run | 0.9191 | 0.9114 | 0.91525 | **173.1 ms/slice** | 0.9162015104166666 |
| `EXP032_varnet_c6_ch12_s8_e30` | one official run | **0.9197** | **0.9120** | **0.91585** | 212.2 ms/slice | **0.9167811458333334** |

EXP032 is the current one-shot official leader: quality improved by `+0.00060` and total score by `+0.0005796354166667239` versus EXP031, despite a `+39.1 ms/slice` timing cost. EXP031 and EXP030 remain protected references; the final 30-run timing cohort is deferred until model freeze.

Local promotion uses leaderboard-faithful equal-acceleration validation quality. EXP031's reference is `0.9154137446412757`; EXP032 scored `0.9149434297189161` locally, so the five-epoch lower-LR continuation uses EXP031 as preregistered. Pooled diagnostics are not promotion thresholds.

## Next objective gate

The opt-in EXP034 score-aligned training path is published in `817c9eb`. It preserves default training and inference, passed independent adversarial review, 53 focused tests, 218 discovered tests, and a five-case production VarNet CPU gradient gate. All losses and 4,848,494-element model gradients were finite and nonzero; the largest score/legacy gradient ratios were `35.68×` L2 and `47.09×` max-absolute, with absolute score-gradient maxima of `0.7404` L2 and `0.4270`. EXP034 remains gated on EXP033 completion and strict retained-epoch sweep; it will change only the training objective.

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
