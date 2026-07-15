# 2026 FastMRI SNU

VarNet experiments and Phase 2 submission tooling for the 2026 SNU FastMRI Challenge. VESSL is the source of truth for all `EXP###` runs and official evaluation.

<!-- EXP031_STATUS_START -->
## Live VESSL status

_Last update: 2026-07-15 18:22 KST (2026-07-15 09:22 UTC)_

| Check | Value |
|---|---|
| Run | `EXP035_varnet_c8_ch12_s8_e30` (complete; terminal exit code `0`) |
| Change | `final unmodified vanilla capacity comparison: cascades 6 to 8 with chans/sens/LR/epochs/seed fixed` |
| Strict validation | epoch `30/30` won; equal-acc full `0.90666693657748`, bbox `0.9332906818845851`, quality `0.9199788092310326` |
| Coverage | all 30 retained epochs; each `30 volumes / 791 slices / 161 boxes`; skips/unknown/non-finite `0` |
| Official one-shot | full `0.9234`, bbox `0.9177`, quality `0.92055`, `250.7 ms/slice`, total `0.92146109375` |
| Provenance | immutable generation `3e8af14268a64d67a308ebe30484ddf2`; checkpoint SHA-256 `dc6e034f…3097ffb7` |
| Health | official wrapper exit code `0`; error matches `0`; GPU returned idle |

`EXP035_epoch30` is the new one-shot official leader. It improves total score over EXP033R by `+0.00455984375`. The required repeated timing cohort remains separately approval-gated and has not run.
<!-- EXP031_STATUS_END -->

## Official result

| Candidate | Evidence | SSIM_full | SSIM_bbox | Quality | Time | Total score |
|---|---|---:|---:|---:|---:|---:|
| `EXP030_varnet_c4_ch12_s8_e20` | 30-run minimum | 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | 0.9152513541666667 |
| `EXP031_varnet_c4_ch12_s8_e30` | one official run | 0.9191 | 0.9114 | 0.91525 | **173.1 ms/slice** | 0.9162015104166666 |
| `EXP032_varnet_c6_ch12_s8_e30` | one official run | 0.9197 | 0.9120 | 0.91585 | 212.2 ms/slice | 0.9167811458333334 |
| `EXP033R_varnet_c4_ch12_s8_lr3e4_e33` epoch 32 | one official run | 0.9199 | 0.9120 | 0.91595 | 173.6 ms/slice | 0.91690125 |
| `EXP034_varnet_c4_ch12_s8_lr3e4_scorealigned_e33` | one official run | 0.9181 | 0.9106 | 0.91435 | 173.7 ms/slice | 0.9153011979166666 |
| `EXP035_varnet_c8_ch12_s8_e30` epoch 30 | one official run | **0.9234** | **0.9177** | **0.92055** | 250.7 ms/slice | **0.92146109375** |

EXP035 epoch 30 is the current one-shot official leader. Versus EXP033R, it improves full by `+0.0035`, bbox by `+0.0057`, quality by `+0.0046`, and total score by `+0.00455984375`; its `+77.1 ms/slice` latency costs only `0.00004015625` in time score. EXP033R and the 30-run EXP030 fallback remain protected. Final repeated timing is deferred until explicit freeze approval.

Local promotion uses leaderboard-faithful equal-acceleration validation quality. EXP031's reference is `0.9154137446412757`; EXP032 scored `0.9149434297189161` locally, so the five-epoch lower-LR continuation uses EXP031 as preregistered. Pooled diagnostics are not promotion thresholds.

## Latest capacity result

EXP035 closed the current unmodified vanilla capacity track with a clear PASS. The strict retained sweep selected terminal epoch 30 at quality `0.9199788092310326`, `+0.004296353336923686` over the EXP033R LOCAL reference. The authorized one-shot confirmed the gain on the official path at total `0.92146109375`. See the [self-contained validation and official report](reports/phase2/EXP035_epoch30_official_20260715_090400/validation_summary.md). No c9/c10/c12 capacity run or 30-repeat timing cohort starts automatically.

## Prepared next work

The matched LOCAL AdamW V10 probe is closed after an exact quality/component tie and 7.50% slower runtime. When the LOCAL GPU is free, the only prepared vanilla follow-up is a dry-run-gated pair from the same immutable EXP035 epoch-30 state: Adam LR `1e-3` versus `3e-4`, epochs 31–35. No arm has started. In parallel, CPU-only model-family work is adapter-first: MIT Feature/FI commit `91f2df47` is source-pinned, while PromptMR+ remains blocked pending written competition/license confirmation. See [`docs/exp035_matched_continuation_runbook.md`](docs/exp035_matched_continuation_runbook.md) and [`docs/upstream_model_feasibility.md`](docs/upstream_model_feasibility.md).

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

- [`docs/our_strategy.md`](docs/our_strategy.md): current execution strategy, RTX 3090 training role, and promotion gates
- [`docs/current_state.md`](docs/current_state.md): current candidate, active work, and next actions
- [`docs/score_optimization_40_day_roadmap.md`](docs/score_optimization_40_day_roadmap.md): experiment portfolio through final freeze
- [`docs/final_evaluation_server.md`](docs/final_evaluation_server.md): final GTX 1080/16 GB server and deployment acceptance contract
- [`docs/vessl_workflow.md`](docs/vessl_workflow.md): training and validation commands
- [`docs/phase2_plan.md`](docs/phase2_plan.md): scoring and official-evaluation rules
- [`docs/final_submission_checklist.md`](docs/final_submission_checklist.md): external upload checklist
- [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md): verified EXP030 official result
- [`reports/phase2/EXP031_validation_summary.md`](reports/phase2/EXP031_validation_summary.md): final EXP031 validation result
