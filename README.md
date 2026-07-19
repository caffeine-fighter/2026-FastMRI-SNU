# 2026 FastMRI SNU

VarNet experiments and Phase 2 submission tooling for the 2026 SNU FastMRI Challenge. VESSL is the source of truth for all `EXP###` runs and official evaluation.

<!-- EXP031_STATUS_START -->
## Live VESSL publisher status

_Last publisher update: 2026-07-19 08:22 KST (2026-07-18 23:22 UTC)_

| Check | Value |
|---|---|
| Experiment | `EXP037` — matched standard Control |
| Run | `EXP037_MATCHED_STANDARD_CONTROL_E30_TO_E35_R1` (not running) |
| Manifest status | `TRAINING_COMPLETE` |
| Design | epoch-30→35 exact continuation; Adam LR `0.001`; c8/ch12/s8; batch 1; seed `430` |
| Progress | epoch `34/35`, iteration `4500/4651`, continuation `99.4%`; latest loss `0.09342` |
| Completed epochs | `5/5`; retained epoch checkpoints observed `0` |
| Best validation loss | epoch `33`: `3.130437840` |
| Provenance health | sampler epoch-state records `5/5`; error matches `0` |

The Control publisher manifest is terminal, but the Candidate remains blocked until the final manifest SHA, effective LR, dataset identity, source generation, and epochs 31–35 sampler states pass the fail-closed gate. Strict validation and official evaluation are not launched by this publisher.
<!-- EXP031_STATUS_END -->

## Prize-first research queue

_Operational snapshot: 2026-07-19 14:47 KST._

- EXP035 epoch 30 remains the protected official leader at total `0.92146109375`.
- R1 ended naturally and is disabled. R4.1 owns durable monitoring, but its dispatcher is `ACTIVE_SAFE_IDLE` because there is no valid reviewed launch manifest.
- The local RTX 3090 is compute-idle; the RunPod A6000 remains occupied by a protected SNU AI Challenge job; the VESSL workspace control plane is running, the EXP037 Control publisher is terminal, and Candidate progress is unavailable from the local observer.
- Annotation-aware V1 is `SOURCE_DIVERGED / REVIEW_INVALIDATED`, not launchable. PromptMR requires a finite diagnostic before any new scratch production run.
- Organizer Issues [#408](https://github.com/LISTatSNU/FastMRI_challenge/issues/408#issuecomment-5013509585) and [#409](https://github.com/LISTatSNU/FastMRI_challenge/issues/409#issuecomment-5013857396) now fix the external-weight and external-compute/augmentation/ensemble/cloud rules.
- Midnight is a snapshot, not a shutdown or dispatch cutoff.

See [`docs/prize_first_r4_status.md`](docs/prize_first_r4_status.md) for the current queue and [`docs/official_rule_clarifications_20260719.md`](docs/official_rule_clarifications_20260719.md) for the rule contract.

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

## Latest matched continuation result

The VESSL EXP035 epoch-30 continuation pair is complete. Lowering Adam LR from `1e-3` to `3e-4` improved the Candidate over the independently continued Control by only `+0.0003468637`, below the preregistered `+0.0005` gate, while acc8 bbox decreased by `0.0003703822`. The lower-LR Candidate, second seed, epoch-40 continuation, and c9/c10/c12 vanilla expansion are closed without official evaluation; Candidate epoch 34 is a research artifact only. The target total is `0.94`, leaving `+0.01853890625` at EXP035's time score, so the next bounded work is the pinned MIT Feature/FI source/license and 8 GB inference feasibility gate. PromptMR+ remains blocked pending written competition/license confirmation. See the [matched continuation report](reports/local_comparisons/exp035_matched_continuation_r1_20260716.md) and [`docs/upstream_model_feasibility.md`](docs/upstream_model_feasibility.md).

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
- Do not run official evaluation while training is active. Standing authorization permits eligible evaluation and submission only after exact candidate, evaluator, provenance, coverage, archive, and hardware gates pass.
- Treat mounted `Data` directories as read-only.
- Never commit data, H5 files, checkpoints, result directories, `.env` files, or credentials.
- Keep LOCAL/RunPod probes exploratory; only end-to-end VESSL scratch runs can become official candidates, and learned state must not cross that boundary.

## Documentation

Start with [`docs/README.md`](docs/README.md). The most useful pages are:

- [`docs/our_strategy.md`](docs/our_strategy.md): current execution strategy, RTX 3090 training role, and promotion gates
- [`docs/current_state.md`](docs/current_state.md): current candidate, active work, and next actions
- [`docs/prize_first_r4_status.md`](docs/prize_first_r4_status.md): R4.1 continuity, live research gates, and compute-lane ownership
- [`docs/official_rule_clarifications_20260719.md`](docs/official_rule_clarifications_20260719.md): organizer answers for weights, external screening, augmentation/remasking, ensemble/TTA, and cloud processing
- [`docs/score_optimization_40_day_roadmap.md`](docs/score_optimization_40_day_roadmap.md): experiment portfolio through final freeze
- [`docs/final_evaluation_server.md`](docs/final_evaluation_server.md): final GTX 1080/16 GB server and deployment acceptance contract
- [`docs/vessl_workflow.md`](docs/vessl_workflow.md): training and validation commands
- [`docs/phase2_plan.md`](docs/phase2_plan.md): scoring and official-evaluation rules
- [`docs/final_submission_checklist.md`](docs/final_submission_checklist.md): external upload checklist
- [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md): verified EXP030 official result
- [`reports/phase2/EXP031_validation_summary.md`](reports/phase2/EXP031_validation_summary.md): final EXP031 validation result
