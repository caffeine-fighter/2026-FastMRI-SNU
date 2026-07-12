# 2026 FastMRI SNU

VarNet experiments and Phase 2 submission tooling for the 2026 SNU FastMRI Challenge. VESSL is the source of truth for all `EXP###` runs and official evaluation.

<!-- EXP031_STATUS_START -->
## Live VESSL status

_Last update: 2026-07-13 01:56 KST (2026-07-12 16:56 UTC)_

| Check | Value |
|---|---|
| Run | `EXP032_varnet_c6_ch12_s8_e30` (running) |
| Change | `cascades 4 to 6; all else fixed` |
| Progress | epoch `28/30`, iteration `4650/4651`, `96.7%` |
| ETA | `1.42 hours`; finish `2026-07-13 03:21 KST (2026-07-12 18:21 UTC)` |
| Best validation loss | epoch `26`: `3.1754857592465457` |
| Validation snapshot | pending final validation |
| Health | `0` error matches; checkpoints `present` |

`EXP031` remains the one-shot official leader while this one-variable experiment trains. No official evaluation starts automatically.
<!-- EXP031_STATUS_END -->

## Official result

| Candidate | Evidence | SSIM_full | SSIM_bbox | Quality | Time | Total score |
|---|---|---:|---:|---:|---:|---:|
| `EXP030_varnet_c4_ch12_s8_e20` | 30-run minimum | 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | 0.9152513541666667 |
| `EXP031_varnet_c4_ch12_s8_e30` | one official run | **0.9191** | **0.9114** | **0.91525** | **173.1 ms/slice** | **0.9162015104166666** |

EXP031 leads the completed official runs by `+0.00095015625` versus EXP030's finalized 30-run score. It stays protected while the 40-day optimization program searches for a stronger final candidate; the final timing cohort is deferred until model freeze.

Local promotion uses leaderboard-faithful equal-acceleration validation quality. EXP031's reference is `0.9154137446412757`; its historical pooled diagnostic `0.9174406281804748` is not the promotion threshold.

## Desktop LOCAL candidate status

_Last verified: 2026-07-13 01:44 KST (2026-07-12 16:44 UTC). These are RTX 4070 Ti SUPER exploratory results, not official VESSL scores or launch authority._

| Candidate | Configuration / evidence | Equal-acc quality | Status |
|---|---|---:|---|
| `LOCAL_EXP039` | c4/ch12/s12/e1, seed 431 | 0.882647730465 | rejected: seed-replication gate failed |
| `LOCAL_EXP040` | 50/50 average of c4/ch16/s8 epochs 5 and 10 | 0.907946926286 | rejected: -0.001647492152 versus epoch 10 |
| `LOCAL_EXP041` | c4/ch16/s8, epoch 10 -> 15 at LR 3e-4 | 0.913672051958 | retained same-basin training leader; all component floors passed |
| `LOCAL_EXP042` | attempted epoch 15 -> 18 continuation at LR 1e-4 | n/a | technical failure before checkpoint or retained result publication |
| `LOCAL_EXP043` | EXP042 plumbing retry; recovered orphan epoch-16 reconstruction | 0.913476635177 | diagnostic only; -0.000195416780 versus EXP041; reject LR 1e-4 direction |
| `LOCAL_EXP044` | matched standard-SSIM epoch-16 continuation at LR 3e-4 | 0.913223060119 | rejected; -0.000448991839 versus EXP041 |
| `LOCAL_EXP046` | matched epoch-16 sparse metric-aligned objective | 0.911710246795 | rejected; -0.001512813323 versus EXP044 and every protected component regressed |
| `LOCAL_EXP047` | 75/25 image-space blend of EXP041 and EXP046 epoch 16 | 0.913973642372 | robust +0.000301591492 versus EXP041, but below EXP031 and requires two forwards |
| `LOCAL_EXP048` | 75/25 same-basin parameter interpolation, inference only | 0.913996916454 | post-run exact-byte review rejected; method gate void; non-authoritative diagnostic only |

EXP043 had no successful training terminal or epoch-16 checkpoint, and its delayed adversarial launch review failed. EXP044 then closed ordinary fixed-LR `3e-4` duration escalation, and EXP046 rejected the first sparse metric-aligned objective. EXP047's paired 200,000-replicate volume-cluster bootstrap supported its small blend gain, but it remained `-0.001440102540` below the documented EXP031 LOCAL reference. EXP048 execution produced quality `0.913996916454`, but a previously dispatched exact-byte reviewer returned six blocking runner/evidence findings. The approval is revoked, the method gate is void, and the metrics are retained only as non-authoritative post-run diagnostic data; they cannot authorize SWA, promotion, or official follow-up.

Source-backed report: [`reports/local_comparisons/local_continuation_campaign_20260711.md`](reports/local_comparisons/local_continuation_campaign_20260711.md).

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
