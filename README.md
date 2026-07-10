# 2026 FastMRI SNU

VarNet experiments and Phase 2 submission tooling for the 2026 SNU FastMRI Challenge. VESSL is the source of truth for all `EXP###` runs and official evaluation.

<!-- EXP031_STATUS_START -->
## Live VESSL status

_Last update: 2026-07-11 05:31 KST (2026-07-10 20:31 UTC)_

| Check | Value |
|---|---|
| Run | `EXP031_varnet_c4_ch12_s8_e30` (running) |
| Progress | epoch `26/30`, iteration `4100/4651`, `68.8%` |
| ETA | `3.38 hours`; finish `2026-07-11 08:54 KST (2026-07-10 23:54 UTC)` |
| Best validation loss | epoch `23`: `3.187740346715812` |
| Validation snapshot | `interim epoch-23`: full `0.904252`, bbox `0.928967`, quality `0.916609` (`+0.001930` vs EXP030) |
| Health | `0` error matches; checkpoints present |

`EXP030` remains the official candidate until EXP031 completes validation and receives approval. No official evaluation starts automatically.
<!-- EXP031_STATUS_END -->

## Official result

| Candidate | Config | SSIM_full | SSIM_bbox | Quality | Min time | Total score |
|---|---|---:|---:|---:|---:|---:|
| `EXP030_varnet_c4_ch12_s8_e20` | c4 / ch12 / s8 / 20 epochs | 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | **0.9152513541666667** |

The official 30-run timing evaluation is complete. The remaining submission task is the external organizer upload. EXP031 is a follow-up candidate, not an automatic replacement.

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
- [`docs/vessl_workflow.md`](docs/vessl_workflow.md): training and validation commands
- [`docs/phase2_plan.md`](docs/phase2_plan.md): scoring and official-evaluation rules
- [`docs/final_submission_checklist.md`](docs/final_submission_checklist.md): external upload checklist
- [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md): verified EXP030 result
