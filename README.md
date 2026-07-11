# 2026 FastMRI SNU

VarNet experiments and Phase 2 submission tooling for the 2026 SNU FastMRI Challenge. VESSL is the source of truth for all `EXP###` runs and official evaluation. Desktop `LOCAL_` probes are exploratory only.

<!-- EXP031_STATUS_START -->
## Live VESSL status

_Last update: 2026-07-11 08:59 KST (2026-07-10 23:59 UTC)_

| Check | Value |
|---|---|
| Run | `EXP031_varnet_c4_ch12_s8_e30` (training complete) |
| Progress | epoch `29/30`, iteration `4650/4651`, `100.0%` |
| Best validation loss | epoch `27`: `3.1818922822357556` |
| Validation snapshot | `final telemetry`: full `0.904501`, bbox `0.930380`, quality `0.917441` (`+0.002762` vs EXP030 validation quality) |
| Handoff | final experiment-log row and `EXP031_validation_handoff.json` pending |
| Health | `0` error matches; checkpoints reported present |

`EXP030` remains the official candidate until the EXP031 checkpoint identity and file-backed validation handoff are verified and replacement is approved. No official evaluation starts automatically.
<!-- EXP031_STATUS_END -->

## Official result

| Candidate | Config | SSIM_full | SSIM_bbox | Quality | Min time | Total score |
|---|---|---:|---:|---:|---:|---:|
| `EXP030_varnet_c4_ch12_s8_e20` | c4 / ch12 / s8 / 20 epochs | 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | **0.9152513541666667** |

The official 30-run timing evaluation is complete. The remaining submission task is the external organizer upload. EXP031 is a follow-up candidate, not an automatic replacement.

## LOCAL exploratory result

The desktop campaign completed 17 one-epoch probes and five adaptive follow-ups without failed runs or skipped validation files.

| Probe | Config | Seed | SSIM_full | SSIM_bbox | Local quality |
|---|---|---:|---:|---:|---:|
| `LOCAL_EXP029` | c4/ch12/s8/e5 | 430 | 0.8957743251 | 0.9116992114 | 0.9037367682 |
| `LOCAL_EXP032` | c4/ch16/s8/e5 | 430 | 0.8986326562 | 0.9185488328 | 0.9085907445 |
| `LOCAL_EXP033` | c4/ch12/s8/e1 | 431 | 0.8823435134 | 0.8932403356 | 0.8877919245 |
| `LOCAL_EXP034` | c4/ch16/s8/e1 | 431 | 0.8808135834 | 0.8923592767 | 0.8865864301 |
| `LOCAL_EXP035` | c4/ch16/s8/e10 | 430 | 0.9021009122 | 0.9229051363 | 0.9125030243 |

The c4/ch16/s8 candidate won the matched seed-430 five-epoch comparison by `+0.0048539763` quality and improved from e5 to e10 by `+0.0039122798`. It failed seed confirmation: seed-431 quality changed by `-0.0012054944`, and full-image SSIM changed by `-0.0015299300`, below the allowed `-0.001` component floor.

**LOCAL decision: do not promote c4/ch16/s8.** LOCAL checkpoints, metrics, and timing are never official.

Source-backed reports:

- [`reports/local_comparisons/local_probe_summary.md`](reports/local_comparisons/local_probe_summary.md)
- [`reports/local_comparisons/local_probe_sweep_20260711_round2.md`](reports/local_comparisons/local_probe_sweep_20260711_round2.md)
- [`reports/local_comparisons/local_probe_adaptive_followup_20260711_final.md`](reports/local_comparisons/local_probe_adaptive_followup_20260711_final.md)
- [`reports/local_comparisons/local_probe_adaptive_followup_20260711.json`](reports/local_comparisons/local_probe_adaptive_followup_20260711.json)
- [`reports/local_comparisons/local_probe_adaptive_followup_plan_20260711.json`](reports/local_comparisons/local_probe_adaptive_followup_plan_20260711.json)

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

`recon_eval.sh` defaults to the final EXP030 architecture (`cascade=4`, `chans=12`, `sens_chans=8`) and invokes the fixed `recon_eval.py` harness. Mounted leaderboard data must remain read-only.

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

The official 30-repeat EXP030 evaluation has already completed. Do not repeat it for submission; see [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md).

## Safety rules

Never commit data, H5 files, checkpoints, result directories, `.env` files, or credentials. Do not modify `recon_eval.py`, mounted `Data`, model code, loss code, or official metric implementations without explicit approval. Run official evaluation only on VESSL after approval.

## Repository map

| Path | Purpose |
|---|---|
| `train.py` | VESSL training entrypoint |
| `reconstruct.py` | Reconstruction entrypoint |
| `recon_eval.sh` | Official Phase 2 entrypoint |
| `scripts/` | Validation, preflight, scoring, and reporting helpers |
| `experiments/experiment_log.csv` | Experiment registry |
| `reports/phase2/` | Official score and submission reports |
| `reports/local_comparisons/` | Exploratory LOCAL evidence |
| `docs/` | Workflow, status, rules, and decision history |

## Documentation

Start with [`docs/README.md`](docs/README.md). Key pages:

- [`docs/current_state.md`](docs/current_state.md): current candidate, completed LOCAL study, and next actions
- [`docs/exp031_post_training_handoff.md`](docs/exp031_post_training_handoff.md): required EXP031 handoff schema
- [`docs/vessl_workflow.md`](docs/vessl_workflow.md): training and validation commands
- [`docs/phase2_plan.md`](docs/phase2_plan.md): scoring and official-evaluation rules
- [`docs/final_submission_checklist.md`](docs/final_submission_checklist.md): external upload checklist
- [`reports/phase2/final_score_summary.md`](reports/phase2/final_score_summary.md): verified EXP030 result
