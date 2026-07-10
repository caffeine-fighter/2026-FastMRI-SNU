# Phase 2 Final Score Summary

## Final candidate

- Candidate tag: `EXP030_official`
- Experiment: `EXP030`
- Checkpoint: `/root/result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt`
- Checkpoint symlink: `checkpoints_phase2/best_model.pt`
- Checkpoint SHA-256: `ef74bec4243e7aa39d5aa8dae031e1bb83e771c26be00c4e5330e17b60a66085`
- Model configuration: cascade `4`, channels `12`, sensitivity channels `8`

The checkpoint symlink resolved to the selected EXP030 checkpoint before and after evaluation. The symlink and target hashes matched.

## Official 30-repeat result

The official command completed all 30 runs successfully:

```bash
bash scripts/repeat_recon_eval.sh EXP030_official 30
```

| metric | value |
|---|---:|
| completed runs | 30 |
| best_by_time | `EXP030_official_run01` |
| best_by_total_score | `EXP030_official_run01` |
| minimum time_ms_per_slice | 173.4 |
| SSIM_full | 0.9178 |
| SSIM_bbox | 0.9108 |
| quality_score | 0.9143 |
| time_score | 0.0009513541666666666 |
| total_score | 0.9152513541666667 |

`EXP030_official_run01` is both the minimum-time run and the highest-total-score run. All 30 repeats reported `SSIM_full = 0.9178` and `SSIM_bbox = 0.9108`.

## Result artifacts

- [Repeat summary JSON](repeat_EXP030_official_20260710_011126/repeat_summary.json)
- [Repeat summary CSV](repeat_EXP030_official_20260710_011126/repeat_summary.csv)
- [Best-run evaluation log](repeat_EXP030_official_20260710_011126/EXP030_official_run01_20260710_011126/eval_run.log)
- [Best-run score JSON](repeat_EXP030_official_20260710_011126/EXP030_official_run01_20260710_011126/score.json)
- [Official one-shot score JSON](EXP030_official_20260710_005122/score.json)
- [Official candidate comparison](official_candidate_comparison.md)
- [Phase 2 scoreboard](scoreboard.csv)
- [Model-description deck](EXP030_model_description.pptx)
- [Final submission checklist](../../docs/final_submission_checklist.md)

## Selection rationale

EXP030 remains the final candidate because the official one-shot comparison showed a quality-score advantage of `0.0063` over EXP012, exceeding the `0.001` threshold at which quality takes priority over timing. EXP030 also improved both `SSIM_full` and `SSIM_bbox`, so there was no conflicting full/bbox tradeoff. The completed 30-repeat run confirms stable quality and selects the minimum valid official timing result for the final score.

## Integrity checks

- `scripts/phase2_preflight.sh` passed before the repeat run.
- `recon_eval.py` remained unmodified; SHA-256: `a93cf978b4938a060d4f5a204d3f7118fb8c17bf12408cbe44e6e7954ba5a135`.
- Mounted `/root/Data` remained a read-only input mount.
- No training process ran concurrently with the official repeat evaluation.
- No checkpoint or mounted Data file is included in this report.
- Submission implementation commit `fbbddf6700cd65b1e2b52c1c6418f48a5eef9b82` was pushed to `phase2/eval-wrapper-vessl` and fast-forwarded into the GitHub default branch `baseline/2026-baby-varnet`; a fresh clone was verified.
