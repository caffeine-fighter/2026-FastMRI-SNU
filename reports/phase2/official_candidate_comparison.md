# Official Phase 2 candidate comparison

These are VESSL `recon_eval` results. EXP012, EXP030, and EXP031 each have one official run; EXP030 also has a completed 30-run timing cohort.

## One-shot official runs

| Candidate | SSIM_full | SSIM_bbox | Quality | Time | Time score | Total score |
|---|---:|---:|---:|---:|---:|---:|
| `EXP012_official` | 0.9120 | 0.9040 | 0.9080 | 146.4 ms/slice | 0.0009654166666667 | 0.9089654166666667 |
| `EXP030_official` | 0.9178 | 0.9108 | 0.9143 | 173.3 ms/slice | 0.0009514062500000 | 0.9152514062500000 |
| `EXP031_official` | **0.9191** | **0.9114** | **0.91525** | 173.1 ms/slice | 0.0009515104166667 | **0.9162015104166666** |

## EXP031 versus EXP030 one-shot

| Metric | Delta |
|---|---:|
| SSIM_full | +0.0013 |
| SSIM_bbox | +0.0006 |
| Quality | +0.00095 |
| Time | -0.2 ms/slice |
| Time score | +0.0000001041666667 |
| Total score | +0.0009501041666666 |

EXP031 improves both quality components and is slightly faster in the one-shot comparison. There is no quality/timing tradeoff.

## EXP030 finalized 30-run result

| Metric | Final value |
|---|---:|
| Completed runs | 30 |
| Minimum valid time | 173.4 ms/slice |
| SSIM_full | 0.9178 |
| SSIM_bbox | 0.9108 |
| Quality | 0.9143 |
| Time score | 0.0009513541666667 |
| Total score | 0.9152513541666667 |

EXP031's one-shot score leads this finalized EXP030 score by `+0.00095015625`.

## Decision

EXP031 is the one-shot official leader and recommended replacement candidate. The final timing rule uses the minimum valid time from at least 30 official runs, so a separately approved EXP031 repeat cohort is required before final replacement and packaging.

Do not start the repeat cohort automatically.

## Evidence

- [`EXP012_official_20260710_003807/score.json`](EXP012_official_20260710_003807/score.json)
- [`EXP030_official_20260710_005122/score.json`](EXP030_official_20260710_005122/score.json)
- [`EXP031_official_20260711_014023/score.json`](EXP031_official_20260711_014023/score.json)
- [`repeat_EXP030_official_20260710_011126/repeat_summary.json`](repeat_EXP030_official_20260710_011126/repeat_summary.json)

Safety gates remain unchanged: do not modify `recon_eval.py`, do not touch mounted `Data`, and never commit model weights or reconstruction H5 files.
