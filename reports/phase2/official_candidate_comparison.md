# Official Phase 2 Candidate Comparison

Source files:

- `reports/phase2/scoreboard.csv`
- `reports/phase2/EXP012_official_20260710_003807/score.json`
- `reports/phase2/EXP030_official_20260710_005122/score.json`

These are official one-shot VESSL `recon_eval` wrapper results. No additional evaluation was run to generate this report.

## Official runs

| candidate | Leaderboard SSIM_full | Leaderboard SSIM_bbox | quality_score | time_ms_per_slice | time_score | total_score |
|---|---:|---:|---:|---:|---:|---:|
| `EXP012_official` | 0.912 | 0.904 | 0.908 | 146.4 | 0.0009654166666666666 | 0.9089654166666667 |
| `EXP030_official` | 0.9178 | 0.9108 | 0.9143 | 173.3 | 0.00095140625 | 0.91525140625 |

## EXP030 minus EXP012

| metric | difference |
|---|---:|
| Leaderboard SSIM_full | 0.0058 |
| Leaderboard SSIM_bbox | 0.0068 |
| quality_score | 0.0063 |
| time_ms_per_slice | 26.9 |
| time_score | -0.0000140104166666666 |
| total_score | 0.0062859895833333 |

## Decision

Decision rule:

- If the quality-score difference is greater than `0.001`, prefer higher quality.
- If the difference is at most `0.001`, timing can decide.
- Prioritize `SSIM_bbox` when the full/bbox tradeoff is unclear.

Result:

- **Current best official candidate: `EXP030_official`.**
- Reason: quality_score gap 0.0063 is greater than 0.001.
- `EXP030_official` improves both `SSIM_full` and `SSIM_bbox`; there is no conflicting full/bbox tradeoff.
- Although EXP030 is 26.9 ms/slice slower in this one-shot comparison, its time-score penalty relative to EXP012 is only 0.0000140104166666666, far smaller than its quality gain of 0.0063.
- Its one-shot total-score advantage is 0.0062859895833333.

## Completed 30-repeat result

The approved 30-repeat timing suite completed successfully for
`EXP030_official`; no additional official evaluation is needed.

| metric | final value |
|---|---:|
| completed runs | 30 |
| best by minimum time | `EXP030_official_run01` |
| best by total score | `EXP030_official_run01` |
| minimum time_ms_per_slice | 173.4 |
| SSIM_full | 0.9178 |
| SSIM_bbox | 0.9108 |
| quality_score | 0.9143 |
| time_score | 0.0009513541666666666 |
| total_score | 0.9152513541666667 |

All 30 runs produced the same quality metrics. The final timing uses the
minimum valid value, as required. See `final_score_summary.md` and
`repeat_EXP030_official_20260710_011126/repeat_summary.json`.

Safety gates remain unchanged: do not modify `recon_eval.py`, do not touch
mounted `Data`, and never commit model weights or reconstruction H5 files.
