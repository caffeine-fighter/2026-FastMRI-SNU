# EXP031 validation summary

`EXP031_varnet_c4_ch12_s8_e30` completed VESSL training through epoch 29 and passed final validation-only evaluation.

## Training

| Check | Result |
|---|---|
| Completion | clean exit code 0 |
| Final epoch | 29 |
| Best epoch | 27 |
| Best validation loss | 3.1818922822357556 |
| Error-pattern matches | 0 |
| Best checkpoint SHA-256 | `3e68d94922f68d9a536e4bdbe7802785f8b43792524ddac252dbbe8d5c11d31f` |

## Validation comparison

| Metric | EXP030 validation | EXP031 validation | Delta |
|---|---:|---:|---:|
| SSIM_full | 0.903370354781 | **0.904500772769** | +0.001130417988 |
| SSIM_bbox | 0.925987817116 | **0.930380483592** | +0.004392666476 |
| Quality | 0.914679085949 | **0.917440628180** | +0.002761542231 |

Coverage:

- 30 volumes
- 791 slices
- 161 bounding-box annotations
- 0 skipped inputs

Acceleration breakdown:

| Scope | SSIM_full | SSIM_bbox |
|---|---:|---:|
| acc4 | 0.919428644661 | 0.937730002626 |
| acc8 | 0.888678783551 | 0.915817547728 |

## Official evaluation

The first approved official run completed successfully:

| Metric | EXP030 finalized | EXP031 one-shot | Delta |
|---|---:|---:|---:|
| SSIM_full | 0.9178 | **0.9191** | +0.0013 |
| SSIM_bbox | 0.9108 | **0.9114** | +0.0006 |
| Quality | 0.9143 | **0.91525** | +0.00095 |
| Time | 173.4 ms/slice | **173.1 ms/slice** | -0.3 ms/slice |
| Total score | 0.9152513541666667 | **0.9162015104166666** | +0.00095015625 |

Evidence: [`EXP031_official_20260711_014023/score.json`](EXP031_official_20260711_014023/score.json).

## Recommendation

EXP031 is the validation and one-shot official leader. Run a 30-repeat EXP031 timing cohort only after separate approval, then replace and package EXP030 only if EXP031 remains ahead.
