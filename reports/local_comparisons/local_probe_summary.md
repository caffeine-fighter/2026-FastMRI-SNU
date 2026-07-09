# Desktop LOCAL Probe Summary

Generated from `../result/LOCAL_*/metrics/metrics.csv` and each probe's `skipped.json`.

**Scope:** LOCAL_ desktop probes are exploratory only. They are not official VESSL scores and must not be treated as final candidates.

Formula:

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
```

## Metrics table

| exp_id | cascade | chans | sens_chans | epochs | SSIM_full | SSIM_bbox | quality_score | acc4 quality | acc8 quality | skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `LOCAL_EXP012_varnet_c4_ch12_s4_e1` | 4 | 12 | 4 | 1 | 0.8794828387 | 0.8889114335 | 0.8841971361 | 0.8985789049 | 0.8644184298 | `[]` |
| `LOCAL_EXP013_varnet_c4_ch12_s8_e1` | 4 | 12 | 8 | 1 | 0.8807929440 | 0.8890808084 | 0.8849368762 | 0.9012535192 | 0.8616591192 | `[]` |
| `LOCAL_EXP014_varnet_c6_ch12_s8_e1` | 6 | 12 | 8 | 1 | 0.8811104767 | 0.8903269135 | 0.8857186951 | 0.9017071749 | 0.8631502224 | `[]` |
| `LOCAL_EXP015_varnet_c6_ch12_s4_e1` | 6 | 12 | 4 | 1 | 0.8747640465 | 0.8805270891 | 0.8776455678 | 0.8930437597 | 0.8563069000 | `[]` |
| `LOCAL_EXP016_varnet_c3_ch12_s8_e1` | 3 | 12 | 8 | 1 | 0.8775208604 | 0.8807431730 | 0.8791320167 | 0.8943828902 | 0.8578022018 | `[]` |

## Ranking by overall quality

1. `LOCAL_EXP014_varnet_c6_ch12_s8_e1` — quality=0.8857186951, SSIM_full=0.8811104767, SSIM_bbox=0.8903269135
2. `LOCAL_EXP013_varnet_c4_ch12_s8_e1` — quality=0.8849368762, SSIM_full=0.8807929440, SSIM_bbox=0.8890808084
3. `LOCAL_EXP012_varnet_c4_ch12_s4_e1` — quality=0.8841971361, SSIM_full=0.8794828387, SSIM_bbox=0.8889114335
4. `LOCAL_EXP016_varnet_c3_ch12_s8_e1` — quality=0.8791320167, SSIM_full=0.8775208604, SSIM_bbox=0.8807431730
5. `LOCAL_EXP015_varnet_c6_ch12_s4_e1` — quality=0.8776455678, SSIM_full=0.8747640465, SSIM_bbox=0.8805270891

## Interpretation

- `LOCAL_EXP014_varnet_c6_ch12_s8_e1` (`c6/ch12/s8`) is the local 1-epoch quality leader.
- `LOCAL_EXP014` beats `LOCAL_EXP013` by only about `0.00078` quality, so Phase 2 timing may matter.
- `LOCAL_EXP016_varnet_c3_ch12_s8_e1` is quality-lower and unlikely to recover with timing alone.
- `LOCAL_EXP015_varnet_c6_ch12_s4_e1` is clearly weaker than `LOCAL_EXP014` (`c6/ch12/s8`).
- `LOCAL_` results are exploratory only and not official VESSL scores.

## Recommendation

- Do not prioritize `c6` on VESSL until `EXP030` official/validation results are known.
- Use `EXP030` / `EXP012` official `recon_eval` after VESSL training finishes.
