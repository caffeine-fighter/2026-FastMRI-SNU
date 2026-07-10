# Desktop LOCAL Probe Summary

Generated from `/home/ray1001/result/LOCAL_*/metrics/metrics.csv`, loss logs, and `skipped.json`.

**Scope:** Every `LOCAL_` result is an exploratory desktop probe. It is not an official VESSL score, final candidate, or submission checkpoint.

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
```

`forward_time_s` is a noisy local diagnostic from the training log, not official `recon_eval` timing and not `ms/slice`.

## Metrics

| exp_id | c | ch | s | SSIM_full | SSIM_bbox | quality | acc4 quality | acc8 quality | val_loss | forward_time_s | skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `LOCAL_EXP012_varnet_c4_ch12_s4_e1` | 4 | 12 | 4 | 0.8794828387 | 0.8889114335 | 0.8841971361 | 0.8985789049 | 0.8644184298 | 3.7162691977 | 0.3020 | `[]` |
| `LOCAL_EXP013_varnet_c4_ch12_s8_e1` | 4 | 12 | 8 | 0.8807929440 | 0.8890808084 | 0.8849368762 | 0.9012535192 | 0.8616591192 | 3.7060742388 | 0.3446 | `[]` |
| `LOCAL_EXP014_varnet_c6_ch12_s8_e1` | 6 | 12 | 8 | 0.8811104767 | 0.8903269135 | 0.8857186951 | 0.9017071749 | 0.8631502224 | 3.7675869340 | 0.4382 | `[]` |
| `LOCAL_EXP015_varnet_c6_ch12_s4_e1` | 6 | 12 | 4 | 0.8747640465 | 0.8805270891 | 0.8776455678 | 0.8930437597 | 0.8563069000 | 3.9070480032 | 0.3065 | `[]` |
| `LOCAL_EXP016_varnet_c3_ch12_s8_e1` | 3 | 12 | 8 | 0.8775208604 | 0.8807431730 | 0.8791320167 | 0.8943828902 | 0.8578022018 | 3.8033460612 |  | `[]` |
| `LOCAL_EXP017_varnet_c5_ch12_s8_e1` | 5 | 12 | 8 | 0.8792087180 | 0.8883334321 | 0.8837710750 | 0.8991224114 | 0.8624411931 | 3.7508514223 | 0.2931 | `[]` |
| `LOCAL_EXP018_varnet_c4_ch16_s8_e1` | 4 | 16 | 8 | 0.8854211528 | 0.9004961319 | 0.8929586423 | 0.9092240471 | 0.8699221048 | 3.6148317265 | 0.3057 | `[]` |
| `LOCAL_EXP019_varnet_c4_ch12_s12_e1` | 4 | 12 | 12 | 0.8844853268 | 0.8948144457 | 0.8896498863 | 0.9045920514 | 0.8686692827 | 3.6491647699 | 0.3080 | `[]` |
| `LOCAL_EXP020_varnet_c4_ch18_s8_e1` | 4 | 18 | 8 | 0.8817238236 | 0.8924220486 | 0.8870729361 | 0.9021213991 | 0.8660649222 | 3.6518833169 | 0.4023 | `[]` |
| `LOCAL_EXP021_varnet_c6_ch12_s12_e1` | 6 | 12 | 12 | 0.8844415618 | 0.8939631755 | 0.8892023687 | 0.9070642763 | 0.8627274112 | 3.7623022610 | 0.3641 | `[]` |
| `LOCAL_EXP022_varnet_c4_ch16_s12_e1` | 4 | 16 | 12 | 0.8831212078 | 0.8911024973 | 0.8871118525 | 0.9033776048 | 0.8636826972 | 3.6625753789 | 0.3438 | `[]` |
| `LOCAL_EXP023_varnet_c6_ch16_s8_e1` | 6 | 16 | 8 | 0.8792693435 | 0.8890259181 | 0.8841476308 | 0.8999979823 | 0.8617481715 | 3.7699207988 | 0.3505 | `[]` |

## Ranking by one-epoch local quality

1. `LOCAL_EXP018_varnet_c4_ch16_s8_e1` — quality=0.8929586423, SSIM_full=0.8854211528, SSIM_bbox=0.9004961319
2. `LOCAL_EXP019_varnet_c4_ch12_s12_e1` — quality=0.8896498863, SSIM_full=0.8844853268, SSIM_bbox=0.8948144457
3. `LOCAL_EXP021_varnet_c6_ch12_s12_e1` — quality=0.8892023687, SSIM_full=0.8844415618, SSIM_bbox=0.8939631755
4. `LOCAL_EXP022_varnet_c4_ch16_s12_e1` — quality=0.8871118525, SSIM_full=0.8831212078, SSIM_bbox=0.8911024973
5. `LOCAL_EXP020_varnet_c4_ch18_s8_e1` — quality=0.8870729361, SSIM_full=0.8817238236, SSIM_bbox=0.8924220486
6. `LOCAL_EXP014_varnet_c6_ch12_s8_e1` — quality=0.8857186951, SSIM_full=0.8811104767, SSIM_bbox=0.8903269135
7. `LOCAL_EXP013_varnet_c4_ch12_s8_e1` — quality=0.8849368762, SSIM_full=0.8807929440, SSIM_bbox=0.8890808084
8. `LOCAL_EXP012_varnet_c4_ch12_s4_e1` — quality=0.8841971361, SSIM_full=0.8794828387, SSIM_bbox=0.8889114335
9. `LOCAL_EXP023_varnet_c6_ch16_s8_e1` — quality=0.8841476308, SSIM_full=0.8792693435, SSIM_bbox=0.8890259181
10. `LOCAL_EXP017_varnet_c5_ch12_s8_e1` — quality=0.8837710750, SSIM_full=0.8792087180, SSIM_bbox=0.8883334321
11. `LOCAL_EXP016_varnet_c3_ch12_s8_e1` — quality=0.8791320167, SSIM_full=0.8775208604, SSIM_bbox=0.8807431730
12. `LOCAL_EXP015_varnet_c6_ch12_s4_e1` — quality=0.8776455678, SSIM_full=0.8747640465, SSIM_bbox=0.8805270891

## Interpretation

- `LOCAL_EXP018` (`c4/ch16/s8`) is the new one-epoch LOCAL leader at `0.8929586423`.
- Against `LOCAL_EXP013` (`c4/ch12/s8`), `LOCAL_EXP018` gains `+0.0080217661` quality while keeping the same cascade and sensitivity settings.
- Increasing cascade width further to 18 (`LOCAL_EXP020`) loses `-0.0058857063` versus ch16, so the one-epoch response is non-monotonic.
- Widening only sensitivity channels to 12 (`LOCAL_EXP019`) gains `+0.0047130101` over `c4/ch12/s8`.
- Combining ch16 and s12 (`LOCAL_EXP022`) loses `-0.0058467898` versus `c4/ch16/s8`; the width gains do not combine at one epoch.
- `LOCAL_EXP021` (`c6/ch12/s12`) gains `+0.0034836736` over the previous `c6/ch12/s8` probe but remains below `LOCAL_EXP018`.
- `LOCAL_EXP023` (`c6/ch16/s8`) loses `-0.0088110116` versus `c4/ch16/s8`; more cascades did not help this wider model at one epoch.
- These are one-epoch LOCAL rankings and may not predict long-run VESSL ordering. Do not use any LOCAL checkpoint as a candidate.

## Recommendation

- Let the active VESSL EXP030 epoch-21–30 continuation finish without any Git or GPU interference from this branch.
- After the official continuation is validated, `c4/ch16/s8` is the strongest architecture suggested by this LOCAL sweep if another VESSL experiment is explicitly approved.
- Do not merge this local-probe branch into the VESSL/default branch while VESSL is training; review it after the VESSL handoff.
