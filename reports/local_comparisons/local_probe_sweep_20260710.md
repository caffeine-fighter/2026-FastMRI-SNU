# LOCAL Probe Sweep — 2026-07-10/11

Six serial one-epoch desktop probes ran on `Cafe-Pastelica` with an RTX 4070 Ti SUPER. All completed successfully with `skipped=[]`.

These results are exploratory only. No LOCAL checkpoint is an official candidate, and no official `recon_eval.sh` was run.

## Sweep ranking

| rank | experiment | config | SSIM_full | SSIM_bbox | quality | val_loss | local forward s |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `LOCAL_EXP018_varnet_c4_ch16_s8_e1` | c4/ch16/s8/e1 | 0.8854211528 | 0.9004961319 | 0.8929586423 | 3.6148317265 | 0.3057 |
| 2 | `LOCAL_EXP019_varnet_c4_ch12_s12_e1` | c4/ch12/s12/e1 | 0.8844853268 | 0.8948144457 | 0.8896498863 | 3.6491647699 | 0.3080 |
| 3 | `LOCAL_EXP021_varnet_c6_ch12_s12_e1` | c6/ch12/s12/e1 | 0.8844415618 | 0.8939631755 | 0.8892023687 | 3.7623022610 | 0.3641 |
| 4 | `LOCAL_EXP022_varnet_c4_ch16_s12_e1` | c4/ch16/s12/e1 | 0.8831212078 | 0.8911024973 | 0.8871118525 | 3.6625753789 | 0.3438 |
| 5 | `LOCAL_EXP020_varnet_c4_ch18_s8_e1` | c4/ch18/s8/e1 | 0.8817238236 | 0.8924220486 | 0.8870729361 | 3.6518833169 | 0.4023 |
| 6 | `LOCAL_EXP023_varnet_c6_ch16_s8_e1` | c6/ch16/s8/e1 | 0.8792693435 | 0.8890259181 | 0.8841476308 | 3.7699207988 | 0.3505 |

## Main result

`LOCAL_EXP018_varnet_c4_ch16_s8_e1` is the new one-epoch desktop leader with quality `0.8929586423`.

- Delta versus `LOCAL_EXP013 c4/ch12/s8`: `+0.0080217661` quality.
- Delta versus prior leader `LOCAL_EXP014 c6/ch12/s8`: `+0.0072399473` quality.
- The ch18, combined ch16/s12, and c6/ch16 probes did not beat c4/ch16/s8 at one epoch.
- Local forward diagnostics are noisy and cannot replace official VESSL timing.

## Collision and safety controls

- Runs used unique `LOCAL_EXP018`–`LOCAL_EXP023` names under `/home/ray1001/result/`.
- The Git work ran on `local/probe-sweep-20260710-desktop4070ti`, based on the verified GitHub default commit before the active VESSL continuation.
- `train.py`, `recon_eval.py`, model code, data code, losses, transforms, and official metrics were not modified.
- Checkpoints and reconstruction H5 files remain outside Git.
- The branch must not be merged into the VESSL/default branch until the active VESSL training handoff is complete.
