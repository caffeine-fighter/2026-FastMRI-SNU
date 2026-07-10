# LOCAL Probe Sweep — Round 2 (2026-07-11)

Five serial one-epoch probes ran on desktop `Cafe-Pastelica` with an RTX 4070 Ti SUPER. All completed successfully with 30 volumes, 791 slices, 161 bbox annotations, and `skipped=[]`.

These are exploratory LOCAL results. No checkpoint or timing value in this report is official, and no official `recon_eval.sh` was run.

## Results

| Rank | Experiment | Config | SSIM_full | SSIM_bbox | Quality | Acc4 quality | Acc8 quality | Val loss |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `LOCAL_EXP028_varnet_c7_ch12_s8_e1` | c7/ch12/s8/e1 | 0.8843235364 | 0.8948626322 | 0.8895930843 | 0.9036611248 | 0.8696873903 | 3.6353778787 |
| 2 | `LOCAL_EXP027_varnet_c6_ch14_s8_e1` | c6/ch14/s8/e1 | 0.8841284133 | 0.8931268794 | 0.8886276464 | 0.9037073026 | 0.8672733414 | 3.6680497184 |
| 3 | `LOCAL_EXP025_varnet_c4_ch12_s10_e1` | c4/ch12/s10/e1 | 0.8834250094 | 0.8925935345 | 0.8880092719 | 0.9033175714 | 0.8662361600 | 3.6461109876 |
| 4 | `LOCAL_EXP024_varnet_c4_ch14_s8_e1` | c4/ch14/s8/e1 | 0.8795853271 | 0.8853956554 | 0.8824904913 | 0.8982502596 | 0.8604053287 | 3.7197679815 |
| 5 | `LOCAL_EXP026_varnet_c4_ch16_s10_e1` | c4/ch16/s10/e1 | 0.8794680239 | 0.8835868291 | 0.8815274265 | 0.8970262402 | 0.8595336448 | 3.7231682369 |

## Controlled comparisons

- `LOCAL_EXP028 c7/ch12/s8` led Round 2 but remained `-0.0033655580` below the overall one-epoch leader, `LOCAL_EXP018 c4/ch16/s8`.
- `LOCAL_EXP027 c6/ch14/s8` improved `+0.0029089513` over `LOCAL_EXP014 c6/ch12/s8`, showing that width helped the c6 family at one epoch.
- `LOCAL_EXP025 c4/ch12/s10` improved `+0.0030723957` over `LOCAL_EXP013 c4/ch12/s8`, but remained `-0.0016406143` below `LOCAL_EXP019 c4/ch12/s12`.
- `LOCAL_EXP024 c4/ch14/s8` was `-0.0024463850` below c4/ch12/s8; the width response did not interpolate smoothly.
- `LOCAL_EXP026 c4/ch16/s10` was `-0.0114312159` below c4/ch16/s8. Increasing sensitivity width damaged the strong ch16 result at one epoch.

## Decision

`c4/ch16/s8` remains the selected LOCAL architecture for matched longer-run and seed-robustness testing. The active adaptive campaign compares it with c4/ch12/s8 at five epochs, repeats both at seed 431, and trains c4/ch16/s8 for ten epochs.

The selected architecture may be proposed for a future VESSL experiment only if it passes the fail-closed criteria in `docs/exp031_post_training_handoff.md`. LOCAL checkpoints remain in `/home/ray1001/result/` and are never candidates for official submission.
