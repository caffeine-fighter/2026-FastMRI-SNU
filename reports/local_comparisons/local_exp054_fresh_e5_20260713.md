# LOCAL EXP054 fresh five-epoch capacity follow-up

_Status: completed and independently re-evaluated on 2026-07-13. This is RTX 4070 Ti SUPER exploratory evidence, not an official VESSL result, candidate, or launch authority._

## Question

Does the seed-robust one-epoch c8/ch12/s8 capacity signal remain positive in a fresh five-epoch comparison against the matched seed-430 c4/ch16/s8 five-epoch baseline?

## Design

| Field | Value |
|---|---|
| Experiment | `LOCAL_EXP054_varnet_c8_ch12_s8_e5_seed430` |
| Candidate | c8/ch12/s8, 5 epochs, seed 430 |
| Matched baseline | `LOCAL_EXP032_varnet_c4_ch16_s8_e5` |
| Batch / learning rate | 1 / 0.001 |
| Training mode | fresh; no resume |
| Evaluator | immutable strict LOCAL evaluator; equal mean of acc4 and acc8 |
| Required coverage | 30 volumes, 791 slices, 161 boxes, unknown=0, skipped=[] |

## Independently reproduced metrics

| Metric | EXP054 | Matched EXP032 e5 | Delta vs EXP032 | Current EXP041 e15 best | Delta vs EXP041 |
|---|---:|---:|---:|---:|---:|
| Equal-acc full SSIM | 0.899931753582 | 0.898141374042 | +0.001790379540 | 0.903443785095 | -0.003512031513 |
| Equal-acc bbox SSIM | 0.917462469968 | 0.913509984254 | +0.003952485714 | 0.923900318820 | -0.006437848852 |
| Equal-acc quality | 0.908697111775 | 0.905825679148 | **+0.002871432627** | 0.913672051958 | **-0.004974940182** |
| Acc4 quality | 0.924097966829 | 0.921926942182 | +0.002171024647 | 0.927865588874 | -0.003767622046 |
| Acc8 quality | 0.893296256722 | 0.889724416115 | +0.003571840607 | 0.899478515041 | -0.006182258319 |

The independent evaluator rerun reproduced the published rows exactly with 30 volumes, 791 slices, 161 bounding-box annotations, no unknown inputs, and `skipped=[]`.

## Training and publication integrity

- Started: 2026-07-13 16:57:02 KST.
- Completed: 2026-07-13 18:46:34 KST.
- End-to-end elapsed time: 1 h 50 m 1.095 s; this is LOCAL diagnostic timing, not official inference timing.
- Epoch-5 checkpoint and best checkpoint are byte-identical: `ab73a20331e01d31f04e9e6d12786a99ddcf65d89ed41b5ef909023bbcff0cf7`.
- The checkpoint contains 224 finite model tensors / 9,212,090 elements and 672 finite optimizer tensors / 18,424,404 elements.
- All five validation epochs improved. The history stores the 30-volume aggregate `3.275654299384549`; its normalized mean is `0.10918847664615164`, matching the checkpoint's best validation loss.
- Exactly 30 reconstructions were evaluated and published.
- The immutable 40-entry published-tree manifest is `360e1e52f95edb97f16a9e8ababf6099284645a50ac3ce245a5fb7dec3ecdbd3`.
- A launch-evidence sample observed 13,062 MiB VRAM and 77–94% GPU utilization. This is LOCAL training telemetry, not an exhaustive peak-memory audit and not official inference memory.

## Verification chain

| Evidence | SHA-256 |
|---|---|
| Frozen runner | `6330d3d04a8ded6aeb4787e7984aa9cb2e1dfe4d097df5bde5911099e0d2acc0` |
| Independent launch review | `c1accfb84657803e65001ea05ef709d19af0a4b48a2fed0003d8830142ae5ccf` |
| One-shot launch approval | `fba74c50e2f1cb543ccba242abfe0825c1092a7f952eacb06087d9a1e9825ec8` |
| Immutable terminal | `45608e526145eea79e26578eff28cc6a4c28fd10685afe9816c25822d5d8d194` |
| Independent post-run audit | `564405fbf1fcfd08386f3547c13220f5cf66dccaee8df7f2c7965eb49fbdf8c4` |
| Launch evidence | `c62ffe1f798211abe428f25f879aed38246bfe7baf8a921de7d823f974da6404` |

The post-run audit reloaded and scanned the checkpoint, recomputed the complete published-tree manifest, reran the strict evaluator from the immutable reconstruction cohort, and independently recomputed every formula and baseline delta.

## Decision

EXP054 is a meaningful matched-five-epoch architecture-capacity signal: quality improves by `+0.002871432627` over EXP032 and every protected component is positive. It is **not a new overall LOCAL best**: quality remains `-0.004974940182` below the retained EXP041 e15 training leader and farther below the documented EXP031 LOCAL promotion reference.

The result supports reviewing a duration-matched c8/ch12 direction, but it does not authorize one. The observed approximately 13 GiB LOCAL training footprint also requires a separate training-versus-inference memory audit before treating this architecture as compatible with the official 8192 MiB constraint.

`candidate_eligible=false`; `official_followup_authorized=false`; `automatic_longer_followup_authorized=false`; `further_launches_authorized=false`.
