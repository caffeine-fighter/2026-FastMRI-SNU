# LOCAL c8 capacity screen: EXP050-EXP053

Verified through: 2026-07-13 15:13:59 KST (2026-07-13 06:13:59 UTC)

**Scope:** Desktop RTX 4070 Ti SUPER exploratory evidence only. These runs are not official VESSL scores, timing evidence, candidates, or authorization for another launch.

## Capacity-budget clue

The operator reports that four of five prize-winning solutions used nearly the full official GTX1080 8192 MB VRAM budget, while the fifth reported about a 6 GB footprint with gradient checkpointing. This observation is not independently source-verified in this repository, and the measurement context is not yet known. In particular, gradient checkpointing normally reduces training activation memory and does not by itself prove lower inference memory.

Treat this as a capacity-planning hypothesis: measure peak inference memory and runtime on the largest legal input before promoting a larger architecture, leave safety margin below the official 8 GB ceiling, and verify the finalist on VESSL-compatible hardware.

## Protocol and provenance

- Campaign purpose: test whether larger cascade capacity gives a seed-robust one-epoch signal before spending a longer trajectory budget.
- Immutable training source commit: `11f44825c7ae378a00119f877481f690ec4cebfa`.
- Immutable training source tree: `b5f9cdcaa4d28110623c60e899a9ced6b320a45d`.
- Evaluator: strict leaderboard-equal-acceleration evaluator frozen under the EXP048 evaluator package.
- Coverage per run: 30 volumes, 791 slices, 161 bbox annotations, zero unknown scope, and `skipped=[]`.
- Baselines: matched c4/ch12/s8 one-epoch runs for seed 430 and seed 431.
- No automatic longer follow-up or official evaluation was authorized.

The first campaign generation failed before CUDA because its pinned `fastmri` package path was absent. RETRY1 then completed all four training workers and strict evaluators with exit code 0, but each final publication failed after evaluation because the reviewed runner froze the experiment root to mode `0500` before a cross-parent no-clobber rename. A separately exact-byte-reviewed salvage publisher matched all 194 source entries, changed no scientific artifact bytes, and published the preserved outputs. Its independent review returned `APPROVE`, and its terminal status is `recovered_publication_done`.

## Equal-acceleration results

| Run | Configuration | Seed | SSIM full | SSIM bbox | Quality | Delta vs matched c4/ch12 | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `LOCAL_EXP050` | c8/ch12/s8/e1 | 430 | 0.884052150231 | 0.889988401845 | 0.887020276038 | **+0.005563956839** | positive, meaningful; all component floors pass |
| `LOCAL_EXP051` | c8/ch18/s8/e1 | 430 | 0.880917293885 | 0.881864560382 | 0.881390927133 | -0.000065392066 | reject; not positive or meaningful |
| `LOCAL_EXP052` | c8/ch12/s8/e1 | 431 | 0.883414234607 | 0.888526733863 | 0.885970484235 | **+0.001327582851** | positive; all component floors pass |
| `LOCAL_EXP053` | c8/ch18/s8/e1 | 431 | 0.880025646424 | 0.886390948422 | 0.883208297423 | -0.001434603960 | reject; component floors fail |

For EXP050, every protected component improved over the matched seed-430 c4/ch12/s8 baseline:

- SSIM full: `+0.003847604147`;
- SSIM bbox: `+0.007280309531`;
- acc4 quality: `+0.004765260961`;
- acc8 quality: `+0.006362652717`.

EXP052 independently kept the c8/ch12/s8 direction positive under seed 431. The recovered campaign summary therefore marks c8/ch12/s8 as `seed_robust_positive=true` and `longer_followup_eligible=true`, with minimum per-seed quality delta `+0.001327582851`. Automatic follow-up remains false.

## Controlled-neighbor interpretation

Against the earlier c7/ch12/s8 seed-430 probe, EXP050 improved:

- quality: `+0.000346018490`;
- SSIM full: `+0.000246592623`;
- SSIM bbox: `+0.000445444357`.

The comparison with c4/ch16/s8 is seed-sensitive:

- seed 430: c8/ch12 is `-0.002552799947` quality below c4/ch16;
- seed 431: c8/ch12 is `+0.003206673656` quality above c4/ch16.

Therefore c8/ch12/s8 is a supported depth direction, not an established overall architecture winner. The c8/ch18/s8 ceiling screen failed under both seeds, so depth and aggressive width should not be combined without new evidence.

## Decision

1. Retain c8/ch12/s8 as eligible for a separately reviewed longer comparison.
2. Do not authorize or automatically launch that comparison from this report.
3. Compare a longer c8/ch12/s8 trajectory against a matched c4/ch16/s8 control if and only if VESSL EXP032/EXP033R evidence and resource priorities still justify it.
4. Before any VESSL escalation, measure largest-input inference VRAM and runtime against the official 8 GB hardware constraint.
5. Keep every EXP050-EXP053 checkpoint, reconstruction, and desktop timing result LOCAL-only.

## Evidence

- Recovered terminal: `/home/ray1001/result/AUTONOMOUS_SCORE_LOOP_20260711/LOCAL_CAPACITY_SCREEN_RETRY1_SALVAGE_V1/terminal.json`
- Salvage review: `/home/ray1001/result/AUTONOMOUS_SCORE_LOOP_20260711/LOCAL_CAPACITY_SCREEN_RETRY1_SALVAGE_review_f6b2458e/review.json`
- Campaign design: `/home/ray1001/result/AUTONOMOUS_SCORE_LOOP_20260711/LOCAL_CAPACITY_SCREEN_RETRY1_exact_review_snapshot_1bb703db/LOCAL_CAPACITY_SCREEN_V1_design.json`
- Published LOCAL results: `/home/ray1001/result/LOCAL_EXP050_varnet_c8_ch12_s8_e1_seed430` through `LOCAL_EXP053_varnet_c8_ch18_s8_e1_seed431`

Raw checkpoints and reconstruction H5 files remain outside Git.
