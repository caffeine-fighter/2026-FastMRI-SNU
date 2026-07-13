# Current state

The root [`README.md`](../README.md) is the compact status dashboard. This page records candidate state and remaining decisions.

## Candidate status

| Role | Experiment | Status |
|---|---|---|
| Finalized 30-run fallback | `EXP030_varnet_c4_ch12_s8_e20` | Official score and timing cohort complete |
| One-shot official leader | `EXP031_varnet_c4_ch12_s8_e30` | Training, validation, and first approved official run complete |
| Recent completed quality experiment | `EXP032_varnet_c6_ch12_s8_e30` | VESSL run complete; metrics pending a Git-tracked VESSL handoff |
| Active VESSL experiment | `EXP033R` | Running on VESSL; exact configuration, progress, and metrics pending tracked evidence |

## Official results

| Candidate | Evidence | SSIM_full | SSIM_bbox | Quality | Time | Total score |
|---|---|---:|---:|---:|---:|---:|
| EXP030 | 30-run minimum | 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | 0.9152513541666667 |
| EXP031 | one official run | **0.9191** | **0.9114** | **0.91525** | **173.1 ms/slice** | **0.9162015104166666** |

EXP031 improved official SSIM_full by `+0.0013`, SSIM_bbox by `+0.0006`, quality by `+0.00095`, and one-shot total score by `+0.0009501041666666` relative to EXP030's one-shot result. It leads EXP030's finalized 30-run score by `+0.00095015625`.

Evidence: [`../reports/phase2/EXP031_official_20260711_014023/score.json`](../reports/phase2/EXP031_official_20260711_014023/score.json).

## EXP031 provenance

- best training epoch: 27
- best validation loss: `3.1818922822357556`
- best checkpoint SHA-256: `3e68d94922f68d9a536e4bdbe7802785f8b43792524ddac252dbbe8d5c11d31f`
- official run: `EXP031_official_20260711_014023`
- official evaluator exit code: 0
- official error-pattern matches: 0

## Active optimization gate

EXP031 remains the protected one-shot official leader while the 40-day optimization program runs. EXP032 is complete on VESSL, but its final validation and official metrics are not yet present in Git-tracked evidence. EXP033R is running on VESSL. Neither run changes candidate authority until its source-backed validation is recorded and passes the promotion gate.

Local promotion uses the fixed evaluator's equal-acceleration semantics, not the historical pooled `overall` row:

| EXP031 local reference | SSIM_full | SSIM_bbox | Quality |
|---|---:|---:|---:|
| Leaderboard-faithful equal-acc | 0.904053714106 | 0.926773775177 | **0.915413744641** |
| Pooled diagnostic only | 0.904500772769 | 0.930380483592 | 0.917440628180 |

The validation set has 407 acc4 versus 384 acc8 slices and 107 versus 54 boxes, so pooled aggregation overweights acc4. Leaderboard-faithful strict evaluation, retained validation epochs, epoch sweep, averaging, and candidate materialization are published in commit `24bf00677c64e7a0bd84f95d68ded8beb7925b12`.

The final 30-run timing cohort is intentionally deferred until the model is frozen near the deadline. Running it now would measure a candidate that may soon be replaced.

Full schedule: [`score_optimization_40_day_roadmap.md`](score_optimization_40_day_roadmap.md).

## Desktop LOCAL candidate evidence

All rows below are RTX 4070 Ti SUPER exploratory evidence evaluated with leaderboard-equal-acceleration aggregation. They are neither official scores nor VESSL candidate authority.

| Candidate | Result | Decision |
|---|---:|---|
| `LOCAL_EXP039` sensitivity-width seed replication | quality 0.882647730465; delta -0.001995170919 | reject c4/ch12/s12 replication |
| `LOCAL_EXP040` e5/e10 checkpoint average | quality 0.907946926286; delta -0.001647492152 versus e10 | reject this average |
| `LOCAL_EXP041` e10 -> e15 continuation at LR 3e-4 | quality 0.913672051958; delta +0.004077633519 versus re-evaluated EXP035 | retained same-basin training leader |
| `LOCAL_EXP042` e15 -> e18 attempt at LR 1e-4 | no score or committed epoch-16 checkpoint | technical failure during retained-output publication |
| `LOCAL_EXP043` plumbing-only retry of EXP042 | orphan e16 diagnostic quality 0.913476635177; delta -0.000195416780 versus EXP041 | diagnostic only; reject LR 1e-4 continuation and do not rerun |
| `LOCAL_EXP044` matched standard-SSIM e15 -> e16 at LR 3e-4 | quality 0.913223060119; delta -0.000448991839 versus EXP041 | reject further fixed-LR duration escalation |
| `LOCAL_EXP046` matched sparse metric-aligned e15 -> e16 | quality 0.911710246795; delta -0.001512813323 versus EXP044 | reject this objective; every protected component regressed |
| `LOCAL_EXP047` 75/25 image-space blend | quality 0.913973642372; delta +0.000301591492 versus EXP041 | robust LOCAL blend signal, but below EXP031 and two-forward only |
| `LOCAL_EXP048` 75/25 parameter interpolation | quality 0.913996916454; delta +0.000324864496 versus EXP041 | post-run exact-byte review rejected; method gate void; diagnostic only |
| `LOCAL_EXP050` / `LOCAL_EXP052` c8/ch12/s8 one-epoch seed pair | qualities 0.887020276038 / 0.885970484235; matched c4 deltas +0.005563956839 / +0.001327582851 | seed-robust positive; longer follow-up eligible but not authorized |
| `LOCAL_EXP051` / `LOCAL_EXP053` c8/ch18/s8 one-epoch seed pair | qualities 0.881390927133 / 0.883208297423; matched c4 deltas -0.000065392066 / -0.001434603960 | reject combined depth/width ceiling direction |

EXP041 remains the strongest completed same-basin training checkpoint. EXP047 established a statistically supported image-space complementarity signal: 200,000 paired, acceleration-stratified volume-cluster replicates gave a 90% BCa quality-delta interval of `[+0.000188730881, +0.000397414539]`. EXP048 numerically reproduced that signal, but a previously dispatched exact-byte reviewer found six blocking runner/evidence defects; the frozen test failure was independently reproduced. EXP048's approval is revoked, its runner-consumed approval path now contains a tested `passed=false` denial marker, and its method gate is void. Its outputs remain immutable, independently rehashed, non-authoritative diagnostic data only and cannot authorize interpolation/SWA follow-up.

The exact-byte-reviewed EXP050-EXP053 salvage recovered four scientifically complete one-epoch runs without changing artifact bytes. c8/ch12/s8 was positive under both seeds and met the predeclared longer-follow-up eligibility rule; c8/ch18/s8 did not. The c8/ch12 versus c4/ch16 ranking still flips by seed, so this is a supported depth direction rather than an established architecture winner. The operator also reports that four of five prize-winning solutions used nearly the official GTX1080 8192 MB VRAM budget and the fifth reported about 6 GB with gradient checkpointing; this is a planning clue pending source and measurement-context verification.

Details:

- [`../reports/local_comparisons/local_continuation_campaign_20260711.md`](../reports/local_comparisons/local_continuation_campaign_20260711.md)
- [`../reports/local_comparisons/local_capacity_screen_20260713.md`](../reports/local_comparisons/local_capacity_screen_20260713.md)

## Resume/checkpoint infrastructure status

Resume/LR-override and history-prefix support is published in commit `431c69018678c47ae90ecba9c3863a5ef47ab68b`. A CPU-only preflight against the safe EXP031 artifact verified checkpoint schema, optimizer restoration, epoch-28 history, the `3e-4` LR override, retained epoch generations, and the required inexact-resume opt-in. The operator reports that EXP033R is now running on VESSL; its exact launch/configuration evidence is not yet tracked here, so local documentation does not infer its configuration or metrics.

## Remaining actions

1. Monitor EXP033R on VESSL without launching a competing GPU task or automatic official evaluation.
2. Evaluate EXP032, record its checkpoint provenance and leaderboard-faithful validation metrics, and add its evidence to the experiment log/reports.
3. After EXP033R completes, evaluate and record it with the same provenance, coverage, and equal-acceleration validation requirements.
4. Compare EXP032 and EXP033R against the protected EXP031 reference; request a separate approval for official `bash recon_eval.sh` only if a meaningful validation winner exists.
5. Keep EXP041 as the same-basin training leader; treat EXP048 as post-run-review-rejected diagnostic data and do not use it to authorize interpolation/SWA follow-up.
6. Retain c8/ch12/s8 as a LOCAL longer-follow-up-eligible direction, but do not launch it unless EXP032/EXP033R evidence, an 8 GB inference-memory audit, and a separate exact-byte review justify the spend.
7. Prefer a predeclared scheduler from before the plateau, exact mask/ACS A/B, width replication, and acceleration-specialist screens over more same-basin micro-tuning.
8. Freeze the final candidate around August 15, then run its approved 30-run timing cohort.
9. Build, verify, and upload the final package before August 20 using [`final_submission_checklist.md`](final_submission_checklist.md).

## Submission state

- GitHub default branch: `baseline/2026-baby-varnet`
- Existing EXP030 implementation commit: `fbbddf6700cd65b1e2b52c1c6418f48a5eef9b82`
- Existing EXP030 package: `/root/submissions/EXP030_final_fbbddf6.zip`
- Existing package SHA-256: `65b150fb749b772db99e4fde77a636ed58eb19f215e859dbc77cf60ea3aeb18f`
- EXP031 package: not built pending timing cohort
- External organizer upload: pending

## Guardrails

- Do not modify `recon_eval.py`.
- Do not run official evaluation without approval.
- Do not modify mounted `Data`.
- Do not commit data, H5 files, checkpoints, result directories, `.env` files, or credentials.
