# Current state

The root [`README.md`](../README.md) is the compact status dashboard. This page records candidate state and remaining decisions.

## Candidate status

| Role | Experiment | Status |
|---|---|---|
| Finalized 30-run fallback | `EXP030_varnet_c4_ch12_s8_e20` | Official score and timing cohort complete |
| One-shot official leader | `EXP031_varnet_c4_ch12_s8_e30` | Training, validation, and first approved official run complete |
| Active quality experiment | `EXP032_varnet_c6_ch12_s8_e30` | VESSL training; cascades 4 -> 6, all other main variables fixed |

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

EXP031 remains the protected one-shot official leader while the 40-day optimization program runs. EXP032 is the first architecture-capacity test. Its final validation result must beat EXP031 before any official evaluation is considered.

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
| `LOCAL_EXP048` 75/25 parameter interpolation | quality **0.913996916454**; delta +0.000324864496 versus EXP041 | method gate passed; supports future SWA review only; not candidate eligible |

EXP041 remains the strongest completed same-basin training checkpoint. EXP047 established a statistically supported image-space complementarity signal: 200,000 paired, acceleration-stratified volume-cluster replicates gave a 90% BCa quality-delta interval of `[+0.000188730881, +0.000397414539]`. EXP048 reproduced and slightly exceeded that gain in one interpolated model, with positive aggregate full and bbox deltas and exact 30/791/161 coverage. Both remain below EXP031's `0.915413744641` LOCAL reference; neither is a promotion candidate or official authority.

Details: [`../reports/local_comparisons/local_continuation_campaign_20260711.md`](../reports/local_comparisons/local_continuation_campaign_20260711.md).

## Resume/checkpoint infrastructure status

Resume/LR-override and history-prefix support is published in commit `431c69018678c47ae90ecba9c3863a5ef47ab68b`. A CPU-only preflight against the safe EXP031 artifact verified checkpoint schema, optimizer restoration, epoch-28 history, the `3e-4` LR override, retained epoch generations, and the required inexact-resume opt-in. EXP033 remains launch-gated on the handoff worker proving EXP032 and its authorized evaluation have exited.

## Remaining actions

1. Complete EXP032 without launching official evaluation automatically; preserve and hash its immutable per-epoch checkpoint generations before cleanup.
2. Reconstruct and score-faithfully sweep the available EXP032 late epochs before deciding whether c6 deserves any follow-up.
3. Keep EXP041 as the same-basin training leader and EXP048 as method evidence only; do not repeat fixed-LR late continuation or the rejected sparse metric-aligned objective.
4. Prefer a predeclared scheduler from before the plateau, exact mask/ACS A/B, width replication, and acceleration-specialist screens over more same-basin micro-tuning.
5. Use separately approved one-shot official runs only for meaningful validation winners.
6. Freeze the final candidate around August 15, then run its approved 30-run timing cohort.
7. Build, verify, and upload the final package before August 20 using [`final_submission_checklist.md`](final_submission_checklist.md).

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
