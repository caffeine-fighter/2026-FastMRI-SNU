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

The validation set has 407 acc4 versus 384 acc8 slices and 107 versus 54 boxes, so pooled aggregation overweights acc4. `scripts/evaluate_val.py` now has a test-first `leaderboard_equal_acc` row under independent review.

The final 30-run timing cohort is intentionally deferred until the model is frozen near the deadline. Running it now would measure a candidate that may soon be replaced.

Full schedule: [`score_optimization_40_day_roadmap.md`](score_optimization_40_day_roadmap.md).

## Resume/checkpoint infrastructure status

The local resume/LR-override implementation now passes 40 focused unit tests after test-first remediation of race, interruption-consistency, checkpoint-schema, CLI, and CUDA RNG-topology findings. It remains uncommitted: a fresh independent review must clear the revised implementation and its two documented cross-file/concurrent-publication concerns before it can gate EXP033.

## Remaining actions

1. Complete and locally evaluate EXP032 without launching official evaluation automatically.
2. Run the queued EXP033 five-epoch continuation from EXP031 best at LR 3e-4 after EXP032, avoiding GPU contention.
3. Test score-aligned foreground/bbox loss, supported scheduler/cascade follow-ups, and no-cost checkpoint averaging through local promotion gates.
4. Use separately approved one-shot official runs only for meaningful validation winners.
5. Freeze the final candidate around August 15, then run its approved 30-run timing cohort.
6. Build, verify, and upload the final package before August 20 using [`final_submission_checklist.md`](final_submission_checklist.md).

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
