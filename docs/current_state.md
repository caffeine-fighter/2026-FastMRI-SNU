# Current state

The root [`README.md`](../README.md) is the compact status dashboard. This page records candidate state and remaining decisions.

## Candidate status

| Role | Experiment | Status |
|---|---|---|
| Finalized 30-run candidate | `EXP030_varnet_c4_ch12_s8_e20` | Official score and timing cohort complete |
| One-shot official leader | `EXP031_varnet_c4_ch12_s8_e30` | Training, validation, and first approved official run complete |

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

## Decision gate

EXP031 is the one-shot official leader and the recommended replacement candidate. Final replacement and packaging remain gated on a separately approved 30-run EXP031 timing cohort because the final timing rule uses the minimum valid time from at least 30 runs.

Do not start that repeat cohort automatically.

## Remaining actions

1. Approve or reject the 30-run EXP031 timing cohort.
2. If approved, verify 30/30 successful runs and select the minimum valid timing.
3. Recompute the final EXP031 total score.
4. Replace and package EXP030 only if EXP031 remains ahead.
5. Complete the external organizer upload using [`final_submission_checklist.md`](final_submission_checklist.md).

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
