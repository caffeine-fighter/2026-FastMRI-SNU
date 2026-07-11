# Current state

The root [`README.md`](../README.md) is the live VESSL dashboard. This page records the stable official state, consolidated LOCAL evidence, and remaining decision gates.

_Last consolidated: 2026-07-11 KST after EXP031 training and the desktop LOCAL campaign completed._

## Candidate status

| Role | Experiment | Status |
|---|---|---|
| Official candidate | `EXP030_varnet_c4_ch12_s8_e20` | selected and officially evaluated |
| Follow-up | `EXP031_varnet_c4_ch12_s8_e30` | training complete; final file-backed handoff pending |
| LOCAL architecture candidate | c4/ch16/s8 | rejected by predefined seed-robustness gate |

## Official EXP030 result

| SSIM_full | SSIM_bbox | Quality | Min time | Total score |
|---:|---:|---:|---:|---:|
| 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | **0.9152513541666667** |

The official 30-repeat evaluation is complete. Thirty repeats are timing repetitions of the same checkpoint, not training epochs. The external organizer upload remains pending.

## EXP031 training-complete telemetry

VESSL status commit `bc13bcce550136cd96781dbecb544a0cb10eaf26` reports:

- completion by 2026-07-11 08:59 KST;
- best validation loss `3.1818922822357556` at epoch 27;
- final validation snapshot: full `0.904501`, bbox `0.930380`, quality `0.917441`;
- no matched training errors and checkpoints reported present.

These values are training/status telemetry only. The repository still lacks the final EXP031 experiment-log row, checkpoint identity, and `reports/phase2/EXP031_validation_handoff.json`. EXP030 therefore remains authoritative, and no official EXP031 evaluation starts automatically.

## Desktop LOCAL campaign

All LOCAL evidence is exploratory desktop evidence only. It is not an official VESSL score or timing result.

Seventeen one-epoch probes and five adaptive runs completed without failed runs or skipped validation files. The adaptive comparison was:

| Probe | Config | Seed | SSIM_full | SSIM_bbox | quality_score | val_loss |
|---|---|---:|---:|---:|---:|---:|
| `LOCAL_EXP029` | c4/ch12/s8/e5 | 430 | 0.8957743251 | 0.9116992114 | 0.9037367682 | 3.3905022666 |
| `LOCAL_EXP032` | c4/ch16/s8/e5 | 430 | 0.8986326562 | 0.9185488328 | 0.9085907445 | 3.3032180536 |
| `LOCAL_EXP033` | c4/ch12/s8/e1 | 431 | 0.8823435134 | 0.8932403356 | 0.8877919245 | 3.6904079145 |
| `LOCAL_EXP034` | c4/ch16/s8/e1 | 431 | 0.8808135834 | 0.8923592767 | 0.8865864301 | 3.6965833257 |
| `LOCAL_EXP035` | c4/ch16/s8/e10 | 430 | 0.9021009122 | 0.9229051363 | 0.9125030243 | 3.2429721451 |

Gate outcome:

- matched seed-430 e5 quality gain: `+0.0048539763` — pass;
- candidate e10 minus e5: `+0.0039122798` — pass;
- seed-431 quality delta: `-0.0012054944` — fail;
- seed-431 full-image delta: `-0.0015299300`, below the allowed `-0.001` floor — fail.

**Decision: do not promote c4/ch16/s8.** See [`../reports/local_comparisons/local_probe_adaptive_followup_20260711_final.md`](../reports/local_comparisons/local_probe_adaptive_followup_20260711_final.md).

## Consolidated Git state

- Default branch: `baseline/2026-baby-varnet`
- VESSL operational branch: `phase2/eval-wrapper-vessl`
- EXP030 implementation commit: `fbbddf6700cd65b1e2b52c1c6418f48a5eef9b82`
- Phase 2 score-parser hardening: merged through PR #18
- LOCAL reports: consolidated into the default and VESSL branches; temporary LOCAL branches are retired after remote verification

## Next actions

1. Require the final EXP031 experiment-log row and `EXP031_validation_handoff.json`, including checkpoint SHA-256, training provenance, best epoch, metrics, subgroup counts, and skipped list.
2. Run `python scripts/build_exp031_decision_report.py --check`; generate the combined report only when the official sources are present and valid.
3. Keep EXP030 authoritative unless a complete official EXP031 handoff passes validation and replacement is approved.
4. Keep c4/ch16/s8 rejected unless separately approved evidence resolves the seed-robustness failure.
5. Complete the external organizer upload using [`final_submission_checklist.md`](final_submission_checklist.md).

## Submission state

- Verified EXP030 package: `/root/submissions/EXP030_final_fbbddf6.zip`
- Package SHA-256: `65b150fb749b772db99e4fde77a636ed58eb19f215e859dbc77cf60ea3aeb18f`
- External organizer upload: pending

## Guardrails

- Do not modify `recon_eval.py` or official metric implementations.
- Do not run official evaluation without explicit approval.
- Treat mounted `Data` directories as read-only.
- Never commit data, H5 files, checkpoints, result directories, `.env` files, or credentials.
- Keep LOCAL evidence explicitly exploratory and unofficial.
