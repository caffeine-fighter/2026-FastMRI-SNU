# Current state

The root [`README.md`](../README.md) is the live VESSL dashboard. This page records the stable state and decision gates.

## Candidate status

| Role | Experiment | Status |
|---|---|---|
| Official candidate | `EXP030_varnet_c4_ch12_s8_e20` | Selected and evaluated |
| Active follow-up | `EXP031_varnet_c4_ch12_s8_e30` | Continuing EXP030 from epoch 20 through 29 |

EXP030 official result:

| SSIM_full | SSIM_bbox | Quality | Min time | Total score |
|---:|---:|---:|---:|---:|
| 0.9178 | 0.9108 | 0.9143 | 173.4 ms/slice | **0.9152513541666667** |

EXP031 interim validation at its current best, epoch 23:

| SSIM_full | SSIM_bbox | Quality | Delta vs EXP030 validation |
|---:|---:|---:|---:|
| 0.904252391699 | 0.928966502966 | 0.916609447333 | +0.001930361384 |

The EXP031 values are validation-only. EXP030 stays official until EXP031 finishes, final validation confirms the gain, and official evaluation is approved.

## What is running

- EXP031 training on VESSL
- Read-only training monitor
- README progress publisher
- Validation-only finalizer, gated on a clean epoch-29 completion

The finalizer writes local validation artifacts under `../result/EXP031_varnet_c4_ch12_s8_e30/metrics_final`. It does not run official `recon_eval`.

## Next actions

1. Let EXP031 finish.
2. Review final `SSIM_full`, `SSIM_bbox`, quality, best epoch, and checkpoint hash.
3. Compare the final result with EXP030.
4. Keep EXP030 unless EXP031 remains clearly better.
5. Run official evaluation only after explicit approval.
6. Complete the external organizer upload using [`final_submission_checklist.md`](final_submission_checklist.md).

## Submission state

- GitHub default branch: `baseline/2026-baby-varnet`
- EXP030 implementation commit: `fbbddf6700cd65b1e2b52c1c6418f48a5eef9b82`
- Verified package: `/root/submissions/EXP030_final_fbbddf6.zip`
- Package SHA-256: `65b150fb749b772db99e4fde77a636ed58eb19f215e859dbc77cf60ea3aeb18f`
- External organizer upload: pending

## Guardrails

- Do not modify `recon_eval.py`.
- Do not run official evaluation during training.
- Do not modify mounted `Data`.
- Do not commit data, H5 files, checkpoints, result directories, `.env` files, or credentials.
