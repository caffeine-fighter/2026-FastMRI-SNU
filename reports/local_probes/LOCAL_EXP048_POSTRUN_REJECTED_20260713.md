# LOCAL EXP048 VESSL handoff — post-run review rejected

This is a sanitized metadata handoff. It contains no checkpoint, H5 reconstruction, target data, credential, or raw result directory.

## Final authority

- Status: `POST_RUN_REVIEW_REJECTED_LAUNCH_PACKAGE`
- Method gate: `VOID_DUE_TO_REVIEW_FAILURE`
- Metrics: non-authoritative post-run diagnostic only
- Candidate eligible: `false`
- Official follow-up authorized: `false`
- Promotion/submission authorized: `false`
- Exact-byte rerun authorized: `false`

## Observed metrics — context only

- Quality: `0.913996916453675`
- SSIM full: `0.903497902088761`
- SSIM bbox: `0.924495930818589`
- Delta vs EXP041: `+0.000324864496057`
- Delta vs EXP047: `+0.000023274081886`
- Delta vs LOCAL EXP031: `-0.001416828187601`
- Coverage: 30 volumes / 791 slices / 161 boxes / zero skips

These values were independently rehashed, but a late exact-byte review found six blocking launch-package defects. They cannot authorize interpolation/SWA work.

## Blocking review findings

1. Frozen test package was not self-contained.
2. Pre-worker namespace setup was outside protected failure handling.
3. Success-looking evidence could precede terminal publication.
4. Post-evaluator provenance closure was absent.
5. Lock and failure-path race closure was incomplete.
6. Permanent false authority fields were absent on non-success paths.

The prior approval was revoked and actively replaced at the runner-consumed path by a tested `passed=false` / `REJECT` denial marker.

## VESSL action

- Do **not** promote, officially evaluate, or rerun EXP048.
- Keep EXP031 protected.
- Preserve and hash EXP032 `.checkpoint-generation-*-model.pt` files before cleanup.
- Use a separately reviewed score-faithful handoff for any EXP032 epoch sweep.

Machine-readable record: `LOCAL_EXP048_POSTRUN_REJECTED_20260713.json`

JSON SHA-256: `184fe8e7b7c7c7d0045abec26be1d31d6069a295e8d509494770aade360482d9`
