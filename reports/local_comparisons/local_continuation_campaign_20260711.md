# LOCAL continuation campaign: EXP039-EXP043

Verified through: 2026-07-12 02:25 KST (2026-07-11 17:25 UTC)

**Scope:** Desktop RTX 4070 Ti SUPER exploratory evidence only. Nothing here is an official VESSL score, timing result, checkpoint promotion, or launch authorization.

## Candidate ledger

| Candidate | Configuration / evidence | SSIM full | SSIM bbox | Equal-acc quality | Classification |
|---|---|---:|---:|---:|---|
| `LOCAL_EXP039` | c4/ch12/s12/e1, seed 431 | 0.882554265162 | 0.882741195767 | 0.882647730465 | rejected seed replication |
| `LOCAL_EXP040` | c4/ch16/s8, 50/50 e5/e10 checkpoint average | 0.900574274114 | 0.915319578458 | 0.907946926286 | rejected average |
| `LOCAL_EXP041` | c4/ch16/s8, exact resume e10 -> e15, LR 3e-4, seed 430 | 0.903443785095 | 0.923900318820 | **0.913672051958** | current completed LOCAL leader |
| `LOCAL_EXP042` | attempted exact resume e15 -> e18, LR 1e-4 | n/a | n/a | n/a | technical failure; no candidate result |
| `LOCAL_EXP043` | plumbing-only EXP042 retry; recovered orphan e16 reconstruction | 0.903816298291 | 0.923136972064 | 0.913476635177 | diagnostic only; rejected direction |

## Decisions

### EXP039: reject sensitivity-width replication

Relative to the matched c4/ch12/s8 seed-431 baseline:

- equal-acc quality: `-0.001995170919`;
- SSIM full: `+0.000770736484`;
- SSIM bbox: `-0.004761078322`;
- acc4 quality: `-0.001388194201`;
- acc8 quality: `-0.002602147637`.

Decision: `DO_NOT_ESCALATE_c4_ch12_s12`.

### EXP040: reject the no-cost checkpoint average

The inference-only 50/50 average of matching epoch-5 and epoch-10 floating tensors scored `-0.001647492152` below the re-evaluated epoch-10 source and regressed every guarded component.

Decision: `DO_NOT_ADVANCE_THIS_AVERAGE`.

### EXP041: retain as the completed LOCAL leader

The c4/ch16/s8 seed-430 state resumed exactly from epoch 10 through epoch 15 at LR 3e-4. Against the authoritative equal-acc re-evaluation of LOCAL_EXP035:

- equal-acc quality: `+0.004077633519`;
- SSIM full: `+0.001805178644`;
- SSIM bbox: `+0.006350088394`;
- acc4 quality: `+0.002188127392`;
- acc8 quality: `+0.005967139646`.

Integrity: 30 volumes, 791 slices, 161 bbox annotations, `unknown=0`, and `skipped=[]`.

Decision: EXP041 remains the current completed LOCAL continuation leader. It has no official authority.

### EXP042: technical failure, no candidate result

EXP042 attempted the same basin from epoch 15 through epoch 18 at LR 1e-4. It completed the epoch-16 training batches but failed before retained reconstruction publication because the worker's relative `../result` traversed a symlink that the fail-closed retained publisher correctly rejected.

No epoch-16 checkpoint, validation-history update, retained reconstruction, or scientific terminal was committed. EXP042 therefore has no score and is not evidence for or against the LR 1e-4 hypothesis.

### EXP043: failed retry with a diagnostic-only orphan reconstruction

EXP043 changed execution plumbing only and retried the same scientific protocol. It completed epoch-16 training and 30-volume validation reconstruction, then failed before retained publication because absolute staging and relative destination parents failed the same-parent guard. No epoch-16 checkpoint or successful training terminal exists.

The complete unpublished reconstruction was sealed and evaluated on CPU with the fixed equal-acc evaluator:

| Metric | EXP043 orphan e16 | EXP041 | Delta |
|---|---:|---:|---:|
| SSIM full | 0.903816298291 | 0.903443785095 | +0.000372513196 |
| SSIM bbox | 0.923136972064 | 0.923900318820 | -0.000763346757 |
| Equal-acc quality | 0.913476635177 | 0.913672051958 | **-0.000195416780** |
| acc4 quality | 0.928090716379 | 0.927865588874 | +0.000225127504 |
| acc8 quality | 0.898862553976 | 0.899478515041 | -0.000615961065 |

Limitations are binding:

- `training_terminal_success=false`;
- `checkpoint_available=false`;
- `candidate_or_promotion_evidence=false`;
- a delayed adversarial review marked the launch package failed review;
- the reconstruction is diagnostic only and cannot authorize promotion.

Decision: `STOP_LOW_LR_1E-4_CONTINUATION_DIRECTION_AND_DO_NOT_SPEND_A_THIRD_FULL_RERUN`. No `LOCAL_EXP044` exists.

## Evidence integrity

Authoritative LOCAL evidence remains outside Git under ignored result roots:

- EXP039 terminal: `LOCAL_SENS_WIDTH_PROTOCOL_V2_20260711_DESKTOP4070TI/terminal/EXP039_terminal.json`
- EXP040 terminal: `LOCAL_CHECKPOINT_AVERAGING_PROTOCOL_V2_20260711_DESKTOP4070TI/terminal/EXP040_terminal.json`
- EXP041 terminal: `LOCAL_EXP041_POSTTRAIN_RECON_EVAL_PROTOCOL_V1_20260711_DESKTOP4070TI/terminal/EXP041_terminal.json`
- EXP042 failure: `LOCAL_LOW_LR_CONTINUATION_PROTOCOL_V4_20260711_DESKTOP4070TI/failure/EXP042_training_failure.json`
- EXP043 failure: `LOCAL_LOW_LR_CONTINUATION_PROTOCOL_V5_20260712_DESKTOP4070TI/failure/EXP043_training_failure.json`
- EXP043 diagnostic terminal: `LOCAL_EXP043_ORPHAN_E16_DIAGNOSTIC_20260712_DESKTOP4070TI/terminal.json`
  - SHA-256: `c89700257ee2373d4d01ff9aafa12685de45042ff300e76f455715928a421a7e`
  - reconstruction tree SHA-256: `844cc53c0b9345ddd95e2f5d4d66bdab0c0bc9a780b59bb98cbfaaea047d5f5b`

Raw checkpoints and H5 reconstructions remain ignored and are not committed.

## Guardrails

- Keep EXP041 as the current completed LOCAL leader.
- Do not treat EXP043's orphan diagnostic as a successful candidate or checkpoint.
- Do not repeat the rejected LR 1e-4 direction.
- Do not overlap desktop CUDA work with an active controlled workload.
- Do not promote LOCAL checkpoints or desktop timing to official status.
- Any VESSL follow-up requires separate source-backed review and explicit approval.
