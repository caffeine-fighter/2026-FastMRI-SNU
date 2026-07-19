# Prize-first R4.1 status

_Snapshot: 2026-07-19 14:47 KST. This is an operational research snapshot, not launch authority._

## Protected result

`EXP035_varnet_c8_ch12_s8_e30` epoch 30 remains the protected one-shot official leader:

- SSIM full: `0.9234`
- SSIM bbox: `0.9177`
- quality: `0.92055`
- official time: `250.7 ms/slice`
- total: `0.92146109375`

The completed lower-LR continuation did not pass its promotion threshold. The vanilla capacity and continuation lane remains closed.

## Compute continuity

- R1 reached its natural terminal state and its Windows task is disabled, not deleted.
- R4.1 is the durable guardian. Its crash/restart and single-owner checks passed.
- The dispatcher is `ACTIVE_SAFE_IDLE` because no reviewed exact launch manifest is currently valid.
- The local RTX 3090 has sustained zero compute processes, but idle hardware alone is not launch authority.
- The active RunPod A6000 process belongs to the parallel SNU AI Challenge and remains protected.
- VESSL workspace `85899410676` is control-plane `running`. The tracked EXP037 Control publisher is `TRAINING_COMPLETE`, while Candidate progress is unavailable to the local observer; treat unknown progress as degraded rather than inventing it.
- EXP037 is an EXP034/ScoreAlignedLoss robustness replication. It is not a normalized-L1 PromptMR fork and is not automatically promotable.

There is no midnight cutoff. A midnight record is a status snapshot only; healthy, reviewed work continues to its normal terminal gate.

## Active research gates

| Lane | Status | Next valid transition |
|---|---|---|
| Annotation-aware VarNet V1 | `SOURCE_DIVERGED / REVIEW_INVALIDATED` | Restore an isolated exact-byte snapshot, refreeze, and run one replacement independent review |
| PromptMR+ scratch candidate | `HARD_BLOCKED` | Add finite guards and run a bounded diagnostic before any new scratch e0-e5 run; final promotion remains license-gated |
| Feature/FI alignment | CPU repair | Prove runtime shape, measured-data, mask, and inference parity before a GPU screen |
| Augmentation/remasking | CPU parity package | Prove physics-consistent transforms and deterministic provenance |
| Acceleration-routed MoE | CPU package | Route only from legal mask structure and keep inference annotation-free |
| TTA/ensemble | Design/package | Keep every reconstruction operation inside timed `recon_slice()` |

The first annotation review was revoked because `reconstruction_weight=0` was accepted. A v2 manifest corrected that contract and passed 131 focused tests, but seven of its eighteen frozen paths later changed in the live research worktree. The old review and v2 manifest therefore cannot authorize a launch.

## Rule resolution

Organizer [Issue #409](https://github.com/LISTatSNU/FastMRI_challenge/issues/409#issuecomment-5013857396) permits all four requested operations under the stated conditions:

1. LOCAL or external-server architecture and scalar-hyperparameter screening with allowed data.
2. Train/validation k-space mask and MRI-coordinate augmentation/remasking.
3. General ensemble and TTA under the declared inference constraints.
4. Private external-cloud processing while honoring the DSA.

Final-submitted model components must be trained end-to-end on VESSL from an allowed, code-reproducible initialization. No checkpoint, optimizer, scheduler, scaler, EMA/SWA, RNG state, teacher output, or reconstruction cache from LOCAL or RunPod screening may initialize the final VESSL run.

Organizer [Issue #408](https://github.com/LISTatSNU/FastMRI_challenge/issues/408#issuecomment-5013509585) remains a separate hard constraint: external-data-derived or externally supplied/file-loaded initialization weights are prohibited.

## Standing operational authorization

Official evaluation, eligible score submission, and additive Git publication have standing user authorization. They still require the exact candidate, evaluator, provenance, coverage, archive, hardware, and quota gates. Authorization never converts a diagnostic, canary, LOCAL/RunPod screen weight, or invalidated manifest into an eligible submission.
