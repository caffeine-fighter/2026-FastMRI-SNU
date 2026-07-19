# Current state

The root [`README.md`](../README.md) is the live VESSL dashboard. This page records the current candidate hierarchy and the decisions that govern the next work.

For the full execution plan and RTX 3090 training role, see [`our_strategy.md`](our_strategy.md). Final promotion is governed by the exact GTX 1080, 16 GB RAM, and runtime acceptance rules in [`final_evaluation_server.md`](final_evaluation_server.md).

## Candidate status

| Role | Experiment | Status |
|---|---|---|
| Protected official leader | `EXP035_varnet_c8_ch12_s8_e30`, epoch 30 | One approved official run; current one-shot leader |
| Verified fallback | `EXP030_varnet_c4_ch12_s8_e20` | Official score and 30-run timing cohort complete |
| Prior one-shot leader | `EXP033R_varnet_c4_ch12_s8_lr3e4_e33`, epoch 32 | Preserved immutable fallback candidate |
| Rejected score-aligned-loss direction | `EXP034_varnet_c4_ch12_s8_lr3e4_scorealigned_e33` | Official one-shot trailed the protected leader |
| Rejected optimizer direction | LOCAL AdamW V10 | Exact quality/component tie versus Adam and 7.50% slower; closed |
| Rejected lower-LR continuation | EXP035 matched continuation R1 | Candidate beat Control by only `+0.0003468637 < +0.0005`, with acc8 bbox regression; closed |

## Official reference results

| Candidate | Evidence | Quality | Time | Total score |
|---|---|---:|---:|---:|
| EXP030 | 30-run minimum | 0.91430 | 173.4 ms/slice | 0.9152513541666667 |
| EXP031 | one official run | 0.91525 | 173.1 ms/slice | 0.9162015104166666 |
| EXP032 | one official run | 0.91585 | 212.2 ms/slice | 0.9167811458333334 |
| EXP033R, epoch 32 | one official run | 0.91595 | 173.6 ms/slice | 0.91690125 |
| EXP034 | one official run | 0.91435 | 173.7 ms/slice | 0.9153011979166666 |
| EXP035, epoch 30 | one official run | **0.92055** | 250.7 ms/slice | **0.92146109375** |

`EXP035` is protected because its `+0.0046` quality gain over EXP033R materially exceeds its `0.00004015625` time-score penalty, improving total by `+0.00455984375`. `EXP033R` remains the faster prior leader, and `EXP030` remains the finalized timing fallback.

## Active prize-first queue — 2026-07-19

| Lane | Current state | Constraint |
|---|---|---|
| R4.1 continuity | Durable guardian running; dispatcher `ACTIVE_SAFE_IDLE` | R1 ended naturally and is disabled; no reviewed exact launch manifest exists |
| LOCAL RTX 3090 | Sustained compute-process zero | Hardware availability is not launch authority |
| RunPod A6000 | Protected SNU AI Challenge process healthy | Do not interrupt; any later FastMRI weights are disposable screens only |
| VESSL EXP037 | Workspace running; Control publisher `TRAINING_COMPLETE`; Candidate progress unavailable | EXP034/ScoreAlignedLoss robustness replication only; not a normalized-L1 PromptMR fork |
| EXP036 PromptMR+ | `HARD_BLOCKED` | NaN at iteration 590 and no e1/e5 checkpoint; unchanged restart prohibited and final promotion remains license-gated |
| Annotation-aware V1 | `SOURCE_DIVERGED / REVIEW_INVALIDATED` | Seven of eighteen v2 manifest paths changed; restore/refreeze and independently re-review |
| Feature/FI and physics policies | CPU package work | Shape/data-consistency and physics parity must precede a GPU screen |

The organizer has resolved the former rule unknowns. External architecture/hyperparameter screening, train/validation k-space transformation, ensemble/TTA, and private cloud processing are allowed under [Issue #409](https://github.com/LISTatSNU/FastMRI_challenge/issues/409#issuecomment-5013857396). External-data-derived or externally supplied/file-loaded initialization weights remain prohibited by [Issue #408](https://github.com/LISTatSNU/FastMRI_challenge/issues/408#issuecomment-5013509585). Final-submitted components must be trained end-to-end on VESSL; learned state from LOCAL/RunPod screens must not cross into the final run.

See [`prize_first_r4_status.md`](prize_first_r4_status.md) for the timestamped operational snapshot.

## Completed gate: EXP035

`EXP035` tested `c8/ch12/s8` over 30 epochs against the prior c6/ch12 capacity result. It completed with terminal exit code 0 and closed the unmodified vanilla VarNet capacity track.

- All 30 retained epochs passed exact `30 volumes / 791 slices / 161 boxes`, `skipped=[]`, unknown 0, and finite-output gates.
- Epoch 30 won at local quality `0.9199788092310326`, `+0.004296353336923686` over EXP033R LOCAL.
- The approved official one-shot scored full `0.9234`, bbox `0.9177`, quality `0.92055`, and total `0.92146109375` at `250.7 ms/slice`.
- The immutable checkpoint generation is `3e8af14268a64d67a308ebe30484ddf2`, SHA-256 `dc6e034f18df2a7872c416d4dccb4bb00e6e5b41fb89e438a86682db3097ffb7`.
- Epoch 30 was the global retained winner. From epoch 26 to 30, quality/full/bbox changed by `+0.0003553055859381 / +0.0002618608353651 / +0.0004487503365111`; the trajectory was positive but non-monotonic.

The completed gate used the leaderboard-faithful EXP033R LOCAL reference `0.9156824558941089`. New matched recipe candidates must now compare against EXP035 epoch 30 at `0.9199788092310326`.

## Completed gate: matched LR continuation

The VESSL R1 pair independently resumed the exact EXP035 epoch-30 generation and trained epochs 31–35 with only Adam LR changed.

- Control LR `0.001`: epoch 34 won at quality `0.9202459832836087`.
- Candidate LR `0.0003`: epoch 34 won at quality `0.9205928470035448`.
- Candidate minus Control: `+0.0003468637199360858`, below the `+0.0005` promotion threshold.
- Protected acc8 bbox changed by `-0.00037038215884455106`.
- Both arms passed exact coverage, H5, checkpoint, history, source-hash, and finite-output gates.
- Decision: `DO_NOT_PROMOTE`; no official evaluation or repeated timing was run.
- Candidate epoch 34 is a research artifact only. Do not run a second seed, epoch-40 continuation, or c9/c10/c12 vanilla expansion.
- The vanilla capacity/continuation track is closed by the matched `+0.0005` promotion gate and protected-component regression, not by a fixed target-score gap. Prize success is final rank `<= 5`, the stretch objective is rank `1`, and `target_score=null`; future work is prioritized by its evidence-backed chance of improving those rank outcomes.

See the [complete matched continuation report](../reports/local_comparisons/exp035_matched_continuation_r1_20260716.md).

| EXP035 result | Decision |
|---|---|
| quality `<= 0.9156824558941089` | Reject c8 and stop unmodified vanilla depth scaling. |
| gain `0 ~ 0.0005` | Require matched robustness or seed evidence. |
| quality `>= 0.9161824558941088` | Use c8 as the vanilla baseline for controlled recipe tests. |

## Next actions

1. Protect EXP035 epoch 30 as the one-shot official leader; do not rerun its one-shot.
2. Keep the matched lower-LR continuation and the entire vanilla capacity/continuation track closed. Do not run another block, second seed, c9/c10/c12 expansion, or official evaluation; Candidate epoch 34 is research-only.
3. AdamW-only is closed. Do not run a second seed, long rescue, scheduler combination, VESSL promotion, or official evaluation for AdamW.
4. Do not launch c9/c10/c12 capacity rescue. Prioritize an exact-byte re-freeze/re-review of annotation-aware V1, a PromptMR finite diagnostic, and Feature/FI/physics parity packages.
5. Use LOCAL/RunPod only for architecture and scalar-recipe screening. Transfer no learned state; retrain every final component end-to-end on VESSL.
6. Before any new family trains for promotion, run the maximum-input GTX 1080 FP32/no-grad deployment contract and exact-manifest gate.
7. Official evaluation, eligible score submission, and additive Git publication have standing authorization, but may run only after their candidate-specific integrity and eligibility gates pass.
8. Run the final 30-repeat timing cohort, fresh-clone package verification, and upload only after the finalist is frozen.

## Guardrails

- Do not modify `recon_eval.py` or mounted `Data`.
- Do not commit data, H5 files, checkpoints, result directories, `.env` files, or credentials.
- LOCAL/RunPod results are screening evidence only. Their weights and learned state never become or initialize official candidates.
- Final-submitted model components must be trained end-to-end on VESSL from allowed, code-reproducible initialization.
- Midnight is an atomic status snapshot, not a shutdown or dispatch cutoff.
- Gradient checkpointing reduces training activation memory only. It does not reduce final checkpoint size, inference VRAM, or inference time; the final model must fit 8 GB structurally.
- Do not reinvent the post-EXP035 model stack. Pin, license, and smoke-test upstream code before writing repository adapters.
