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

## Completed gate: EXP035

`EXP035` tested `c8/ch12/s8` over 30 epochs against the prior c6/ch12 capacity result. It completed with terminal exit code 0 and closed the unmodified vanilla VarNet capacity track.

- All 30 retained epochs passed exact `30 volumes / 791 slices / 161 boxes`, `skipped=[]`, unknown 0, and finite-output gates.
- Epoch 30 won at local quality `0.9199788092310326`, `+0.004296353336923686` over EXP033R LOCAL.
- The approved official one-shot scored full `0.9234`, bbox `0.9177`, quality `0.92055`, and total `0.92146109375` at `250.7 ms/slice`.
- The immutable checkpoint generation is `3e8af14268a64d67a308ebe30484ddf2`, SHA-256 `dc6e034f18df2a7872c416d4dccb4bb00e6e5b41fb89e438a86682db3097ffb7`.

The completed gate used the leaderboard-faithful EXP033R LOCAL reference `0.9156824558941089`. New matched recipe candidates must now compare against EXP035 epoch 30 at `0.9199788092310326`.

| EXP035 result | Decision |
|---|---|
| quality `<= 0.9156824558941089` | Reject c8 and stop unmodified vanilla depth scaling. |
| gain `0 ~ 0.0005` | Require matched robustness or seed evidence. |
| quality `>= 0.9161824558941088` | Use c8 as the vanilla baseline for controlled recipe tests. |

## Next actions

1. Protect EXP035 epoch 30 as the new one-shot official leader; do not rerun its one-shot.
2. Decide separately whether to freeze it for the approval-gated 30-repeat cohort or spend one bounded Adam lower-LR continuation with architecture/objective/data fixed.
3. Do not launch c9/c10/c12 capacity rescue. Any next model-family work must pin a licensed upstream implementation and integrate it through a thin repository adapter.
4. Before any new family trains, run the maximum-input GTX 1080 FP32/no-grad deployment contract; PromptMR+ stops immediately if its license is incompatible.
5. After explicit final freeze approval, run the 30-repeat timing cohort, fresh-clone package verification, and upload.

## Guardrails

- Do not modify `recon_eval.py` or mounted `Data`.
- Do not commit data, H5 files, checkpoints, result directories, `.env` files, or credentials.
- LOCAL results are evidence only; they do not become official candidates without independent validation and approval.
- Gradient checkpointing reduces training activation memory only. It does not reduce final checkpoint size, inference VRAM, or inference time; the final model must fit 8 GB structurally.
