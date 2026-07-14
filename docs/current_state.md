# Current state

The root [`README.md`](../README.md) is the live VESSL dashboard. This page records the current candidate hierarchy and the decisions that govern the next work.

For the full execution plan and RTX 3090 training role, see [`our_strategy.md`](our_strategy.md). Final promotion is governed by the exact GTX 1080, 16 GB RAM, and runtime acceptance rules in [`final_evaluation_server.md`](final_evaluation_server.md).

## Candidate status

| Role | Experiment | Status |
|---|---|---|
| Protected official leader | `EXP033R_varnet_c4_ch12_s8_lr3e4_e33`, epoch 32 | One approved official run; current one-shot leader |
| Verified fallback | `EXP030_varnet_c4_ch12_s8_e20` | Official score and 30-run timing cohort complete |
| Active capacity experiment | `EXP035_varnet_c8_ch12_s8_e30` | VESSL training; one-variable cascade comparison against EXP032 |
| Rejected score-aligned-loss direction | `EXP034_varnet_c4_ch12_s8_lr3e4_scorealigned_e33` | Official one-shot trailed the protected leader |
| Rejected LOCAL optimizer direction | `LOCAL_EXP068_RETRY8_varnet_c8_ch12_s8_e5_adamw_wd1e6_seed430` | Exact tie with matched Adam; 7.50% slower; no follow-up authorized |

## Official reference results

| Candidate | Evidence | Quality | Time | Total score |
|---|---|---:|---:|---:|
| EXP030 | 30-run minimum | 0.91430 | 173.4 ms/slice | 0.9152513541666667 |
| EXP031 | one official run | 0.91525 | 173.1 ms/slice | 0.9162015104166666 |
| EXP032 | one official run | 0.91585 | 212.2 ms/slice | 0.9167811458333334 |
| EXP033R, epoch 32 | one official run | **0.91595** | 173.6 ms/slice | **0.91690125** |
| EXP034 | one official run | 0.91435 | 173.7 ms/slice | 0.9153011979166666 |

`EXP033R` is protected because it improves both the quality/timing balance and the total score. `EXP030` remains the safe fallback because its 30-run timing cohort is complete.

## Closed LOCAL recipe gate: AdamW

The independently verified V10 matched optimizer probe held source, `c8/ch12/s8`, seed `430`, LR `0.001`, batch size 1, five epochs, and evaluation fixed. Adam and AdamW (`weight_decay=1e-6`) both selected epoch 4 and tied exactly at equal-acc quality `0.9079597004022214`; all seven reported metric deltas were zero. AdamW was `9m51s` (`7.50%`) slower. Retain Adam and stop the AdamW branch: no second seed, longer AdamW run, scheduler rescue, VESSL promotion, or official evaluation is authorized. The independently audited evidence is recorded in [`../reports/local_comparisons/local_adamw_matched_e5_v10_20260714.md`](../reports/local_comparisons/local_adamw_matched_e5_v10_20260714.md).

## Active gate: EXP035

`EXP035` tests `c8/ch12/s8` over 30 epochs against the prior c6/ch12 capacity result. It is the final decisive capacity test for the unmodified vanilla VarNet track.

- Do not start a competing VESSL GPU job while it trains.
- Verify every retained candidate independently with 30 volumes, 791 slices, 161 boxes, and `skipped=[]`.
- Recompute full/bbox × acc4/acc8 and equal-acc quality from the source artifacts.
- Measure inference VRAM and official-path timing only after training is terminal and approval is granted.

The leaderboard-faithful LOCAL reference for `EXP033R` is `0.9156824558941089`.

| EXP035 result | Decision |
|---|---|
| quality `<= 0.9156824558941089` | Reject c8 and stop unmodified vanilla depth scaling. |
| gain `0 ~ 0.0005` | Require matched robustness or seed evidence. |
| quality `>= 0.9161824558941088` | Use c8 as the vanilla baseline for controlled recipe tests. |

## Next actions

1. Finish and independently validate EXP035; no official evaluation is automatic.
2. Use the local RTX 3090 24 GB environment for main training, longer matched runs, and seed confirmation; use 8 GB VESSL only to prove final inference compatibility and run approved official evaluations.
3. Keep Adam as the optimizer baseline. After EXP035 selects the architecture baseline, consider at most one independently preregistered Adam scheduler-only comparison; do not continue the rejected AdamW branch automatically.
4. Add opt-in memory controls with output/resume parity tests, then test masked SSIM + L1 separately from optimizer, scheduler, and architecture changes.
5. Run a bounded Feature/FI-VarNet versus reduced PromptMR feasibility race only after their largest-input 8 GB inference contract is viable.
6. Freeze one finalist and one fallback, then run the approved 30-repeat timing cohort, fresh-clone package verification, and upload.

## Guardrails

- Do not modify `recon_eval.py` or mounted `Data`.
- Do not commit data, H5 files, checkpoints, result directories, `.env` files, or credentials.
- LOCAL results are evidence only; they do not become official candidates without independent validation and approval.
- Gradient checkpointing reduces training activation memory only. It does not reduce final checkpoint size, inference VRAM, or inference time; the final model must fit 8 GB structurally.
