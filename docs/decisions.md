# Decision log

Detailed experiment metrics live in [`../experiments/experiment_log.csv`](../experiments/experiment_log.csv). Official results live in [`../reports/phase2/`](../reports/phase2/).

| Date | Decision | Reason |
|---|---|---|
| 2026-07-06 | Use the official baby VarNet as the baseline. | It matches the challenge starting point and official I/O. |
| 2026-07-06 | Treat VESSL `EXP###` runs as the source of truth. | Desktop `LOCAL_` runs are probes only. |
| 2026-07-07 | Continue after EXP000/EXP001 smoke tests. | Data mounts, training, checkpoints, reconstructions, validation, and plotting worked. |
| 2026-07-08 | Prefer c4/ch12/s8 for longer training. | Increasing cascades and sensitivity channels improved validation quality. |
| 2026-07-10 | Select EXP030 over EXP012. | EXP030 improved both validation and official quality; timing did not offset the gain. |
| 2026-07-10 | Use EXP030 as the Phase 2 submission candidate. | Official score 0.9152513541666667; all 30 timing runs completed. |
| 2026-07-10 | Deliver EXP030 to the GitHub default branch. | Submission checks and fresh-clone verification passed. |
| 2026-07-10 | Continue EXP030 from epoch 20 to create EXP031. | It tests training duration without repeating the first 20 epochs. |
| 2026-07-10 | Keep EXP030 official while EXP031 trains. | EXP031 must finish final validation and receive approval before official evaluation. |
| 2026-07-11 | Recommend EXP031 for one approved official evaluation. | EXP031 completed cleanly and improved validation SSIM_full, SSIM_bbox, and quality; best epoch 27. |
| 2026-07-11 | Treat EXP031 as the one-shot official leader. | Its approved official run scored 0.9162015104166666, ahead of EXP030's finalized score by 0.00095015625. |
| 2026-07-11 | Defer the final 30-run timing cohort until model freeze. | Forty days remain, so quality experiments can still replace EXP031; timing a provisional model now would be wasteful. |
| 2026-07-11 | Start EXP032 with c6/ch12/s8. | A worst-width GPU probe passed, and increasing cascades is the highest-information one-variable capacity test from EXP031. |
| 2026-07-11 | Queue a five-epoch EXP031 continuation at LR 3e-4 after EXP032. | EXP031 peaked at epoch 27 and then oscillated under fixed LR 1e-3; published VarNet recipes and the codebase audit favor a lower LR before more fixed-rate epochs. |
| 2026-07-11 | Rank score-aligned foreground/bbox loss above channel widening. | The official objective weights foreground and bbox equally, training has 2,593 boxes, and historical ch9 -> ch12 improved validation quality by only 0.000206. |
| 2026-07-11 | Use equal-acceleration local quality for promotion. | The fixed evaluator averages acc4 and acc8 equally; pooled local validation overweights acc4 because slice and box counts differ, biasing EXP031's diagnostic quality upward by 0.002026883539. |
| 2026-07-11 | Stop broad training by August 11 and cap exploration at 720 GPU-hours. | This preserves nine days for confirmation, approval-gated official evaluation, immutable freeze, repeated timing, package verification, and upload. |

## Current gate

EXP031 stays protected while EXP032 and the forty-day roadmap run. Promote only local validation winners to separately approved official one-shot evaluation. Run the 30-repeat timing cohort only for the frozen final candidate.

## Permanent constraints

- Do not modify `recon_eval.py`.
- Do not commit checkpoints, data, H5 files, result directories, `.env` files, or credentials.
- Do not run training or official evaluation without checking GPU/process state and approval scope.
