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

## Current gate

Run a 30-repeat EXP031 timing cohort only with separate approval. Replace and package EXP030 only if EXP031 remains ahead after the required timing selection.

## Permanent constraints

- Do not modify `recon_eval.py`.
- Do not commit checkpoints, data, H5 files, result directories, `.env` files, or credentials.
- Do not run training or official evaluation without checking GPU/process state and approval scope.
