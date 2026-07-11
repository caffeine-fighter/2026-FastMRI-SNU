# 40-day FastMRI score-optimization roadmap

Window: 2026-07-11 through 2026-08-20.

## Operating rules

1. Keep EXP031 and the verified EXP030 package immutable as fallbacks.
2. Change one major variable at a time until a direction is validated.
3. Use leaderboard-faithful local validation as the promotion gate: average acc4 and acc8 within each metric before averaging foreground and bbox. Keep pooled `overall` only as a diagnostic. Official `recon_eval` remains explicit-approval only.
4. Do not run the final 30-repeat timing cohort until the model is frozen.
5. Stop weak directions early after enough evidence; spend GPU time on confirmed improvements.
6. Preserve checkpoint hashes, exact commands, code commit, validation metrics, and error scans for every promoted run.
7. Cap exploratory training at 720 GPU-hours and stop broad training by 2026-08-11, preserving the final nine days for official evaluation, freeze, timing, packaging, and submission.

## Compute and evidence budget

- Use at most 720 GPU-hours for screening, promotion, and seed confirmation; unused time is banked rather than spent on weak ideas.
- Evaluate declared gate checkpoints instead of selecting an undeclared transient best after inspecting the curve.
- Carry at most two finalists into seed confirmation. A finalist should beat EXP031 in at least two of three total seeds, have positive mean gain, and reproduce under deterministic rescoring.
- Allow at most three new approval-gated official evaluations before final freeze.

## Promotion gates

A candidate advances when all of these hold:

- leaderboard-faithful validation quality exceeds the current leader (EXP031: 0.9154137446412757); the pooled EXP031 diagnostic is 0.9174406281804748 and must not be used as the promotion threshold;
- neither SSIM_full nor SSIM_bbox regresses materially without a compensating total-quality gain;
- 30 validation volumes, 791 slices, and 161 bounding boxes are evaluated with zero skips;
- checkpoint provenance and clean training completion are verified;
- one approved official run confirms that the local gain transfers.

Suggested thresholds:

- strong promotion: quality gain >= 0.0005;
- marginal promotion: gain 0 to 0.0005, requiring repeat/local robustness checks;
- reject: quality <= leader or any unexplained subgroup collapse.

## Phase 1 — architecture and optimization (July 11–21)

### EXP032 — running now

- VarNet cascades 6, channels 12, sensitivity channels 8, 30 epochs from scratch at LR 1e-3.
- One-variable change from EXP031: cascades 4 -> 6.
- The prior one-epoch local c6 probe beat c4 quality by 0.000782.
- Worst-width one-step GPU probe passed: 6,338 MiB allocated, 7,386 MiB reserved.
- Estimated full runtime: about 39.1 hours.
- Keep this already-running capacity measurement, but do not interpret one seed as final proof.

### EXP033 — highest-priority post-EXP032 screen

- Resume EXP031's best epoch-27 checkpoint from the safe converted artifact.
- Run five additional epochs, checkpoint epoch 28 through total epoch 33, with optimizer LR overridden to 3e-4.
- Preserve model weights and Adam moments. EXP031's legacy checkpoint lacks RNG/data-order state, so this continuation is valid but not bit-exact.
- Promotion gate: leaderboard-faithful validation quality must exceed EXP031's 0.9154137446412757; retain 0.9174406281804748 only as the historical pooled diagnostic.
- If promoted, continue the new best for three additional epochs at LR 1e-4.
- Estimated initial cost: 5.4 GPU-hours. CPU-only resume preflight passed.

### EXP034 — score-aligned loss screen

- Use the same EXP031 source checkpoint, architecture, LR 3e-4, and five-epoch budget as EXP033.
- Change only the objective, using an unbiased per-slice estimator of the fixed evaluator: equal 25% weight for foreground acc4, foreground acc8, bbox acc4, and bbox acc8; every box is a separate observation.
- Do not use the naive per-annotated-slice 0.5 foreground + 0.5 mean-box loss: with only 1,824 of 4,651 slices annotated, it would yield about 80.39% foreground / 19.61% bbox globally and would weight annotated slices rather than boxes.
- Compute acceleration-specific slice and box counts from the selected training set at loader initialization; do not hard-code dataset counts into the loss.
- Training has 2,593 boxes across 1,824 of 4,651 slices. Annotations remain training-only and are structurally forbidden from inference.
- Require parity tests against `utils/common/metrics.py`, including empty boxes, multiple boxes, crop boundaries, the seven-pixel SSIM window, volume-level `max`, equal-acc aggregation, and finite gradients.
- Stop if bbox improvement is paid for by a larger full-image regression.

### Required infrastructure and audits

1. Finish reviewed resume-LR/history-prefix support before EXP033.
2. Use leaderboard-faithful equal-acc validation quality for promotion, and add challenge-quality checkpoint selection; current training still selects `best_model.pt` using whole-image skimage SSIM with `target.max()` and can mis-rank epochs.
3. Audit mask/ACS detection, normalization, target range, crop, dtype, and metric aggregation against the official pipeline. Initial audit found masking/crop/normalization aligned, uint8 masks as compatibility debt, and inferred ACS one central line narrower than the contiguous run as an A/B hypothesis.
4. Keep c4/ch18 as a lower-priority fallback; historical ch9 -> ch12 gained only 0.000206 quality, so width ranks below LR, loss alignment, and cascades.

### Selection

Compare EXP031, EXP032, EXP033, and EXP034 using full, bbox, combined quality, acc4/acc8 breakdowns, complete coverage, and reproducible checkpoint provenance. Official one-shot evaluation remains approval-gated and is reserved for a meaningful, confirmed validation winner.

## Phase 2 — supported follow-ups (July 21–31)

Prioritize by measured gain per GPU-day:

1. Low-LR/scheduler finalists: Adam 3e-4 with a StepLR-style late drop versus a separately controlled cosine decay. Do not change architecture and schedule together.
2. Cascades: if EXP032 is promising, test a lower-LR continuation or a c4 -> c6 warm start with the added cascades initialized as no-op residual updates.
3. Acceleration specialists: audit mask/ACS behavior first, then consider separate acc4/acc8 models routed only by the observed mask.
4. Same-basin checkpoint interpolation and EMA/SWA: validation-only, no additional inference cost.
5. One independent seed of the strongest configuration; output ensembling is viable only when its quality gain exceeds its measured timing penalty.
6. Final train+validation retraining only after hyperparameters are frozen and only if the challenge rules explicitly permit use of public validation labels for final training.

Deprioritize broad width sweeps, sensitivity width above 8, fixed-LR 1e-3 continuation, GAN/perceptual/diffusion objectives, test-time training, GRAPPA/ESPIRiT assistance, and full FI-VarNet/XPDNet rewrites. Wavelet or feature-propagation changes are contingency research only after the supported interventions have been tested.

Every change starts with a short feasibility probe and gets abandoned when it fails the declared gate.

## Phase 3 — averaging, confirmation, and inference (July 31–August 10)

1. Screen same-basin checkpoint interpolation or weight averaging first because it has no inference penalty.
2. Test one independent-seed image-space ensemble only if both component models are individually credible. Require validation gain greater than 0.0001; measured doubling from 173 to 346 ms/slice costs about 0.0000902 total score.
3. Do not carry the existing EXP030+EXP031 blend forward; its measured validation gain was only 0.00000244.
4. Confirm no more than two finalists across seeds, with complete full/bbox and acc4/acc8 analysis.
5. Profile inference to remove avoidable overhead without altering outputs, and confirm 8 GB inference, submission-size, and runtime constraints.

## Phase 4 — robustness and freeze (August 10–15)

- Re-run local validation and provenance checks for the top two candidates.
- Use approved one-shot official evaluations only where validation evidence justifies them.
- Select one final candidate and one fallback.
- Freeze code, model, checkpoint hash, dependencies, and exact reconstruction command.

## Phase 5 — final evidence and submission (August 15–20)

- Run the final candidate's approved 30-repeat timing cohort.
- Require 30/30 successful runs and scan all logs for failures.
- Finalize score using the selected valid timing statistic.
- Build, hash, and fresh-clone verify the submission package.
- Upload before the final day and retain the organizer receipt/submission ID.
- Keep EXP030 and EXP031 packages untouched as emergency fallbacks.

## Evidence base

- E2E-VarNet used substantially more cascades and a lower Adam learning rate over a longer schedule: <https://arxiv.org/abs/2004.06688>.
- The official fastMRI VarNet training module provides the reference loss, optimizer, and StepLR conventions: <https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/varnet_module.py>.
- Feature/FI-VarNet reports late gains from longer scheduled optimization but its largest models exceed this 8 GB budget: <https://www.nature.com/articles/s41598-024-59705-0>.
- MRAugment is retained as a lower-priority option because it reports diminishing benefit at full-data scale and requires physics-correct complex-coil transforms: <https://arxiv.org/abs/2106.14947>.
- SWA is a generic no-inference-cost averaging option after MRI-specific interventions: <https://arxiv.org/abs/1803.05407>.

## Current action

EXP032 is training on VESSL. Do not start a competing GPU task. CPU-only analysis, resume hardening, experiment design, and documentation can proceed while it runs.
