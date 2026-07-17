# 40-day FastMRI score-optimization roadmap

Window: 2026-07-11 through 2026-08-20.

## Operating rules

1. Target total `0.94`. EXP035 total `0.92146109375` leaves `+0.01853890625` at the same time score, so marginal vanilla continuation is not the primary route.
2. Keep EXP035 and the verified EXP030 package immutable as leader and fallback.
3. Use leaderboard-faithful local validation as the promotion gate: average acc4 and acc8 within each metric before averaging foreground and bbox. Keep pooled `overall` only as a diagnostic. Official `recon_eval` remains explicit-approval only.
4. Do not run the final 30-repeat timing cohort until the model is frozen.
5. Stop weak directions early after enough evidence; spend GPU time on confirmed improvements.
6. Preserve checkpoint hashes, exact commands, code commit, validation metrics, and error scans for every promoted run.
7. Cap exploratory training at 720 GPU-hours and stop broad training by 2026-08-11, preserving the final nine days for official evaluation, freeze, timing, packaging, and submission.
8. Separate training memory from inference memory. Before any long LOCAL architecture run, preflight its untrained maximum-input forward path on the actual 8 GB contract; do not assume a model that needs more than 8 GB to train also needs compression to infer.
9. Treat [`final_evaluation_server.md`](final_evaluation_server.md) as the deployment authority: GTX 1080 Pascal, 16 GB host RAM, driver 550.127.08, and the captured PyTorch runtime—not a generic 8 GB GPU—decide final compatibility.

## Compute and evidence budget

- Use at most 720 GPU-hours for screening, promotion, and seed confirmation; unused time is banked rather than spent on weak ideas.
- Evaluate declared gate checkpoints instead of selecting an undeclared transient best after inspecting the curve.
- Carry at most two finalists into seed confirmation. A finalist should beat EXP031 in at least two of three total seeds, have positive mean gain, and reproduce under deterministic rescoring.
- Allow at most three new approval-gated official evaluations before final freeze.
- Do not spend a long run on a compression-only teacher unless it either passes the 8 GB direct-inference preflight or demonstrates a plausible route to an oracle gap of at least 0.001 over the best fixed 8 GB student.

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

1. Use the resume-LR/history-prefix support published in commit `431c69018678c47ae90ecba9c3863a5ef47ab68b`; its EXP033 CPU-only launch-readiness gate passed against the safe EXP031 artifact.
2. Use the leaderboard-faithful strict evaluation, retained validation epochs, epoch sweep, averaging, and candidate materialization published in commit `24bf00677c64e7a0bd84f95d68ded8beb7925b12`. EXP032 was launched without retained validation epochs, so it cannot sweep immutable per-epoch generations; EXP033 and later runs must use `--retain-val-epochs` for score-faithful post-training selection.
3. Audit mask/ACS detection, normalization, target range, crop, dtype, and metric aggregation against the official pipeline. Initial audit found masking/crop/normalization aligned, uint8 masks as compatibility debt, and inferred ACS one central line narrower than the contiguous run as an A/B hypothesis.
4. Keep c4/ch18 as a lower-priority fallback; historical ch9 -> ch12 gained only 0.000206 quality, so width ranks below LR, loss alignment, and cascades.

### Selection

Compare EXP031 through EXP035 using full, bbox, combined quality, acc4/acc8 breakdowns, complete coverage, and reproducible checkpoint provenance. EXP035 epoch 30 is the protected local and one-shot official leader; its last-five trajectory is positive but non-monotonic. Official one-shot evaluation remains approval-gated and is reserved for a meaningful, confirmed validation winner.

### Revised deployment-first capacity decision

The default is no longer "train an unconstrained teacher, then compress it." Training retains backward activations, whereas the fixed evaluator runs `model.eval()`, batch 1, and `torch.no_grad()`. A model that is expensive to train can therefore still be a valid uncompressed 8 GB inference model.

Use this gate before every PromptMR+/Feature/FI or other large-model run:

1. Freeze the candidate config, commit, maximum input shape/coil count, dtype, and checkpoint schema.
2. On the 8 GB target, warm up and repeatedly run the official `recon_eval.py` call path with untrained weights.
3. Record OOM status, `max_memory_allocated`, `max_memory_reserved`, allocator headroom, and ms/slice. A single successful forward is insufficient.
4. If it passes, train the same architecture on RTX 3090 with training-only checkpointing/accumulation and deploy it without compression.
5. If it fails, test output-equivalent per-coil sensitivity, coil chunking, tensor-lifetime cleanup, and only then parity-checked selective FP16.
6. Distill only if the exact model still cannot fit and a large teacher beats the best fixed 8 GB baseline by at least 0.001 equal-acc quality with healthy full/bbox and acc4/acc8 submetrics.

For distillation, freeze the student and its 8 GB contract first. Start with supervised SSIM/L1 plus final-output imitation; add cascade-output or attention/feature transfer as a separate ablation. Promotion still depends on absolute score: at least +0.0005 over the current leader, complete coverage, repeated 8 GB stability, and an acceptable official-time tradeoff. Do not promote based only on percentage of teacher gain retained.

Post-training unstructured pruning and INT8 are last-resort research tracks, not assumed deployment tools: sparse weights do not guarantee lower GTX 1080 memory/runtime, and complex FFT/data-consistency parity must be demonstrated. If pruning is tested, compare initialization-time pruning against dense and post-training controls rather than presuming last-minute pruning will preserve quality.

## Phase 2 — supported follow-ups (July 21–31)

Prioritize by measured gain per GPU-day:

1. Protect completed EXP035 epoch 30 and keep unmodified vanilla capacity/continuation closed.
2. Matched continuation R1 is complete: Candidate-Control `+0.0003468637 < +0.0005`, acc8 bbox `-0.0003703822`; no second seed, epoch 40, c9/c10/c12 expansion, or official evaluation.
3. Prioritize RU-NCRL PromptMR+ commit `934eeda6d4d18cd39e406fa1eee9e1f70603cb5e` for pinned thin-adapter CPU schema and maximum-input 8 GB feasibility work; its status for this workflow is `NONCOMMERCIAL_COMPETITION_USE_ALLOWED`, with notices/disclaimer preserved and no commercial-rights claim. Keep measured MIT Feature/FI commit `91f2df47` as the fallback.
4. Train only an upstream candidate that passes CPU schema and GTX 1080 maximum-input feasibility; use activation checkpointing and accumulation only as LOCAL training controls.
5. Test output-equivalent coil/sensitivity memory controls before any lossy compression. Treat `inference_mode` as an opt-in parity benchmark against the existing no-grad control. Treat FP16 as a memory experiment, not an assumed speed path on GTX 1080.
6. If a non-deployable teacher clears the 0.001 oracle-gap gate, distill it into a preregistered 8 GB student; otherwise stop the compression track.
7. Cascades: if the capacity signal warrants it, test progressive cascade expansion or a c4 -> c6 warm start with added cascades initialized as no-op residual updates, without simultaneously changing the reconstructor.
8. Acceleration specialists: audit mask/ACS behavior first, then consider separate acc4/acc8 models routed only by the observed mask.
9. Same-basin checkpoint interpolation and EMA/SWA: validation-only, no additional inference cost.
10. One independent seed of the strongest configuration; output ensembling is viable only when its quality gain exceeds its measured timing penalty.
11. Final train+validation retraining only after hyperparameters are frozen and only if the challenge rules explicitly permit use of public validation labels for final training.

Deprioritize broad width sweeps, sensitivity width above 8, unbounded fixed-LR continuation, GAN/perceptual/diffusion objectives, test-time training, GRAPPA/ESPIRiT assistance, and full FI-VarNet/XPDNet rewrites. Do not recreate a winner-scale stack from E2E-VarNet when compatible upstream code exists. Wavelet or feature-propagation changes are contingency research only after the supported interventions have been tested.

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
- PromptMR+ improves information flow and sensitivity estimation, publishes fastMRI results and code, and explicitly separates high training-memory needs from its inference path: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09565.pdf>, <https://github.com/hellopipu/PromptMR-plus>.
- KD-MRI supports attention/feature transfer plus output imitation as a conditional compact-student path: <https://arxiv.org/abs/2004.05319>.
- PUN reports that initialization-time pruning outperformed post-training pruning for an unrolled MoDL setting; this motivates an ablation, not an assumption of real sparse-kernel savings on the target GPU: <https://arxiv.org/abs/2412.18668>.
- SDUM reports progressive cascade expansion and sampling-aware weighted data consistency, but remains a 2025 preprint and is therefore a bounded follow-up rather than the baseline rewrite: <https://arxiv.org/abs/2512.17137>.
- PyTorch documents the memory effect of no-grad, the additional restrictions/overhead reduction of inference mode, and allocated-versus-reserved CUDA peak measurements: <https://docs.pytorch.org/docs/stable/generated/torch.no_grad>, <https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html>, <https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html>.
- MRAugment is retained as a lower-priority option because it reports diminishing benefit at full-data scale and requires physics-correct complex-coil transforms: <https://arxiv.org/abs/2106.14947>.
- SWA is a generic no-inference-cost averaging option after MRI-specific interventions: <https://arxiv.org/abs/1803.05407>.

## Current action

EXP035 epoch 30 is complete and protected, and the vanilla capacity/continuation track is formally closed. PromptMR+ is now the priority structural feasibility candidate at exact commit `934eeda6d4d18cd39e406fa1eee9e1f70603cb5e` under RU-NCRL status `NONCOMMERCIAL_COMPETITION_USE_ALLOWED`; FI-VarNet remains the measured fallback. The bounded task is exact-source verification, thin-adapter CPU schema tests, and no-training maximum-input FP32 8 GB feasibility. No e5/e15/e30 training or official evaluation starts automatically.
