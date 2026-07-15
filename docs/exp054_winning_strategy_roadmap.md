# EXP054 이후 우승 전략 반영 Implementation Plan

> **For Hermes:** Execute this plan phase-by-phase under the `controlled-ml-experiment-operations`, `reproducible-experiment-operations`, test-driven-development, and exact-byte review workflows. LOCAL training, VESSL training, official evaluation, Git publication, and branch deletion are separate approval scopes.

**Goal:** EXP054의 검증된 capacity signal과 `winningstrategies.txt`의 2025년 상위권 공통점을 이용해, 현재 vanilla VarNet 탐색을 빠르게 판정하고 8 GB 제약 안에서 더 높은 품질을 낼 수 있는 학습 레시피와 차세대 모델로 전환한다.

**Architecture:** 완료된 `EXP035` epoch 30을 보호 baseline으로 두고 두 트랙을 병행한다. 첫째, 같은 immutable state에서 Adam LR `1e-3` 대 `3e-4`의 bounded matched continuation을 준비한다. 둘째, LOCAL/CPU에서 메모리 절감과 upstream Feature/FI-VarNet feasibility를 adapter-first로 준비한다. AdamW-only는 종료됐고 PromptMR+는 license confirmation 전까지 코드 사용을 중단한다. 모든 방향은 `LOCAL short screen -> matched longer run -> second-seed confirmation -> separately approved VESSL run -> separately approved official run` 순서로만 승격한다.

**Tech Stack:** PyTorch, repository VarNet pipeline, strict equal-acc evaluator, final Intel i7-8700K 6C/12T + 16 GB RAM + GTX 1080 8192 MiB + driver 550.127.08, desktop RTX 4070 Ti SUPER/RTX 3090 for training, GitHub worktrees/PRs, Python `unittest`. Exact target Python/PyTorch/CUDA-runtime versions remain an environment-capture gate; see [`final_evaluation_server.md`](final_evaluation_server.md).

---

## 0. Authority snapshot

- 사용자가 말한 GitHub “main”은 현재 실제 브랜치 이름이 아니다. live default branch는 `baseline/2026-baby-varnet`이다.
- Source commit: `332f2ac4e4c9dac5634f4e8b8cf4bed86604ac57`.
- `winningstrategies.txt` blob: `31fb893ce6268a9edda83fef92f3992681526ee7`.
- EXP054 terminal: `/home/ray1001/result/LOCAL_EXP054_FRESH_E5_V1_20260713_DESKTOP4070TI/terminal/terminal.json`.
- EXP054 independent audit: `/home/ray1001/result/AUTONOMOUS_SCORE_LOOP_20260711/LOCAL_EXP054_FRESH_E5_V1_independent_audit_6330d3d0.json`.
- `EXP035_varnet_c8_ch12_s8_e30`은 terminal exit code 0으로 완료됐다. 30개 retained epoch가 모두 strict coverage를 통과했고 epoch 30이 LOCAL quality `0.9199788092310326`으로 global best였다.
- Current one-shot official leader: `EXP035_epoch30`, quality `0.92055`, total `0.92146109375`, `250.7 ms/slice`.
- Current leaderboard-faithful LOCAL protected reference: EXP035 epoch 30 `0.9199788092310326`.

## 1. What EXP054 changed

1. **Capacity allocation matters, but width is not monotonic.**
   - `c8/ch12/s8/e5` beats matched `c4/ch16/s8/e5` by `+0.002871432627` equal-acc quality.
   - Full, bbox, acc4, and acc8 components all improved.
   - The gain is larger on acc8 (`+0.003571840607`) than acc4 (`+0.002171024647`).
   - The preceding two-seed screen rejected `c8/ch18`; therefore “more channels” is not the next default.

2. **The result is a direction signal, not a candidate.**
   - EXP054 remains `-0.004974940182` below LOCAL EXP041 e15.
   - It is LOCAL-only and cannot supply an official checkpoint.

3. **The unmodified stack is already near the 8 GB ceiling.**
   - EXP054 desktop training sampled about `13,062 MiB`.
   - VESSL EXP035 reports `7,741 MiB` on the 8,192 MiB device: only `451 MiB` (`5.505%`) remains in that training probe.
   - Training memory, inference memory, and allocator-dependent peaks must remain separate measurements.

4. **The present code is missing several repeatedly winning ingredients.**
   - Default model: vanilla E2E VarNet.
   - Default optimizer: fixed-LR Adam.
   - Default objective: ordinary SSIM.
   - No activation checkpointing, gradient accumulation, gradient clipping, AdamW option, warmup, or cosine schedule is present in the current training path.
   - The prior EXP034 score-aligned sparse objective failed, but it is not the same objective as winner-style foreground-masked SSIM+L1.

5. **Strategic conclusion.**
   - EXP035 passed and closed the current unmodified vanilla capacity track. Do not launch c9/c10/c12 scaling.
   - Epoch 30 is the global best and the final-five component trajectory is positive but non-monotonic, authorizing only a bounded matched continuation.
   - The next large quality bet should be better quality per VRAM—Feature/FI-VarNet or a reduced PromptMR-family model—not `chans > 12` brute force.

---

### Task 1: Rebuild the Git consolidation on the live default branch

**Objective:** Preserve reviewed LOCAL evidence without overwriting newer VESSL/default-branch state.

**Current blocker:** PR #19 and PR #24 both conflict with the EXP035-complete default branch and carry stale overlapping status text. Neither may be merged as-is.

**Files to transplant from reviewed evidence:**
- `reports/local_comparisons/local_capacity_screen_20260713.md`
- `reports/local_comparisons/local_exp054_fresh_e5_20260713.md`
- relevant validated rows from `experiments/desktop_probe_log.csv`
- a manually reconciled LOCAL section in `README.md`

**Steps:**
1. Use a clean worktree from EXP035-complete default commit `e6c7665daca9950e3f1bdb250dcbf7376393da99`; preserve the old dirty worktree unchanged.
2. Copy only reviewed LOCAL reports; do not cherry-pick stale README/current-state text wholesale.
3. Rebuild strategy/current-state text from the current default plus validated AdamW and post-EXP035 evidence.
4. Run:
   - `python scripts/check_submission.py`
   - `git diff --check`
   - exact staged allowlist review
   - fresh remote-worktree verification after push
5. Replace or close PR #19 and PR #24 only after the fresh PR exists and is verified.
6. After merge, archive obsolete consolidation branches; keep source evidence available until remote verification completes.

**Acceptance criteria:** default-branch VESSL status stays current, EXP054 is explicitly LOCAL-only, no raw checkpoints/H5/results enter Git, and the remote merged SHA is independently verified.

---

### Task 2: Close EXP035 and preregister the matched continuation — completed/prepared

**Outcome:** c8/ch12 passed. Epoch 30 is the protected LOCAL and one-shot official leader. The next question is whether another five epochs help by themselves or specifically benefit from LR `3e-4`.

**Prepared next comparison:**
1. Source both LOCAL arms from generation `3e8af14268a64d67a308ebe30484ddf2`, SHA-256 `dc6e034f18df2a7872c416d4dccb4bb00e6e5b41fb89e438a86682db3097ffb7`.
2. Compare Adam LR `1e-3` control against Adam LR `3e-4`, total epoch 35, with architecture/objective/data/batch/seed/RNG fixed.
3. Run [`scripts/plan_exp035_continuations.py`](../scripts/plan_exp035_continuations.py) first. It is dry-run only and must not start CUDA.
4. No GPU arm starts while another process owns the GPU; no official evaluation is automatic.

**Completed EXP035 gate:**
- epoch 30 LOCAL quality `0.9199788092310326`; prior EXP033R reference `0.9156824558941089`; PASS by `+0.004296353336923686`.
- official quality `0.92055`, total `0.92146109375`; new protected one-shot leader.

**Official decision gate:**
- No official run is automatic.
- A future candidate must beat EXP035 total `0.92146109375` using its own measured time.
- Exact break-even formula: `required_quality = 0.92146109375 - time_score(actual_ms_per_slice)`.

**Branching decision:**
- **LR candidate beats fixed control by `>= 0.0005`:** retain Adam and promote lower-LR continuation for confirmation.
- **Both improve similarly:** classify as extra-epoch gain and allow at most one more bounded five-epoch block on the better arm.
- **Both plateau:** stop vanilla continuation and move GPU budget to the upstream feasibility winner.
- **Any protected-component regression:** reject regardless of aggregate quality.

---

### Task 3: Add opt-in memory engineering with parity tests

**Objective:** Free training memory and prove official 8 GB inference compatibility without changing baseline outputs.

**Likely files:**
- Modify: `train.py`
- Modify: `utils/learning/train_part.py`
- Modify: `utils/model/varnet.py`
- Test: `tests/test_train_cli.py`
- Test: `tests/test_training_resume.py`
- Create: `tests/test_memory_efficient_varnet.py`
- Create: `scripts/profile_varnet_memory.py`

**Implement as separate opt-in features:**
1. Cascade-level activation checkpointing for training only.
2. Sensitivity-map coil chunking/per-coil execution with exact inference-output parity.
3. Gradient accumulation with explicit effective-batch accounting.
4. Gradient norm telemetry; add clipping at `0.1` as a separate experiment, not silently by default.

**TDD/verification:**
1. CLI rejection tests for invalid values.
2. Fixed-input eval parity with every memory flag on/off.
3. Fixed minibatch forward/loss/gradient comparison under deterministic settings.
4. Save/resume compatibility and optimizer-state coverage tests.
5. Largest-legal-shape peak allocation/reservation measurements for training and inference separately.
6. Record process peak RSS and available host RAM as well as GPU peaks; the final server has 16 GB system memory.
7. Profile on desktop first; repeat the inference contract on the exact GTX 1080 target only after EXP035 releases the GPU.
8. Confirm `recon_eval.py` remains byte-identical.

**Gate:** a memory feature may enable a larger experiment only if outputs/checkpoint semantics pass, measured memory improves, runtime remains acceptable, and the largest legal inference input fits with a predeclared safety margin. Activation checkpointing alone is not evidence of lower inference memory.

---

### Task 4: Run a controlled optimizer/scheduler ladder

**Objective:** Replace the current fixed Adam recipe only when a source-backed matched comparison wins.

**Completed AdamW gate:** matched LOCAL V10 tied equal-acc quality and every protected component exactly while AdamW was 7.50% slower. AdamW-only is closed.

**Remaining rationale:** EXP035 ended at its global best with a positive but noisy final-five trajectory. Scheduling remains an Adam-only intervention, and the matched fixed-LR arm is required to separate LR from extra epochs.

**Likely files:**
- Modify: `train.py`
- Modify: `utils/learning/train_part.py`
- Modify: `utils/learning/resume.py`
- Test: `tests/test_train_cli.py`
- Test: `tests/test_training_resume.py`
- Create: `tests/test_optimizer_scheduler.py`

**Experiment order:**
1. Do not run another AdamW arm.
2. Execute the preregistered epoch-30-to-35 Adam fixed-LR control and lower-LR candidate only after the GPU is free and a separate launch check passes.
3. Use a short screen only for rejection; require matched longer evidence before promotion.
4. Test accumulation and clipping separately after the continuation decision.
5. Retain and independently rescore every epoch; do not select an undeclared transient checkpoint.

**Gate:** strong quality gain `>= 0.0005`, all four protected cells healthy, exact coverage, and a second seed or matched longer run before VESSL promotion. Weak or non-monotonic short results do not authorize a long official run.

---

### Task 5: Test the winner-style masked SSIM+L1 objective—not EXP034 again

**Objective:** Evaluate the loss family repeatedly reported by winners while preserving full-image quality and inference legality.

**Important distinction:** EXP034 tested a sparse evaluator-aligned foreground/bbox estimator and failed. The proposed probe is a simpler target-derived foreground-masked SSIM plus L1 regularizer, optionally with sqrt-area weighting; it must not reuse the failed objective under a new name.

**Likely files:**
- Modify: `utils/common/loss_function.py`
- Modify: `utils/learning/train_part.py`
- Modify: `utils/data/transforms.py`
- Test: `tests/test_score_aligned_training.py`
- Create: `tests/test_masked_ssim_l1.py`

**Steps:**
1. Reproduce the official foreground-mask construction in a training-only helper.
2. Add L1 as a normalization/regularization term; predeclare its weight from fixed training minibatch scale analysis, not leaderboard search.
3. Add optional sqrt-area weighting as a later, separate switch.
4. Prove finite gradients, empty-mask behavior, crop/window parity, multiple-box independence where boxes are used for diagnostics, and no inference import of training annotations.
5. Run a matched standard-loss control from the same source, seed, sampler order, optimizer, LR, and epoch budget.
6. Report full/bbox × acc4/acc8; reject a bbox gain paid for by a larger full-image regression.

**Gate:** all coverage checks pass and equal-acc quality beats the selected control by the preregistered threshold. Do not combine this first probe with MRAugment, AdamW, MoE, or architecture changes.

---

### Task 6: Start the next model family as a feasibility race

**Objective:** Find better quality per VRAM than vanilla E2E VarNet before spending a multi-day VESSL run.

**Selection order for this repository/deadline:**
1. Pin the MIT Feature/FI implementation at `facebookresearch/fastMRI@91f2df4711adbb6d643df1810f234e4abcf5881b` and reproduce its CPU smoke before adapting it.
2. Keep PromptMR+ at `934eeda6d4d18cd39e406fa1eee9e1f70603cb5e` blocked until written competition/submission confirmation under the Rutgers Non-commercial Research License.
3. Preserve upstream algorithm modules and build only thin repository data/mask/checkpoint/harness adapters and CPU schema tests.
4. Run the same largest-input GTX 1080 probe and estimate integration cost for every cleared family. Select by deployment fit and measured cost, not a fixed paper ranking.
5. No adjacent slices, historical features, or four-way experts in the first architecture comparison.

**Likely files:**
- Create: `utils/model/feature_varnet.py`
- Optionally create later: `utils/model/promptmr.py`
- Modify: `train.py` with an explicit model-family selector
- Modify: `utils/learning/train_part.py` through a narrow model factory
- Modify later: `utils/learning/test_part.py` only after inference design review
- Create: `tests/test_model_factory.py`
- Create: `tests/test_feature_varnet.py`
- Create later: `tests/test_promptmr.py`

**Feasibility sequence:**
1. Pin source, commit, representative config, and license.
2. Pass the upstream installation and CPU smoke unchanged; if dependencies block it, record the exact blocker rather than rewriting.
3. Add thin adapters, random-weight CPU shape tests, and exact checkpoint schema tests.
4. Run largest-input forward-only memory/runtime probes; reject configurations that cannot meet the official inference contract.
5. Run one-epoch LOCAL screens with two seeds against the selected vanilla baseline.
6. Advance only a seed-robust direction to matched e5, then e15/e30 or VESSL.
7. Keep the first comparison architecture-only: same loss, optimizer, schedule, data, seed, and evaluator.

**Stop rule:** if no cleared upstream family produces a seed-robust, component-safe signal within the bounded screen budget, stop the integration and return compute to EXP035. Do not rebuild E2E-VarNet upward or reproduce a full winner system blindly.

---

### Task 7: Audit masks first; treat MRAugment and MoE as conditional specialists

**Objective:** Use the strongest legal mask/augmentation ideas only when a measured bottleneck and oracle upper bound justify them.

**Existing prepared assets:**
- `scripts/analyze_kspace_masks.py`
- `tests/test_analyze_kspace_masks.py`
- `docs/mask_analysis_plan.md`
- `scripts/build_annotation_index.py`
- `docs/annotation_training_plan.md`
- `utils/learning/moe_router.py` and `docs/moe_routing_design.md` are untracked work and require independent review before use.

**Steps:**
1. First run synthetic tests and static inference-rule audits only.
2. Real train/val/public mask scanning requires explicit approval and a fresh, collision-free report namespace.
3. Establish center fraction, ACS width, equispaced period/offset, split drift, and held-out mask-family behavior using k-space-only legal features.
4. For MRAugment, start after the base model approaches convergence, not at epoch 0; use anatomy-aware transform bounds and preserve centroid distributions.
5. Do not apply broad random mask augmentation blindly. The winning reports are mixed: some rejected it, while others used staged fixed/random training or a separate random-mask helper.
6. Before MoE code is promoted, compute an offline, volume-disjoint oracle upper bound from frozen experts.
7. If complementarity is material, implement deterministic acc4/acc8 routing first. EXP054’s larger acc8 gain makes an acc8 specialist the first subgroup hypothesis.
8. Route only from legal k-space/mask structure; never use full filename, exact mask hash, image target, bbox, annotation, GRAPPA, or leaderboard outcome.
9. Load/select only one expert if necessary for the 8 GB contract, and charge all model computation to the official timed path.

**Gate:** routed final score must beat the best fixed model after actual timing, memory, and fallback costs. If the oracle upper bound is too small to clear those costs with margin, terminate MoE before implementation.

---

### Task 8: Average only credible same-basin checkpoints, then confirm and freeze

**Objective:** Extract no-inference-cost gains and produce one reproducible finalist plus one fallback.

**Steps:**
1. Do not reuse EXP048’s rejected exact-byte method gate or EXP040’s failed e5/e10 average as promotion evidence.
2. After a successful scheduled long run, screen EMA/SWA or adjacent retained-checkpoint averaging from the same basin.
3. Independently materialize and evaluate every averaged checkpoint; bind exact input checkpoint hashes and coefficients.
4. Do not use a two-forward output ensemble unless its quality gain exceeds the measured timing penalty and both component models are individually credible.
5. Confirm at most two finalists across seeds; require positive mean gain and wins in at least two of three seeds where the compute budget allows.
6. Profile largest-input 8 GB inference, package size, startup, and official-path timing.
7. Freeze model family, config, checkpoint hash, preprocessing, inference code, and environment by August 13–15.
8. Only after final freeze, separately approve and run the 30-repeat official timing cohort, package verification, and upload.

---

## Recommended calendar

- **Jul 15–16:** publish EXP035/AdamW consolidation, preserve the leader, and finish CPU-only continuation/source preflight. No GPU use while another process owns it.
- **Jul 16–20:** when separately cleared, run the matched fixed-LR versus LR `3e-4` continuation; in parallel reproduce the pinned Feature/FI CPU smoke in the existing LOCAL PyTorch environment.
- **Jul 20–24:** run the first cleared upstream adapter/schema and GTX 1080 feasibility gate; PromptMR+ remains blocked without written license confirmation.
- **Jul 22–31:** one matched longer run for the best recipe or next model family; no broad matrix.
- **Aug 1–8:** one winner-style loss/augmentation follow-up and, only if oracle evidence supports it, one acceleration-specialist probe; seed confirmation.
- **Aug 8–13:** no-cost averaging, finalist comparison, inference optimization, 8 GB proof, code/checkpoint freeze.
- **Aug 13–20:** approved official one-shot if still needed, final 30-run timing, fresh-clone package verification, and submission with receipt. Avoid starting a multi-day final training run after the freeze window.

## Explicit deprioritization / stop list

- No more `chans=18` capacity probes without fundamentally new evidence.
- No c9/c10/c12 launch before Task 3 proves training and inference feasibility.
- No more sparse EXP034-style score-aligned loss iterations before the simpler masked SSIM+L1 gate.
- No four-way brain/knee × acc4/acc8 MoE before a fixed-expert oracle study.
- No exact-mask or filename lookup routing.
- No output ensemble for a gain comparable to EXP047’s small improvement.
- No final 30-run timing before candidate freeze.
- No LOCAL checkpoint promotion.
- No automatic official evaluation.
- No merge of PR #19 as currently based.
- No merge of PR #24 as currently based.
- No AdamW rescue or full model-family reimplementation from E2E-VarNet primitives.

## One-sentence recommendation

Protect EXP035 epoch 30, use one matched fixed-LR-versus-lower-LR continuation to separate epoch and schedule effects, and integrate only pinned licensed upstream model families through thin adapters before considering masked loss, MRAugment, MoE, or ensembles.
