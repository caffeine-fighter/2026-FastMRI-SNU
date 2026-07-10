# Archived Phase 2 issue plan

> Historical draft. Phase 2 evaluation and GitHub delivery are complete. Do not use these issue bodies as current instructions.

No GitHub issues were created from this plan.

## Labels

Use these labels consistently:

### Type

- `type:experiment`
- `type:automation`
- `type:bug`
- `type:docs`
- `type:submission`

### Area

- `area:vessl`
- `area:desktop`
- `area:phase2`
- `area:metrics`
- `area:recon_eval`

### Priority

- `priority:P0`
- `priority:P1`
- `priority:P2`

### Status

- `status:blocked`
- `status:running`
- `status:needs-review`
- `status:done`

### Risk

- `risk:oom`
- `risk:timing`
- `risk:rules`

## Milestones

- **Phase 2 official evaluation** — wrapper, preflight, official one-shot runs, timing, and score selection.
- **Model selection** — EXP012/EXP030 comparison, local-probe interpretation, and any follow-up official candidates.
- **Final submission package** — final README, checklist, artifacts, and explanation materials.

## Initial issues to create later

Do not create these issues until explicitly approved. Bodies below are copy-paste drafts.

### P0 / Phase 2

#### 1. `[P0] Verify VESSL leaderboard Data mount`

- **Labels:** `type:submission`, `area:vessl`, `area:phase2`, `priority:P0`, `status:blocked`, `risk:rules`
- **Milestone:** Phase 2 official evaluation
- **Body:** Verify that VESSL has the official Phase 2 leaderboard `Data` mount available in the expected read-only layout before running any official evaluation.
- **Acceptance criteria:**
  - Leaderboard root exists.
  - Expected `acc4` and `acc8` k-space files are visible.
  - No command writes into mounted `Data`.
  - Result is documented in the run log.
- **Blocking dependencies:** EXP030 must finish; user must approve official evaluation checks.
- **Commands:**
  ```bash
  cd /root/2026-FastMRI-SNU
  test -d /root/Data/leaderboard
  find /root/Data/leaderboard -maxdepth 4 -type d | sort | sed -n '1,120p'
  find /root/Data/leaderboard -type f -name '*.h5' | wc -l
  ```
- **Safety constraints:** read-only inspection only; do not modify `Data/`, `.h5`, or result artifacts.

#### 2. `[P0] Merge Phase 2 recon_eval wrapper into VESSL after EXP030 finishes`

- **Labels:** `type:automation`, `area:vessl`, `area:phase2`, `area:recon_eval`, `priority:P0`, `status:blocked`, `risk:rules`
- **Milestone:** Phase 2 official evaluation
- **Body:** Bring the reviewed `phase2/eval-wrapper` automation onto VESSL only after EXP030 finishes and validation metrics are recorded.
- **Acceptance criteria:**
  - EXP030 process is no longer running.
  - EXP030 metrics are recorded before merge.
  - `recon_eval.py` remains unmodified.
  - `scripts/phase2_preflight.sh` passes after merge.
- **Blocking dependencies:** EXP030 training completion; EXP030 validation issue; clean VESSL workspace.
- **Commands:** See `vessl_after_exp030_runbook.md`; do not run merge commands while EXP030 is active.
- **Safety constraints:** no merge while training is active; do not stage checkpoints, `Data`, `.h5`, `.pt`, `.pth`, `.ckpt`, `.env`, or secrets.

#### 3. `[P0] Run official recon_eval for EXP012`

- **Labels:** `type:submission`, `area:vessl`, `area:phase2`, `area:recon_eval`, `priority:P0`, `status:blocked`, `risk:timing`, `risk:rules`
- **Milestone:** Phase 2 official evaluation
- **Body:** Run official one-shot Phase 2 wrapper for the current best validation candidate `EXP012_varnet_c4_ch12_s4_e10`.
- **Acceptance criteria:**
  - Candidate checkpoint exists.
  - `phase2_preflight.sh` passes.
  - Wrapper calls official `bash recon_eval.sh`.
  - Score/log artifacts are saved under `reports/phase2/`.
- **Blocking dependencies:** Phase 2 wrapper merged; leaderboard Data verified; GPU idle; user approval.
- **Commands:**
  ```bash
  bash scripts/set_phase2_candidate.sh EXP012 ../result/EXP012_varnet_c4_ch12_s4_e10/checkpoints/best_model.pt 4 12 4 'EXP012 completed reference: quality_score=0.9090841340270383'
  bash scripts/phase2_preflight.sh
  bash scripts/run_recon_eval_once.sh EXP012_phase2_once
  ```
- **Safety constraints:** do not modify `recon_eval.py`; do not run repeat-30 until one-shot succeeds.

#### 4. `[P0] Run official recon_eval for EXP030`

- **Labels:** `type:submission`, `area:vessl`, `area:phase2`, `area:recon_eval`, `priority:P0`, `status:blocked`, `risk:timing`, `risk:rules`
- **Milestone:** Phase 2 official evaluation
- **Body:** Run official one-shot Phase 2 wrapper for `EXP030_varnet_c4_ch12_s8_e20` after its validation metrics are recorded.
- **Acceptance criteria:**
  - EXP030 checkpoint exists.
  - EXP030 validation metrics are recorded.
  - `phase2_preflight.sh` passes.
  - One-shot official score/log artifacts are saved.
- **Blocking dependencies:** EXP030 training completion; EXP030 validation comparison; Phase 2 wrapper merged; leaderboard Data verified; user approval.
- **Commands:**
  ```bash
  bash scripts/set_phase2_candidate.sh EXP030 ../result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt 4 12 8 'EXP030 candidate after completed VESSL training'
  bash scripts/phase2_preflight.sh
  bash scripts/run_recon_eval_once.sh EXP030_phase2_once
  ```
- **Safety constraints:** official path must remain `bash recon_eval.sh`; do not use image fields, bbox annotations, or given GRAPPA during inference.

#### 5. `[P0] Run 30-repeat timing for selected final candidate`

- **Labels:** `type:submission`, `area:vessl`, `area:phase2`, `area:metrics`, `priority:P0`, `status:blocked`, `risk:timing`
- **Milestone:** Phase 2 official evaluation
- **Body:** After one-shot official runs succeed, repeat the selected final candidate 30 times and use the minimum valid `ms/slice` result for final timing.
- **Acceptance criteria:**
  - Candidate selected from official one-shot comparison.
  - 30 valid official wrapper runs complete.
  - Summary identifies minimum `ms/slice` and best total score.
  - Output is reviewed before final submission.
- **Blocking dependencies:** official one-shot EXP012/EXP030 results; candidate decision; user approval.
- **Commands:**
  ```bash
  bash scripts/repeat_recon_eval.sh <FINAL_CANDIDATE_TAG> 30
  ```
- **Safety constraints:** run only when GPU is idle; do not edit model/eval code during timing.

#### 6. `[P0] Confirm recon_eval.py is unmodified`

- **Labels:** `type:submission`, `area:phase2`, `area:recon_eval`, `priority:P0`, `status:needs-review`, `risk:rules`
- **Milestone:** Phase 2 official evaluation
- **Body:** Confirm that `recon_eval.py` has no local diff and matches the official Phase 2 evaluator when upstream reference is available.
- **Acceptance criteria:**
  - `git diff -- recon_eval.py` is empty.
  - `git diff --cached -- recon_eval.py` is empty.
  - Upstream byte comparison is documented if upstream Phase 2 reference is available.
- **Blocking dependencies:** upstream Phase 2 reference availability for byte comparison.
- **Commands:**
  ```bash
  git diff -- recon_eval.py
  git diff --cached -- recon_eval.py
  ```
- **Safety constraints:** do not modify `recon_eval.py`.

### P1 / Model selection

#### 7. `[P1] Evaluate EXP030 validation metrics and compare against EXP012`

- **Labels:** `type:experiment`, `area:vessl`, `area:metrics`, `priority:P1`, `status:blocked`
- **Milestone:** Model selection
- **Body:** After EXP030 training finishes, compute validation metrics and compare against EXP012 quality `0.9090841340270383`.
- **Acceptance criteria:**
  - `metrics.csv` and `skipped.json` are printed/reviewed.
  - `SSIM_full`, `SSIM_bbox`, `quality_score`, and `val_loss` are recorded.
  - Delta vs EXP012 is documented.
- **Blocking dependencies:** EXP030 training completion.
- **Commands:** See validation section in `vessl_after_exp030_runbook.md`.
- **Safety constraints:** no `recon_eval.sh`; no training restart; no `Data` modification.

#### 8. `[P1] Decide EXP012 vs EXP030 official candidate`

- **Labels:** `type:experiment`, `area:phase2`, `area:metrics`, `priority:P1`, `status:blocked`, `risk:timing`
- **Milestone:** Model selection
- **Body:** Use validation and official one-shot results to decide whether EXP012 or EXP030 should enter final repeat-30 timing.
- **Acceptance criteria:**
  - Validation quality compared.
  - Official one-shot `SSIM_full`, `SSIM_bbox`, `ms/slice`, and `total_score` compared.
  - Candidate decision recorded with rationale.
- **Blocking dependencies:** EXP030 validation; official one-shot EXP012; official one-shot EXP030.
- **Commands:** Use `reports/phase2/*/score.json` and `scripts/phase2_score.py` outputs.
- **Safety constraints:** choose from official VESSL candidates only; do not use `LOCAL_` checkpoints as final.

#### 9. `[P1] Analyze LOCAL c6/ch12/s8 vs c4/ch12/s8 timing risk`

- **Labels:** `type:experiment`, `area:desktop`, `area:metrics`, `priority:P1`, `risk:timing`
- **Milestone:** Model selection
- **Body:** Interpret desktop local probes where `LOCAL_EXP014 c6/ch12/s8` beats `LOCAL_EXP013 c4/ch12/s8` by only about `0.00078` quality.
- **Acceptance criteria:**
  - Local quality delta documented.
  - Max tiebreaker `0.001` considered.
  - Recommendation recorded for whether to launch any c6 VESSL run.
- **Blocking dependencies:** EXP030 official/validation results.
- **Commands:** Read `reports/local_comparisons/local_probe_summary.md` and `next_experiment_decision.md`.
- **Safety constraints:** local analysis only; no training without explicit approval.

#### 10. `[P1] Consider seed search for best official candidate`

- **Labels:** `type:experiment`, `area:vessl`, `priority:P1`, `status:blocked`, `risk:oom`
- **Milestone:** Model selection
- **Body:** If EXP012/EXP030 official results are close, consider a limited seed search on the selected architecture.
- **Acceptance criteria:**
  - Justification based on official score gap.
  - Run count and seeds specified.
  - Resource/risk review completed.
- **Blocking dependencies:** EXP012 vs EXP030 decision; available VESSL time; user approval.
- **Commands:** TBD after candidate decision.
- **Safety constraints:** do not start training until explicitly approved.

### P1 / Automation

#### 11. `[P1] Harden phase2_score.py parser against output format changes`

- **Labels:** `type:automation`, `area:phase2`, `area:metrics`, `priority:P1`, `risk:rules`
- **Milestone:** Phase 2 official evaluation
- **Body:** Make `scripts/phase2_score.py` robust to small variations in official evaluator output format.
- **Acceptance criteria:**
  - Parser handles known log variants.
  - Unit/smoke fixtures cover `SSIM_full`, `SSIM_bbox`, and `ms/slice` extraction.
  - `python -m py_compile scripts/phase2_score.py` passes.
- **Blocking dependencies:** At least one official one-shot log sample.
- **Commands:**
  ```bash
  python -m py_compile scripts/phase2_score.py
  ```
- **Safety constraints:** no official eval run as part of parser-only edit unless separately approved.

#### 12. `[P1] Add final_score_summary.md generation`

- **Labels:** `type:automation`, `area:phase2`, `area:metrics`, `priority:P1`
- **Milestone:** Final submission package
- **Body:** Generate a Markdown summary of final candidate score, timing, selected checkpoint, and submission rationale.
- **Acceptance criteria:**
  - Summary includes `SSIM_full`, `SSIM_bbox`, `ms/slice`, time score, total score.
  - Summary links to run logs and score JSON.
  - No checkpoint/data files are staged.
- **Blocking dependencies:** official one-shot and/or repeat-30 results.
- **Commands:** TBD script or report generation command.
- **Safety constraints:** report generation only; no model/evaluator modifications.

#### 13. `[P1] Add VESSL runbook sanity checks`

- **Labels:** `type:automation`, `area:vessl`, `area:phase2`, `priority:P1`, `risk:rules`
- **Milestone:** Phase 2 official evaluation
- **Body:** Add additional read-only sanity checks to VESSL runbook before official eval.
- **Acceptance criteria:**
  - Checks cover process status, GPU idle, Data mount, candidate symlink, and `recon_eval.py` diff.
  - Commands are copy-paste safe.
  - No destructive command is included.
- **Blocking dependencies:** current runbook review.
- **Commands:** Update docs only unless a separate script is approved.
- **Safety constraints:** no git merge/eval/training while EXP030 is running.

### P2 / Documentation

#### 14. `[P2] Final README reproduction guide`

- **Labels:** `type:docs`, `area:phase2`, `priority:P2`
- **Milestone:** Final submission package
- **Body:** Finalize README reproduction instructions after final candidate selection.
- **Acceptance criteria:**
  - README reflects final candidate and official results.
  - Reproduction commands are concise and accurate.
  - Forbidden artifact rules remain visible.
- **Blocking dependencies:** final candidate selection.
- **Commands:** Documentation edit only.
- **Safety constraints:** do not include secrets or paths that imply committing artifacts.

#### 15. `[P2] Final submission checklist`

- **Labels:** `type:docs`, `type:submission`, `area:phase2`, `priority:P2`, `risk:rules`
- **Milestone:** Final submission package
- **Body:** Build a final checklist for code, README, metrics, run logs, allowed artifacts, and forbidden files.
- **Acceptance criteria:**
  - Checklist covers files to include/exclude.
  - Checklist includes final `check_submission.py` run.
  - Checklist includes `git status` review.
- **Blocking dependencies:** final candidate and official score selected.
- **Commands:**
  ```bash
  python scripts/check_submission.py
  git status --ignored
  ```
- **Safety constraints:** never stage data/checkpoints/results/secrets.

#### 16. `[P2] Video explanation outline`

- **Labels:** `type:docs`, `type:submission`, `priority:P2`
- **Milestone:** Final submission package
- **Body:** Draft the final video explanation outline for model choice, Phase 2 scoring, runtime tradeoffs, and reproducibility.
- **Acceptance criteria:**
  - Outline includes model architecture summary.
  - Includes EXP012/EXP030 decision rationale.
  - Includes official scoring/timing summary.
  - Includes reproducibility and safety notes.
- **Blocking dependencies:** final candidate selection and official score summary.
- **Commands:** Documentation only.
- **Safety constraints:** no confidential paths, secrets, or unavailable claims.
