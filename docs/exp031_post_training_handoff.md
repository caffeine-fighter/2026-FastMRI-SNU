# EXP031 Post-Training Handoff and Decision Gate

This document is a desktop control-plane checklist. It does not authorize local official evaluation, modify the official metric pipeline, or promote a `LOCAL_` checkpoint.

## Machine boundaries

- **VESSL:** EXP031 training, validation, any official `recon_eval.sh`, and official candidate reproduction.
- **Desktop-4070Ti:** LOCAL probes, report parsing, consistency checks, documentation, and Git safety review.
- **Laptop:** operator interface.

Do not merge the desktop local-probe branch into the default/VESSL branch until the EXP031 handoff is complete and both sides have been reviewed.

## Required EXP031 evidence from VESSL

Before comparing EXP031 with EXP030, capture all of the following from files rather than chat memory:

- [ ] Exact `EXP031_...` experiment name.
- [ ] Branch and commit used for training.
- [ ] Training command and seed.
- [ ] Checkpoint path and SHA-256.
- [ ] Requested epoch count and selected best epoch.
- [ ] Best/final validation loss.
- [ ] Validation `SSIM_full`.
- [ ] Validation `SSIM_bbox`.
- [ ] Derived validation quality: `(SSIM_full + SSIM_bbox) / 2`.
- [ ] Acceleration-4 full and bbox metrics.
- [ ] Acceleration-8 full and bbox metrics.
- [ ] Volume, slice, and bbox-annotation counts.
- [ ] `skipped=[]`, or a reviewed explanation for every skipped item.
- [ ] Confirmation that no training process remains before official evaluation.
- [ ] Confirmation that the selected checkpoint is EXP031, not the old EXP030 symlink target.

The VESSL handoff must commit `reports/phase2/EXP031_validation_handoff.json`. The fail-closed report builder requires this file in addition to the `EXP031` row in `experiments/experiment_log.csv`, and cross-checks them. Required JSON fields are:

- experiment identity, status, branch, commit, command, seed, and epoch count;
- best epoch and validation loss;
- checkpoint path and 64-character SHA-256;
- c4/ch12/s8 configuration;
- `overall`, `acc4`, and `acc8` full/bbox metrics plus their volume, slice, and bbox counts;
- `skipped: []`.

## Official learning-curve comparison

Compare the same c4/ch12/s8 family:

| Run | Epochs | Role |
|---|---:|---|
| EXP013 | 10 | official VESSL validation reference |
| EXP030 | 20 | current official candidate reference |
| EXP031 | 30 | active continuation; pending file-backed result |

Required calculations:

- `EXP030 quality - EXP013 quality`
- `EXP031 quality - EXP030 quality`
- Full-image and bbox deltas separately
- Acceleration-4 and acceleration-8 quality deltas separately
- Validation-loss change and best-epoch location

Do not call the 30-repeat timing result a 30-epoch model. The existing official 30-repeat report used the EXP030 20-epoch checkpoint.

## LOCAL candidate comparison

The adaptive desktop campaign compares c4/ch16/s8 against c4/ch12/s8:

| Evidence | Baseline | Candidate |
|---|---|---|
| one epoch, seed 430 | LOCAL_EXP013 | LOCAL_EXP018 |
| five epochs, seed 430 | LOCAL_EXP029 | LOCAL_EXP032 |
| one epoch, seed 431 | LOCAL_EXP033 | LOCAL_EXP034 |
| ten-epoch trajectory | — | LOCAL_EXP035 |

Every comparison must use the local validation evaluator and must remain labeled exploratory.

## Proposal gate for c4/ch16/s8

A future official VESSL experiment may be proposed only if all checks pass:

- [ ] LOCAL_EXP032 beats LOCAL_EXP029 at five epochs.
- [ ] The five-epoch gain is not created by an unacceptable regression greater than `0.001` in either full-image or bbox SSIM.
- [ ] LOCAL_EXP034 beats LOCAL_EXP033 at seed 431.
- [ ] LOCAL_EXP035 quality does not fall more than `0.001` below LOCAL_EXP032.
- [ ] All required LOCAL evaluations contain 30 volumes, 791 slices, 161 bbox annotations, and no skipped files.
- [ ] c4/ch16/s8 fits the intended VESSL GPU memory budget.
- [ ] Expected official timing is acceptable; local `ForwardTime` is not used as official timing.
- [ ] EXP031’s result does not remove the practical value of another official architecture run.

Passing this gate permits only a proposal. It does not authorize VESSL training, official evaluation, submission, or use of a LOCAL checkpoint.

## Automated fail-closed report

Use:

```bash
source /home/ray1001/.venvs/fastmri/bin/activate
python scripts/build_exp031_decision_report.py --check
```

Expected behavior while sources are incomplete:

- Exit code `3`.
- A list of pending official and LOCAL sources.
- No final decision report is written.

When EXP031 and all adaptive LOCAL metrics are present:

```bash
python scripts/build_exp031_decision_report.py --write
```

Generated only when every required source is valid:

- `reports/local_comparisons/exp031_candidate_decision.json`
- `reports/local_comparisons/exp031_candidate_decision.md`

The generator exits nonzero and writes no final report when a source is missing or invalid.

## Git handoff safety

Before staging any report:

```bash
python scripts/check_submission.py
git diff --check
git diff --name-only -- \
  train.py recon_eval.py reconstruct.py \
  utils/model utils/data utils/learning
git status -sb
```

Stage only an exact allowlist of safe CSV, JSON, Markdown, and PNG report artifacts. Never stage data, H5 reconstructions, checkpoints, model weights, loss arrays, environment files, or secrets.

Re-fetch before pushing. Push only the isolated desktop branch, verify its remote SHA in a fresh detached worktree, and do not merge while the VESSL handoff remains active.
