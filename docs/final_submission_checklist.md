# Final Submission Checklist

> The EXP030 entries below preserve the completed fallback delivery record. EXP030 is no longer the current one-shot leader. EXP035 epoch 30 leads at total `0.92146109375`, and the final prize candidate is not frozen yet.

## Final prize-candidate replacement gate

- [ ] Every submitted model component was trained end-to-end on VESSL from an allowed, code-reproducible initialization.
- [ ] No external data, externally supplied/file-loaded initialization weight, or LOCAL/RunPod learned state entered the final training lineage.
- [ ] The exact source, recipe, seed, dataset, checkpoint-generation, and environment manifests are frozen and reviewed.
- [ ] Any augmentation/remasking is train/validation-only, physics-consistent, deterministic, and covered by provenance.
- [ ] Any ensemble/TTA uses only eligible VESSL components and performs all reconstruction work inside timed `recon_slice()`.
- [ ] Official evaluator bytes, expected files, slice counts, shapes, dtypes, finite outputs, and archive integrity pass.
- [ ] The final 30-repeat timing cohort is complete for the frozen candidate.
- [ ] Submission receipt, artifact SHA-256, returned score, Git commit, and timestamp are recorded.

## Active selected candidate

- Experiment: `TBD — not frozen`
- Protected one-shot leader: `EXP035_varnet_c8_ch12_s8_e30`, epoch 30, total `0.92146109375`
- Verified fallback: `EXP030_varnet_c4_ch12_s8_e20`
- Final package and upload: not authorized by candidate selection until every replacement gate above passes

## Active final handoff

- [ ] Freeze one final candidate and one rollback candidate.
- [ ] Record the exact repository commit, checkpoint generation/SHA-256, model contract, environment, and evaluator hashes.
- [ ] Verify a fresh clone and separately supplied checkpoint reproduce the official path.
- [ ] Copy the organizer-required repository, model description, loss graph, and model weight through the official channel.
- [ ] Verify the organizer-received commit and checkpoint SHA-256.
- [ ] Record the submission identifier and receipt.

## Historical EXP030 fallback record

- Experiment: `EXP030_varnet_c4_ch12_s8_e20`
- Architecture: VarNet, cascade `4`, channels `12`, sensitivity channels `8`
- Official total score: `0.9152513541666667`
- Official minimum timing: `173.4 ms/slice` over 30 runs
- Checkpoint SHA-256: `ef74bec4243e7aa39d5aa8dae031e1bb83e771c26be00c4e5330e17b60a66085`

### Delivery status

- GitHub repository delivery: complete
- Default branch: `baseline/2026-baby-varnet`
- Submission implementation commit: `fbbddf6700cd65b1e2b52c1c6418f48a5eef9b82`
- Fresh-clone verification: passed
- External organizer upload: pending

### Organizer items prepared

- [x] GitHub repository with detailed execution instructions in `README.md`
- [x] Loss graph: `reports/figures/EXP030_varnet_c4_ch12_s8_e20_val_loss.png`
- [x] Model weight exists at `/root/result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt`
- [x] Model description: `reports/phase2/EXP030_model_description.pptx`

The model weight is submitted separately and must never be committed to Git.

### Reproduction record

- [x] `recon_eval.py` is unchanged
- [x] `recon_eval.sh` defaults to cascade `4`, channels `12`, sensitivity channels `8`
- [x] Official invocation is `bash recon_eval.sh`
- [x] `scripts/phase2_preflight.sh` passed before evaluation
- [x] All 30 official repeats completed
- [x] `reports/phase2/final_score_summary.md` records the selected score

## Git safety gate

Before pushing or submitting:

```bash
python scripts/check_submission.py
git diff --cached --name-only
git status --ignored -sb
```

Confirm that none of the following are tracked or staged:

- `Data/`, `data/`, or mounted leaderboard data
- `*.h5`
- `*.pt`, `*.pth`, or `*.ckpt`
- `result/`, `results/`, `runs/`, `checkpoints/`, or `checkpoints_phase2/`
- `.env`, `*.env`, secrets, credentials, or private keys

## Historical EXP030 handoff record

- [x] Merge the verified submission implementation into the GitHub default branch
- [x] Verify the GitHub default and feature branches contain commit `fbbddf6`
- [x] Verify a fresh clone passes the submission-safety and required-artifact checks
- [x] Prepare `/root/submissions/EXP030_final_fbbddf6.zip`
- [x] Verify the local package checkpoint hash against the selected model
- [ ] Copy the four organizer-required items into the organizer's submission channel
- [ ] Verify the organizer-received repository commit and uploaded checkpoint hash
- [ ] Record the organizer receipt or submission identifier outside the repository

No organizer upload URL or authenticated submission CLI is contained in this
repository. The final external upload must therefore be completed through the
official organizer-provided channel.
