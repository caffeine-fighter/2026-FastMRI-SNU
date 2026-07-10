# Final Submission Checklist

## Selected candidate

- Experiment: `EXP030_varnet_c4_ch12_s8_e20`
- Architecture: VarNet, cascade `4`, channels `12`, sensitivity channels `8`
- Official total score: `0.9152513541666667`
- Official minimum timing: `173.4 ms/slice` over 30 runs
- Checkpoint SHA-256: `ef74bec4243e7aa39d5aa8dae031e1bb83e771c26be00c4e5330e17b60a66085`

## Organizer-required items

- [x] GitHub repository with detailed execution instructions in `README.md`
- [x] Loss graph: `reports/figures/EXP030_varnet_c4_ch12_s8_e20_val_loss.png`
- [x] Model weight exists at `/root/result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt`
- [x] Model description: `reports/phase2/EXP030_model_description.pptx`

The model weight is submitted separately and must never be committed to Git.

## Reproduction gate

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

## Final handoff

- [ ] Merge the final verified commit into the GitHub default branch
- [ ] Copy the four organizer-required items into the organizer's submission channel
- [ ] Verify the submitted repository commit ID and uploaded checkpoint hash
- [ ] Record the organizer receipt or submission identifier outside the repository

No organizer upload URL or authenticated submission CLI is contained in this
repository. The final external upload must therefore be completed through the
official organizer-provided channel.
