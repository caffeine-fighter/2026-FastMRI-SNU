# Phase 2 Plan

## Official leaderboard path

The Phase 2 leaderboard submission must be runnable through one bash entrypoint:

```bash
bash recon_eval.sh
```

`recon_eval.sh` is therefore the official leaderboard submission path. Keep it minimal, reproducible, and candidate-driven.

## Final Score formula

```text
Final Score = 0.5 * SSIM_full + 0.5 * SSIM_bbox + Tiebreaker
```

For validation-only comparisons before leaderboard timing:

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
```

## Tiebreaker formula and ms/slice rule

Tiebreaker is based on `ms/slice` reconstruction speed:

```text
if ms_per_slice <= 80:
    time_score = 0.001
elif ms_per_slice >= 2000:
    time_score = 0.0
else:
    time_score = 0.001 * (2000 - ms_per_slice) / (2000 - 80)
```

So:

- `<= 80 ms/slice` receives the full `0.001` tiebreaker.
- `>= 2000 ms/slice` receives `0`.
- Values between 80 and 2000 ms/slice are linearly interpolated.

The helper `scripts/phase2_score.py` implements this formula and reports:

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
time_score    = tiebreaker(ms_per_slice)
total_score   = quality_score + time_score
```

## Hard rules

- Do **not** modify `recon_eval.py`.
- If custom model support is needed, use `utils/learning/test_part.py`.
- `recon_eval.sh` is the official leaderboard submission path.
- Do **not** use image fields, bbox annotations, or given GRAPPA during inference.
- Do **not** modify mounted `Data` directories.
- Do **not** use `LOCAL_` weights as final candidates.
- Do **not** run `recon_eval.sh` while VESSL training is active.

## Wrapper/helper files

- `recon_eval.py`: official evaluator/wrapper code; must remain unmodified.
- `recon_eval.sh`: official single-entry submission script.
- `utils/learning/test_part.py`: place for custom model loading/inference support if needed.
- `scripts/set_phase2_candidate.sh`: sets the selected candidate checkpoint/config using a symlink; does not copy weights.
- `scripts/phase2_preflight.sh`: validates wrapper prerequisites before evaluation.
- `scripts/run_recon_eval_once.sh`: runs the official `bash recon_eval.sh` once and saves logs/score files.
- `scripts/repeat_recon_eval.sh`: repeats official evaluation N times and summarizes best timing/score.
- `scripts/phase2_score.py`: parses logs and computes quality/time/total scores.

## 30-run minimum-time rule for final submission

For final submission timing/tiebreaker selection, run at least 30 repetitions of the official wrapper for the chosen candidate and use the minimum valid `ms/slice` result when checking the timing tiebreaker.

Template after candidate selection, preflight, and user approval:

```bash
bash scripts/repeat_recon_eval.sh EXP030_phase2 30
```

Do not run this until:

1. VESSL training is complete.
2. Candidate checkpoint is selected.
3. Mounted leaderboard `Data` exists.
4. `scripts/phase2_preflight.sh` passes.
5. The user approves official Phase 2 evaluation.

## Mounted Data policy

Mounted `Data` directories are read-only inputs. Do not create, delete, rename, normalize, cache into, or otherwise modify anything under mounted `Data`.

Expected leaderboard data root is detected by preflight from one of:

```text
$FASTMRI_DATA_ROOT
/root/Data
/home/ubuntu/Data
$HOME/fastmri_data/Data
```

On VESSL, Phase 2 requires the mounted leaderboard k-space layout before `recon_eval.sh` is run.

## Candidate decision flow

1. Wait for `EXP030` to finish.
2. Evaluate and log `EXP030` validation metrics.
3. Compare `EXP030` vs `EXP012` by validation quality and expected runtime.
4. Set the better candidate with `scripts/set_phase2_candidate.sh`.
5. Run `scripts/phase2_preflight.sh`.
6. Run one official wrapper evaluation if approved.
7. Run the 30-repeat timing suite if the wrapper is correct.
8. Review `reports/phase2/repeat_*` before final submission.
