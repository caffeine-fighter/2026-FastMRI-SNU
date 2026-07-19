# Phase 2 rules

## Score

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
final_score   = quality_score + time_score
```

Time score:

```text
<= 80 ms/slice    -> 0.001
>= 2000 ms/slice  -> 0
80..2000          -> linear interpolation
```

`scripts/phase2_score.py` implements the formula. Final timing uses the minimum valid `ms/slice` from at least 30 official runs.

## Official path

```bash
bash recon_eval.sh
```

Before any official run:

1. Training is finished and the GPU is idle.
2. The candidate checkpoint and architecture are fixed.
3. Mounted leaderboard data is present and read-only.
4. `scripts/phase2_preflight.sh` passes.
5. The standing authorization record is valid for this exact run and the candidate passes all eligibility gates.

## Candidate flow

1. Train on VESSL.
2. Evaluate validation reconstructions with `scripts/evaluate_val.py`.
3. Compare `SSIM_full`, `SSIM_bbox`, quality, and expected runtime.
4. Set the candidate with `scripts/set_phase2_candidate.sh`.
5. Run one authorized official evaluation for an eligible VESSL end-to-end candidate.
6. Repeat 30 times only after the final candidate is frozen.
7. Record the selected score in `reports/phase2/`.

## Fixed rules

- Do not modify `recon_eval.py`.
- Put custom model loading in `utils/learning/test_part.py`.
- Do not use image fields, bounding-box annotations, or provided GRAPPA during inference.
- Do not modify mounted `Data`.
- Do not use `LOCAL_` checkpoints as official candidates.
- Do not import LOCAL/RunPod model, optimizer, scheduler, scaler, EMA/SWA, RNG, teacher-output, or reconstruction-cache state into a final candidate.
- If ensemble/TTA is used, every component must be independently VESSL end-to-end eligible and all reconstruction computation must occur inside timed `recon_slice()`.
- Do not run official evaluation during training.

## Helper scripts

| Script | Purpose |
|---|---|
| `scripts/set_phase2_candidate.sh` | Select checkpoint and architecture |
| `scripts/phase2_preflight.sh` | Validate paths, data, config, and wrapper state |
| `scripts/run_recon_eval_once.sh` | Run and record one official evaluation |
| `scripts/repeat_recon_eval.sh` | Run the approved timing cohort |
| `scripts/phase2_score.py` | Parse output and calculate scores |

EXP030 has already completed its 30-run official evaluation. Do not repeat it unless the result is invalidated.
