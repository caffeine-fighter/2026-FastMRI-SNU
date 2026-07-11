# Experiment plan

## Completed path

| Stage | Experiments | Outcome |
|---|---|---|
| Pipeline checks | EXP000, EXP001 | VESSL training and validation pipeline verified |
| Capacity sweep | EXP010, EXP011, EXP012, EXP013 | c4/ch12/s8 selected for longer training |
| Long training | EXP030 | Official candidate at 20 epochs |
| Duration follow-up | EXP031 | 30 epochs improved final validation quality |
| Official evaluation | EXP012, EXP030, EXP031 | EXP031 is the one-shot official leader; EXP030 has completed 30-run timing |

Full commands and metrics are in [`../experiments/experiment_log.csv`](../experiments/experiment_log.csv).

## EXP031 result

`EXP031_varnet_c4_ch12_s8_e30` continued EXP030 from epoch 20 through epoch 29 while keeping architecture, learning rate, and seed fixed.

- cascades: 4
- channels: 12
- sensitivity channels: 8
- learning rate: 0.001
- seed: 430
- best epoch: 27
- validation quality: `0.917440628180`
- improvement over EXP030 validation: `+0.002761542231`

EXP031 is the validation and one-shot official leader. Its official total score is `0.9162015104166666`, ahead of EXP030's finalized score by `+0.00095015625`.

## Next decision

1. Obtain separate approval for a 30-run EXP031 timing cohort.
2. Verify 30/30 runs and select the minimum valid timing.
3. Recompute the final EXP031 total score.
4. Replace and package EXP030 only if EXP031 remains ahead.

## Later ideas

- bbox-aware or foreground-weighted loss
- SSIM plus L1/Charbonnier loss
- limited seed search
- c6 timing/quality test

Each new experiment should change one variable and record its command, seed, metrics, checkpoint path, and decision in the experiment log.
