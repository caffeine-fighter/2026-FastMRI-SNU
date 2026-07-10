# Experiment plan

## Completed path

| Stage | Experiments | Outcome |
|---|---|---|
| Pipeline checks | EXP000, EXP001 | VESSL training and validation pipeline verified |
| Capacity sweep | EXP010, EXP011, EXP012, EXP013 | c4/ch12/s8 selected for longer training |
| Long training | EXP030 | Best validated and official candidate at 20 epochs |
| Official evaluation | EXP012 vs EXP030 | EXP030 selected; 30-run timing complete |

Full commands and metrics are in [`../experiments/experiment_log.csv`](../experiments/experiment_log.csv).

## Active experiment

`EXP031_varnet_c4_ch12_s8_e30` continues the EXP030 model from epoch 20 through epoch 29.

It changes training duration only:

- cascades: 4
- channels: 12
- sensitivity channels: 8
- learning rate: 0.001
- seed: 430
- target total epochs: 30

The root [`README.md`](../README.md) shows live progress.

## Decision after EXP031

1. Confirm a clean epoch-29 completion.
2. Evaluate the final best validation reconstructions.
3. Compare final `SSIM_full`, `SSIM_bbox`, and quality with EXP030.
4. Keep EXP030 unless the gain remains clear.
5. Run official evaluation only after approval.

## Later ideas

Do not launch these while EXP031 is active:

- bbox-aware or foreground-weighted loss
- SSIM plus L1/Charbonnier loss
- limited seed search
- c6 timing/quality test

Each new experiment should change one variable and record its command, seed, metrics, checkpoint path, and decision in the experiment log.
