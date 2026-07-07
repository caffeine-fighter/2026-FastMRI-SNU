# Decisions

## 2026-07-06

- Use official 2026_baby_varnet as the primary baseline.
- Treat VESSL as the source of truth for final candidate model training.
- Keep raw data, H5 files, checkpoints, and result artifacts out of Git.
- Add local scripts while VESSL is blocked by resource quota exceeded.
- Reuse official metric helpers for SSIM_full and SSIM_bbox.

## 2026-07-07 EXP000 smoke result

EXP000_smoke_varnet_c1_ch9_s4_e1 completed successfully on VESSL.

Metrics:
- overall SSIM_full: 0.8555344924914702
- overall SSIM_bbox: 0.8488467528212885
- acc4 SSIM_full: 0.8752739521736594
- acc4 SSIM_bbox: 0.8550138779889758
- acc8 SSIM_full: 0.8346127214220663
- acc8 SSIM_bbox: 0.8366267085075378
- volumes: 30
- slices: 791
- bbox annotations: 161
- skipped: []

Conclusion:
The VESSL data mount, training loop, checkpoint saving, validation reconstruction saving, metric script, and loss plot script are working. Proceed to EXP001 5-epoch baseline.

## 2026-07-07 EXP001 baseline result

EXP001_baseline_varnet_c1_ch9_s4_e5 completed successfully on VESSL.

Metrics:
- val_loss: 3.742339516087377
- overall SSIM_full: 0.8775020848333308
- overall SSIM_bbox: 0.8734674827652689
- acc4 SSIM_full: 0.895139145763266
- acc4 SSIM_bbox: 0.8818787876690659
- acc8 SSIM_full: 0.8588086374414464
- acc8 SSIM_bbox: 0.8568006378633005
- volumes: 30
- slices: 791
- bbox annotations: 161

Conclusion:
EXP001 is the initial 5-epoch baseline. Use this as the reference for the first capacity sweep. Next experiment: EXP010, cascade 2 with chans 9 and sens_chans 4.
