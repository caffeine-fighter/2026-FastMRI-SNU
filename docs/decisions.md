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

## 2026-07-07 EXP010 cascade 2 result

EXP010_varnet_c2_ch9_s4_e10 completed successfully on VESSL.

Metrics:
- val_loss: 3.3869223552422363
- overall SSIM_full: 0.8946281794350307
- overall SSIM_bbox: 0.9089210673889018
- acc4 SSIM_full: 0.9110714525204033
- acc4 SSIM_bbox: 0.9182333801394311
- acc8 SSIM_full: 0.8772000228054821
- acc8 SSIM_bbox: 0.8904688921239641
- volumes: 30
- slices: 791
- bbox annotations: 161

Conclusion:
Compare against EXP001. If overall SSIM_bbox and acc8 SSIM_bbox improve, continue the cascade sweep with EXP011 cascade=4. If metrics degrade, consider longer training for cascade=2 or learning-rate adjustment before increasing model size.

## 2026-07-07 EXP010 cascade 2 result

EXP010_varnet_c2_ch9_s4_e10 completed successfully on VESSL.

Metrics:
- overall SSIM_full: 0.8946281794350307
- overall SSIM_bbox: 0.9089210673889018
- acc4 SSIM_full: 0.9110714525204033
- acc4 SSIM_bbox: 0.9182333801394311
- acc8 SSIM_full: 0.8772000228054821
- acc8 SSIM_bbox: 0.8904688921239641
- volumes: 30
- slices: 791
- bbox annotations: 161
- skipped: []

Comparison against EXP001:
- overall SSIM_full improved by +0.0171260946
- overall SSIM_bbox improved by +0.0354535846
- acc4 SSIM_bbox improved by +0.0363545925
- acc8 SSIM_bbox improved by +0.0336682543

Conclusion:
Cascade 2 is a clear improvement over the 5-epoch cascade 1 baseline. Continue cascade sweep with EXP011: cascade=4, chans=9, sens_chans=4, epochs=10.

## 2026-07-07 EXP011 cascade 4 result

EXP011_varnet_c4_ch9_s4_e10 completed successfully on VESSL.

Metrics:
- val_loss: 3.2896190929437226
- overall SSIM_full: 0.8994653502999606
- overall SSIM_bbox: 0.9182903899909547
- acc4 SSIM_full: 0.9153357927277868
- acc4 SSIM_bbox: 0.9281987882106104
- acc8 SSIM_full: 0.8826443344975511
- acc8 SSIM_bbox: 0.8986570824075628
- volumes: 30
- slices: 791
- bbox annotations: 161

Comparison against EXP010:
- overall SSIM_full delta: +0.0048371709
- overall SSIM_bbox delta: +0.0093693226
- acc4 SSIM_bbox delta: +0.0099654081
- acc8 SSIM_bbox delta: +0.0081881903

Conclusion:
Use this comparison to decide whether to continue increasing capacity. If cascade 4 improves bbox metrics, proceed to EXP012 with cascade=4, chans=12, sens_chans=4. If it degrades, keep EXP010 as the current best and test longer training or learning-rate adjustment.

## 2026-07-08 EXP011 cascade 4 result

EXP011_varnet_c4_ch9_s4_e10 completed successfully on VESSL.

Metrics:
- overall SSIM_full: 0.8994653502999606
- overall SSIM_bbox: 0.9182903899909547
- acc4 SSIM_full: 0.9153357927277868
- acc4 SSIM_bbox: 0.9281987882106104
- acc8 SSIM_full: 0.8826443344975511
- acc8 SSIM_bbox: 0.8986570824075628
- volumes: 30
- slices: 791
- bbox annotations: 161
- skipped: []

Comparison against EXP010:
- overall SSIM_full delta: +0.0048371709
- overall SSIM_bbox delta: +0.0093693226
- acc4 SSIM_bbox delta: +0.0099654081
- acc8 SSIM_bbox delta: +0.0081881903

Conclusion:
Cascade 4 is now the best configuration tested so far. Continue with EXP012: cascade=4, chans=12, sens_chans=4, epochs=10.
