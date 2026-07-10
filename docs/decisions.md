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
## 2026-07-08 EXP012 result

EXP012 completed on VESSL.

Configuration:
- cascade: 4
- chans: 12
- sens_chans: 4
- seed: 430

Metrics:
- overall SSIM_full: 0.8994141339351495
- overall SSIM_bbox: 0.9187541341189271
- acc4 SSIM_full: 0.9150149570343242
- acc4 SSIM_bbox: 0.9278178315296352
- acc8 SSIM_full: 0.8828788865357637
- acc8 SSIM_bbox: 0.9007945855458578
- volumes: 30
- slices: 791
- bbox annotations: 161
- skipped: []

Decision:
Compare EXP012 against EXP010 before launching the next VESSL run.
## 2026-07-08 EXP013 result

EXP013 completed on VESSL.

Configuration:
- cascade: 4
- chans: 12
- sens_chans: 8
- seed: 430

Metrics:
- overall SSIM_full: 0.8998157547035651
- overall SSIM_bbox: 0.9205543854221794
- acc4 SSIM_full: 0.9156682505947366
- acc4 SSIM_bbox: 0.931258392668216
- acc8 SSIM_full: 0.8830137603605787
- acc8 SSIM_bbox: 0.8993445932865143
- volumes: 30
- slices: 791
- bbox annotations: 161
- skipped: []

Decision:
Compare EXP013 against EXP012. If EXP013 improves overall SSIM_bbox, prioritize c4/ch12/s8 longer training. Otherwise keep EXP012 as the current best and prioritize c4/ch12/s4 longer training.

## 2026-07-10 EXP030 validation result

EXP030_varnet_c4_ch12_s8_e20 completed validation evaluation.

Configuration:
- cascade: 4
- chans: 12
- sens_chans: 8
- epochs: 20
- seed: 430

Metrics:
- val_loss: 3.202955630212294
- best_epoch: 19
- overall SSIM_full: 0.90337035478141
- overall SSIM_bbox: 0.9259878171156652
- validation quality: 0.9146790859485376
- acc4 SSIM_full: 0.918414056886912
- acc4 SSIM_bbox: 0.9352672735107279
- acc8 SSIM_full: 0.8874255976018807
- acc8 SSIM_bbox: 0.9076007461106336
- volumes: 30
- slices: 791
- bbox annotations: 161
- skipped: []

Comparison against EXP012:
- EXP012 SSIM_full: 0.8994141339351495
- EXP012 SSIM_bbox: 0.9187541341189271
- EXP012 validation quality: 0.9090841340270383
- EXP030 SSIM_full delta: +0.0039562208462605
- EXP030 SSIM_bbox delta: +0.0072336829967381
- EXP030 validation quality delta: +0.0055949519214994

Decision:
EXP030 beats EXP012 on validation quality and is the current best validation candidate. Next step is Phase 2 official recon_eval.

## 2026-07-10 EXP030 official Phase 2 decision

The official one-shot comparison selected `EXP030_official` over
`EXP012_official`, and the approved 30-repeat timing evaluation then completed
successfully for EXP030.

Final official result:
- completed runs: 30
- best run: `EXP030_official_run01`
- SSIM_full: 0.9178
- SSIM_bbox: 0.9108
- quality_score: 0.9143
- minimum time: 173.4 ms/slice
- time_score: 0.0009513541666666666
- total_score: 0.9152513541666667

Decision:
Use `EXP030_varnet_c4_ch12_s8_e20` as the final Phase 2 candidate. Do not run
additional official repeats unless the recorded result is invalidated. Package
the GitHub repository, loss plot, checkpoint, and model-description deck for
submission, keeping the checkpoint and all leaderboard data out of Git.
