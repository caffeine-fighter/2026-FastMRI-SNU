# Decisions

## 2026-07-06

- Use official 2026_baby_varnet as the primary baseline.
- Treat VESSL as the source of truth for final candidate model training.
- Keep raw data, H5 files, checkpoints, and result artifacts out of Git.
- Add local scripts while VESSL is blocked by resource quota exceeded.
- Reuse official metric helpers for SSIM_full and SSIM_bbox.
