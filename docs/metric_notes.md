# Metric notes

The 2026 challenge uses:

- SSIM_full: SSIM averaged inside the foreground mask.
- SSIM_bbox: SSIM inside fastMRI+ lesion bounding boxes.

Rules:

- Reuse utils/common/metrics.py.
- Do not manually reimplement SSIM.
- Read bbox annotations from target image H5 attrs["annotations"].
- Use target H5 attrs["max"] as data_range.
- Report acc4 and acc8 separately.
- Track number of slices and bbox annotations used.
