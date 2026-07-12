"""Scoring core shared by self-evaluation (val) and the leaderboard.

Two metrics are reported for the challenge:
  - SSIM_full: SSIM averaged only inside the foreground mask
  - SSIM_bbox: SSIM inside each fastMRI+ lesion bounding box

Bounding boxes live in the `annotations` attribute of each image H5
(384 x 384 image space). `data_range` is the volume `max` attribute.
Participants can score their own validation reconstructions with the same
functions used on the leaderboard.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.common.loss_function import SSIMLoss


class SSIM(SSIMLoss):
    """SSIMLoss that returns the per-pixel SSIM map for a single 2D image."""

    def forward(self, X, Y, data_range):
        if X.dim() != 2 or Y.dim() != 2:
            raise ValueError("SSIM expects 2D (H, W) inputs")
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        S = (A1 * A2) / (B1 * B2)
        return S[0, 0]


def foreground_mask(target):
    """Binary foreground mask for a 2D image, matching the leaderboard."""
    mask = np.zeros(target.shape)
    mask[target > 2e-5] = 1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)
    return mask


def ssim_full_tensor(ssim, recon_t, target_t, mask_t, data_range):
    """Differentiable foreground SSIM, or None when the mask is empty."""
    data_range = torch.as_tensor(
        data_range, dtype=recon_t.dtype, device=recon_t.device
    )
    ssim_map = ssim(recon_t * mask_t, target_t * mask_t, data_range)
    pad = ssim.win_size // 2
    mask_valid = mask_t[pad:mask_t.shape[0] - pad, pad:mask_t.shape[1] - pad]
    denom = mask_valid.sum()
    if denom <= 0:
        return None
    return (ssim_map * mask_valid).sum() / denom


def ssim_full(ssim, recon_t, target_t, mask_t, data_range):
    """SSIM averaged only inside the foreground mask.

    Returns None when the mask is empty so the caller can skip the slice.
    """
    ssim_map = ssim(recon_t * mask_t, target_t * mask_t, data_range)
    pad = ssim.win_size // 2
    mask_valid = mask_t[pad:mask_t.shape[0] - pad, pad:mask_t.shape[1] - pad]
    denom = mask_valid.sum()
    if denom <= 0:
        return None
    return ((ssim_map * mask_valid).sum() / denom).item()


def ssim_bbox_tensor(ssim, recon_t, target_t, box, data_range):
    """Differentiable SSIM inside one clipped box, or None when too small."""
    win = ssim.win_size
    x0, y0 = max(0, box["x"]), max(0, box["y"])
    x1 = min(target_t.shape[1], box["x"] + box["width"])
    y1 = min(target_t.shape[0], box["y"] + box["height"])
    if (x1 - x0) < win or (y1 - y0) < win:
        return None
    recon_crop = recon_t[y0:y1, x0:x1]
    target_crop = target_t[y0:y1, x0:x1]
    data_range = torch.as_tensor(
        data_range, dtype=recon_t.dtype, device=recon_t.device
    )
    return ssim(recon_crop, target_crop, data_range).mean()


class ScoreAlignedLoss(torch.nn.Module):
    """Training-only objective matching the evaluator's separate score cells."""

    _METADATA_KEYS = {
        "acceleration", "boxes", "box_count", "foreground_mask",
        "full_weight", "box_weight",
    }

    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def _validate(self, output, target, data_range, metadata):
        if output.ndim != 3 or target.shape != output.shape:
            raise ValueError("Score-aligned output and target must be matching (B,H,W)")
        if not isinstance(metadata, dict) or set(metadata) != self._METADATA_KEYS:
            raise ValueError("Malformed score-aligned metadata keys")
        batch_size, height, width = output.shape
        acceleration = metadata["acceleration"]
        boxes = metadata["boxes"]
        box_count = metadata["box_count"]
        foreground = metadata["foreground_mask"]
        full_weight = metadata["full_weight"]
        box_weight = metadata["box_weight"]
        if (
            not torch.is_tensor(acceleration)
            or acceleration.dtype != torch.int64
            or acceleration.shape != (batch_size,)
            or not torch.all((acceleration == 4) | (acceleration == 8))
        ):
            raise ValueError("Malformed score-aligned acceleration")
        if (
            not torch.is_tensor(boxes)
            or boxes.dtype != torch.int64
            or boxes.ndim != 3
            or boxes.shape[0] != batch_size
            or boxes.shape[2] != 4
        ):
            raise ValueError("Malformed score-aligned boxes")
        if (
            not torch.is_tensor(box_count)
            or box_count.dtype != torch.int64
            or box_count.shape != (batch_size,)
            or torch.any(box_count < 0)
            or torch.any(box_count > boxes.shape[1])
        ):
            raise ValueError("Malformed score-aligned box count")
        if (
            not torch.is_tensor(foreground)
            or foreground.dtype != torch.bool
            or foreground.shape != (batch_size, height, width)
        ):
            raise ValueError("Malformed score-aligned foreground mask")
        for weights, name in (
            (full_weight, "full weight"), (box_weight, "box weight")
        ):
            if (
                not torch.is_tensor(weights)
                or weights.shape != (batch_size,)
                or not torch.is_floating_point(weights)
                or not torch.all(torch.isfinite(weights))
                or not torch.all(weights > 0)
            ):
                raise ValueError(f"Malformed score-aligned {name}")
        data_range_t = torch.as_tensor(data_range)
        if (
            data_range_t.shape != (batch_size,)
            or not torch.all(torch.isfinite(data_range_t))
            or not torch.all(data_range_t > 0)
        ):
            raise ValueError("Malformed score-aligned data range")

        for sample_index in range(batch_size):
            count = int(box_count[sample_index].item())
            active = boxes[sample_index, :count]
            padding = boxes[sample_index, count:]
            if active.numel() and torch.any(active[:, 2:] <= 0):
                raise ValueError("Malformed score-aligned active box")
            if padding.numel() and torch.any(padding != 0):
                raise ValueError("Malformed score-aligned box padding")

    def forward(self, output, target, data_range, metadata):
        self._validate(output, target, data_range, metadata)
        losses = []
        for sample_index in range(output.shape[0]):
            recon_t = output[sample_index]
            target_t = target[sample_index]
            mask_t = metadata["foreground_mask"][sample_index].to(
                device=recon_t.device, dtype=recon_t.dtype
            )
            sample_loss = recon_t.sum() * 0
            full_similarity = ssim_full_tensor(
                self.ssim, recon_t, target_t, mask_t, data_range[sample_index]
            )
            full_weight = metadata["full_weight"][sample_index].to(
                device=recon_t.device, dtype=recon_t.dtype
            )
            if full_similarity is not None:
                sample_loss = sample_loss + full_weight * (1 - full_similarity)

            box_weight = metadata["box_weight"][sample_index].to(
                device=recon_t.device, dtype=recon_t.dtype
            )
            count = int(metadata["box_count"][sample_index].item())
            for coordinates in metadata["boxes"][sample_index, :count].tolist():
                x, y, width, height = coordinates
                box_similarity = ssim_bbox_tensor(
                    self.ssim,
                    recon_t,
                    target_t,
                    {"x": x, "y": y, "width": width, "height": height},
                    data_range[sample_index],
                )
                if box_similarity is not None:
                    sample_loss = sample_loss + box_weight * (1 - box_similarity)
            losses.append(sample_loss)
        return torch.stack(losses).mean()


def ssim_bbox(ssim, recon_t, target_t, box, data_range):
    """SSIM inside a single annotation box. Returns None if the box is too small."""
    win = ssim.win_size
    x0, y0 = max(0, box["x"]), max(0, box["y"])
    x1 = min(target_t.shape[1], box["x"] + box["width"])
    y1 = min(target_t.shape[0], box["y"] + box["height"])
    if (x1 - x0) < win or (y1 - y0) < win:
        return None
    recon_crop = recon_t[y0:y1, x0:x1]
    target_crop = target_t[y0:y1, x0:x1]
    return ssim(recon_crop, target_crop, data_range).mean().item()
