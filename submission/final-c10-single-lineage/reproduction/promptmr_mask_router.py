#!/usr/bin/env python3
"""Fail-safe routing policy for legal-mask PromptMR+ checkpoints."""

from __future__ import annotations

import numpy as np
import torch

try:
    from utils.learning.promptmr_legal_mask import parse_legal_cartesian_mask
except ModuleNotFoundError:
    from vessl_ops.staging.promptmr_legal_mask import parse_legal_cartesian_mask


def _cpu_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        value = mask.detach()
        if value.is_cuda:
            value = value.cpu()
        return value.numpy()
    return np.asarray(mask)


def select_component(
    mask: np.ndarray | torch.Tensor,
    *,
    declared_acceleration: int | None = None,
    generalist_component: str,
) -> tuple[str, dict[str, object]]:
    """Choose a specialist from the k-space mask alone.

    ``declared_acceleration`` is retained as a backwards-compatible call
    argument, but is deliberately ignored.  The final timed path must not
    trust a filename- or dataset-derived label: exact mask density, ACS
    width, period/residue, and offset are the complete dispatch evidence.
    """

    if generalist_component not in {"generalist", "acc4", "acc8"}:
        raise ValueError(
            "generalist component must be generalist, acc4, or acc8"
        )
    features = parse_legal_cartesian_mask(
        _cpu_mask(mask),
        acceleration=None,
    )
    if features is None:
        return generalist_component, {
            "route": generalist_component,
            "reason": "UNKNOWN_OR_OUT_OF_FAMILY_MASK",
            "specialist_activated": False,
            "dispatch_source": "kspace_mask_only",
        }
    required = {
        "mask_density",
        "acs_width",
        "period",
        "residue",
        "offset",
    }
    if not required.issubset(features):
        raise RuntimeError("legal-mask parser lacks complete dispatch features")
    route = f"acc{int(features['acceleration'])}"
    return route, {
        "route": route,
        "reason": "EXACT_LEGAL_MASK_FAMILY_MATCH",
        "specialist_activated": route != generalist_component,
        "dispatch_source": "kspace_mask_only",
        "dispatch_features": {
            key: features[key]
            for key in ("mask_density", "acs_width", "period", "residue", "offset")
        },
        "features": features,
    }
