"""Thin runtime adapter around the byte-pinned upstream PromptMR+ v2 source."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.utils.checkpoint
from torch import nn

from utils.promptmr.contracts import PROMPTMR_PLUS_RECIPE


_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "promptmr_plus"


def _is_from_vendor(module) -> bool:
    path = getattr(module, "__file__", None)
    if path is None:
        paths = getattr(module, "__path__", ())
        return any(_VENDOR_ROOT in Path(item).resolve().parents for item in paths)
    resolved = Path(path).resolve()
    return resolved == _VENDOR_ROOT or _VENDOR_ROOT in resolved.parents


def activate_vendor_namespace() -> None:
    vendor = str(_VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    for package in ("data", "mri_utils", "models"):
        existing = sys.modules.get(package)
        if existing is not None and not _is_from_vendor(existing):
            raise RuntimeError(
                f"Cannot activate pinned PromptMR+ because top-level package {package!r} "
                "was already imported from another source"
            )
    __import__("data")
    __import__("mri_utils")


def _apply_upstream_compatibility_shims(promptmr) -> None:
    """Populate attributes read by the pinned upstream cascade forward path."""
    for cascade in promptmr.cascades:
        cascade.n_buffer = cascade.model.n_buffer


class PromptMRPlusAdapter(nn.Module):
    """Expose the repository's two-argument model contract and center output."""

    def __init__(self):
        super().__init__()
        activate_vendor_namespace()
        from models.promptmr_v2 import PromptMR

        self.promptmr = PromptMR(**PROMPTMR_PLUS_RECIPE["architecture"])
        _apply_upstream_compatibility_shims(self.promptmr)
        self.num_adj_slices = PROMPTMR_PLUS_RECIPE["architecture"]["num_adj_slices"]
        self.activation_checkpointing = PROMPTMR_PLUS_RECIPE["runtime"][
            "activation_checkpointing"
        ]
        self.compute_sens_per_coil = PROMPTMR_PLUS_RECIPE["runtime"][
            "compute_sens_per_coil"
        ]

    def forward(self, masked_kspace, mask):
        mask_types = tuple("cartesian" for _ in range(masked_kspace.shape[0]))
        outputs = self.promptmr(
            masked_kspace,
            mask,
            None,
            mask_types,
            use_checkpoint=self.activation_checkpointing,
            compute_sens_per_coil=self.compute_sens_per_coil,
        )
        return outputs["img_pred"]


def build_promptmr_plus_model() -> nn.Module:
    return PromptMRPlusAdapter()


def build_promptmr_plus_loss() -> nn.Module:
    activate_vendor_namespace()
    from mri_utils.losses import SSIMLoss

    loss = PROMPTMR_PLUS_RECIPE["loss"]
    return SSIMLoss(
        win_size=loss["win_size"],
        k1=loss["k1"],
        k2=loss["k2"],
    )
