"""Thin, fail-closed adapter for the pinned upstream PromptMR+ implementation."""

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


PROMPTMR_PLUS_ROOT = (
    Path(__file__).resolve().parents[2] / "third_party" / "promptmr_plus"
)


PROMPTMR_PLUS_MANIFEST_SHA256 = (
    "3d77c331b3d756ea855c12c08efe82d34755b30035eaef1c933c053bfa128876"
)


class PromptMRContractError(ValueError):
    """PromptMR+ input/output violates the pinned adapter contract."""


class PromptMRNonFiniteError(RuntimeError):
    """PromptMR+ produced non-finite output."""


def verify_promptmr_plus_source(root=PROMPTMR_PLUS_ROOT):
    """Verify every vendored upstream/config/license byte before import or use."""
    root = Path(root)
    manifest_path = root / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("PromptMR+ source manifest is missing or unsafe")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != PROMPTMR_PLUS_MANIFEST_SHA256:
        raise ValueError("PromptMR+ manifest checksum mismatch")
    manifest = json.loads(manifest_bytes)
    for relative_path, expected_hash in manifest["files"].items():
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"PromptMR+ source file is missing or unsafe: {relative_path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"PromptMR+ source hash mismatch: {relative_path}")
    executable_artifacts = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}
    )
    if executable_artifacts:
        raise ValueError(
            f"PromptMR+ unlisted executable bytecode: {executable_artifacts}"
        )
    allowed_files = set(manifest["files"]) | {
        "README.md",
        "SOURCE_MANIFEST.json",
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual_files - allowed_files)
    missing = sorted(allowed_files - actual_files)
    if unexpected or missing:
        raise ValueError(
            f"PromptMR+ unexpected vendored files: extra={unexpected}, missing={missing}"
        )
    return manifest


def acceleration_from_filename(filename):
    """Return the unique authoritative acc4/acc8 token from a volume name."""
    matches = [
        token
        for token in Path(filename).stem.lower().split("_")
        if token in {"acc4", "acc8"}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one acc4/acc8 filename token, got {matches!r}"
        )
    return int(matches[0][3:])


def adjacent_slice_indices(slice_index, num_slices, num_adjacent=5):
    """Match upstream edge replication for an odd adjacent-slice window."""
    if num_slices <= 0 or not 0 <= slice_index < num_slices:
        raise ValueError("slice index must address a non-empty volume")
    if num_adjacent <= 0 or num_adjacent % 2 != 1:
        raise ValueError("num_adjacent must be a positive odd integer")
    radius = num_adjacent // 2
    return tuple(
        min(max(slice_index + offset, 0), num_slices - 1)
        for offset in range(-radius, radius + 1)
    )


@dataclass(frozen=True)
class PromptMRInput:
    kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: torch.Tensor
    acceleration: int


def prepare_promptmr_input(
    kspace_volume, mask, *, slice_index, filename, num_adjacent=5
):
    """Adapt one volume position to PromptMR+'s adjacent-slice input contract."""
    volume = np.asarray(kspace_volume)
    if volume.ndim != 4 or not np.iscomplexobj(volume):
        raise PromptMRContractError(
            "kspace_volume must be complex [slices,coils,height,width]"
        )
    indices = adjacent_slice_indices(slice_index, volume.shape[0], num_adjacent)
    window = volume[list(indices)].reshape(
        num_adjacent * volume.shape[1], volume.shape[2], volume.shape[3]
    )
    mask_array = np.asarray(mask).astype(bool, copy=False).reshape(-1)
    if mask_array.shape != (volume.shape[3],):
        raise PromptMRContractError("mask width must match k-space width")
    window = window * mask_array.reshape(1, 1, -1)
    kspace = torch.from_numpy(
        np.stack((window.real, window.imag), axis=-1).astype(np.float32, copy=False)
    ).unsqueeze(0)
    mask_tensor = torch.from_numpy(
        mask_array.reshape(1, 1, 1, -1, 1).copy()
    )
    return PromptMRInput(
        kspace=kspace,
        mask=mask_tensor,
        num_low_frequencies=torch.full((1,), -1, dtype=torch.int64),
        acceleration=acceleration_from_filename(filename),
    )


def _promptmr_class():
    verify_promptmr_plus_source()
    upstream_root = PROMPTMR_PLUS_ROOT / "upstream"
    for module_name in ("models", "mri_utils", "data"):
        module = sys.modules.get(module_name)
        if module is not None:
            origins = [Path(path).resolve() for path in getattr(module, "__path__", [])]
            if origins and not all(
                origin == upstream_root.resolve() or upstream_root.resolve() in origin.parents
                for origin in origins
            ):
                raise RuntimeError(f"conflicting top-level module already loaded: {module_name}")
    sys.path.insert(0, str(upstream_root))
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module("models.promptmr_v2").PromptMR
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        sys.path.remove(str(upstream_root))


def build_promptmr_plus():
    """Build the exact pinned PromptMR+ FastMRI brain/knee architecture."""
    config = verify_promptmr_plus_source()["model_config"]
    promptmr_class = _promptmr_class()
    kwargs = {key: value for key, value in config.items() if key != "class"}
    model = promptmr_class(**kwargs)
    # Pinned upstream PromptMRBlock.forward reads self.n_buffer although the
    # constructor stores it on self.model. Preserve the upstream algorithm and
    # supply only the missing mirrored runtime attribute in this thin adapter.
    for cascade in model.cascades:
        cascade.n_buffer = cascade.model.n_buffer
    return model


def load_promptmr_plus_checkpoint(model, checkpoint):
    """Load only the upstream Lightning ``promptmr.`` state namespace strictly."""
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise RuntimeError("strict PromptMR+ checkpoint requires a state_dict")
    state = {
        key[len("promptmr.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("promptmr.")
    }
    if not state:
        raise RuntimeError("strict PromptMR+ checkpoint has no promptmr. namespace")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"strict PromptMR+ checkpoint mismatch: {exc}") from exc
    return model


def _center_crop(image, crop_size):
    height, width = map(int, crop_size)
    if height <= 0 or width <= 0 or height > image.shape[-2] or width > image.shape[-1]:
        raise PromptMRContractError("crop_size must fit within the PromptMR+ output")
    top = (image.shape[-2] - height) // 2
    left = (image.shape[-1] - width) // 2
    return image[..., top : top + height, left : left + width]


class PromptMRPlusAdapter(nn.Module):
    """Minimal schema/output wrapper around the unmodified pinned core."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(
        self,
        prepared,
        *,
        crop_size=None,
        use_checkpoint=False,
        compute_sens_per_coil=False,
    ):
        if prepared.acceleration not in (4, 8):
            raise PromptMRContractError("PromptMR+ acceleration must be 4 or 8")
        parameter = next(self.core.parameters(), None)
        device = parameter.device if parameter is not None else prepared.kspace.device
        output = self.core(
            prepared.kspace.to(device),
            prepared.mask.to(device),
            prepared.num_low_frequencies.to(device),
            ("cartesian",),
            use_checkpoint=use_checkpoint,
            compute_sens_per_coil=compute_sens_per_coil,
        )
        image = output.get("img_pred") if isinstance(output, dict) else None
        if image is None or image.ndim != 3 or torch.is_complex(image):
            raise PromptMRContractError("PromptMR+ returned an invalid real image")
        image = image.float()
        if crop_size is not None:
            image = _center_crop(image, crop_size)
        if not torch.isfinite(image).all():
            raise PromptMRNonFiniteError("PromptMR+ returned non-finite output")
        return image
