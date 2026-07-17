"""Challenge-layout HDF5 adapter for the pinned PromptMR+ model."""

from __future__ import annotations

from collections import Counter
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.promptmr.contracts import (
    PROMPTMR_PLUS_RECIPE,
    adjacent_slice_indices,
    parse_acceleration_filename,
)
from utils.promptmr.runtime import activate_vendor_namespace


def _sampling_mask_width_axis(mask):
    if isinstance(mask, torch.Tensor):
        if mask.is_complex():
            raise TypeError("Sampling mask must be real-valued")
        shape = tuple(mask.shape)
    elif isinstance(mask, np.ndarray):
        if np.iscomplexobj(mask):
            raise TypeError("Sampling mask must be real-valued")
        shape = mask.shape
    else:
        raise TypeError("Sampling mask must be a NumPy array or torch tensor")

    if len(shape) == 1 and shape[0] > 0:
        return shape[0], 0
    if len(shape) == 2 and shape[0] == 1 and shape[1] > 0:
        return shape[1], 1
    if len(shape) == 3 and shape[0] == shape[2] == 1 and shape[1] > 0:
        return shape[1], 1
    if (
        len(shape) == 4
        and shape[0] == shape[1] == shape[3] == 1
        and shape[2] > 0
    ):
        return shape[2], 2
    raise ValueError(
        "Sampling mask must have shape [W], [1,W], [1,W,1], or [1,1,W,1]"
    )


def resize_sampling_mask(mask, width: int):
    """Center-resize a real sampling mask without changing rank or backend.

    This follows the pinned upstream convention: ``difference // 2`` is assigned
    to the left, and an odd remainder is cropped from or padded on the right.
    """
    if isinstance(width, bool) or not isinstance(width, Integral):
        raise TypeError("Sampling mask width must be a positive integer")
    width = int(width)
    if width <= 0:
        raise ValueError("Sampling mask width must be a positive integer")
    current_width, axis = _sampling_mask_width_axis(mask)
    output_shape = list(mask.shape)
    output_shape[axis] = width

    if isinstance(mask, torch.Tensor):
        if current_width == width:
            return mask.clone()
        if current_width > width:
            start = (current_width - width) // 2
            return mask.narrow(axis, start, width).clone()
        output = torch.zeros(output_shape, dtype=mask.dtype, device=mask.device)
    else:
        if current_width == width:
            return mask.copy()
        if current_width > width:
            start = (current_width - width) // 2
            slices = [slice(None)] * mask.ndim
            slices[axis] = slice(start, start + width)
            return mask[tuple(slices)].copy()
        output = np.zeros(output_shape, dtype=mask.dtype)

    left = (width - current_width) // 2
    destination = [slice(None)] * mask.ndim
    destination[axis] = slice(left, left + current_width)
    output[tuple(destination)] = mask
    return output


def align_promptmr_output_target(output, target):
    """Use the upstream loss alignment without cropping validation inputs."""
    activate_vendor_namespace()
    from data.transforms import center_crop_to_smallest

    return center_crop_to_smallest(output, target)


class PromptMRDataTransform:
    """Reuse the upstream transform while preserving full-resolution validation."""

    def __init__(self, *, training: bool):
        activate_vendor_namespace()
        from data.transforms import FastmriDataTransform

        crop = (
            tuple(PROMPTMR_PLUS_RECIPE["training"]["train_crop"])
            if training
            else None
        )
        self.training = training
        self.crop = crop
        self.transform = FastmriDataTransform(
            mask_func=None,
            uniform_resolution=crop,
            mask_type="cartesian",
            test_num_low_frequencies=None,
        )

    def __call__(self, kspace, mask, target, maximum, fname, slice_num):
        expected_width = self.crop[1] if self.crop is not None else kspace.shape[-1]
        mask_width, _ = _sampling_mask_width_axis(mask)
        if not self.training and mask_width != expected_width:
            raise ValueError(
                f"Validation mask width {mask_width} does not match k-space "
                f"width {expected_width} for {fname}"
            )
        resized_mask = resize_sampling_mask(mask, expected_width)
        if isinstance(resized_mask, torch.Tensor):
            flat_mask = resized_mask.detach().cpu().numpy().reshape(-1)
        else:
            flat_mask = resized_mask.reshape(-1)
        attrs = {
            "max": float(maximum),
            "padding_left": 0,
            "padding_right": expected_width,
            "recon_size": tuple(target.shape[-2:]),
        }
        sample = self.transform(
            kspace,
            flat_mask,
            target,
            attrs,
            fname,
            slice_num,
        )
        masked_kspace = sample.masked_kspace * sample.mask
        return (
            sample.mask,
            masked_kspace,
            sample.target,
            torch.tensor(sample.max_value, dtype=torch.float32),
            sample.fname,
            sample.slice_num,
        )


class PromptMRSliceDataset(Dataset):
    """Load five neighboring slices with clamp-based boundary replication."""

    def __init__(
        self,
        root: Path,
        *,
        transform: PromptMRDataTransform,
        input_key: str = "kspace",
        target_key: str = "image_label",
        max_key: str = "max",
    ):
        self.root = Path(root)
        self.input_key = input_key
        self.target_key = target_key
        self.max_key = max_key
        self.transform = transform
        self.raw_samples = []
        accelerations = Counter()
        kspace_root = self.root / "kspace"
        image_root = self.root / "image"
        if not kspace_root.is_dir() or not image_root.is_dir():
            raise FileNotFoundError(
                f"PromptMR+ expects {kspace_root} and {image_root} directories"
            )
        files = sorted(kspace_root.glob("*.h5"))
        if not files:
            raise ValueError(f"No HDF5 files found in {kspace_root}")
        for kspace_path in files:
            acceleration = parse_acceleration_filename(kspace_path.name)
            target_path = image_root / kspace_path.name
            if not target_path.is_file():
                raise FileNotFoundError(f"Missing target volume: {target_path}")
            with h5py.File(kspace_path, "r") as handle:
                if input_key not in handle or "mask" not in handle:
                    raise ValueError(
                        f"{kspace_path.name} must contain {input_key!r} and 'mask'"
                    )
                shape = handle[input_key].shape
                if len(shape) != 4 or shape[0] <= 0:
                    raise ValueError(
                        f"{input_key!r} must have [slice, coil, height, width] shape "
                        f"in {kspace_path.name}"
                    )
                num_slices = int(shape[0])
            with h5py.File(target_path, "r") as handle:
                if target_key not in handle or max_key not in handle.attrs:
                    raise ValueError(
                        f"{target_path.name} must contain {target_key!r} and "
                        f"{max_key!r} attr"
                    )
                if handle[target_key].shape[0] != num_slices:
                    raise ValueError(f"Slice count mismatch in {target_path.name}")
            accelerations[acceleration] += 1
            self.raw_samples.extend(
                (kspace_path, target_path, index, num_slices, acceleration)
                for index in range(num_slices)
            )
        if any(accelerations[value] == 0 for value in (4, 8)):
            raise ValueError(
                "PromptMR+ dataset routing requires both acc4 and acc8 volumes; "
                f"observed {dict(accelerations)}"
            )
        self.acceleration_volumes = dict(accelerations)

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, index):
        kspace_path, target_path, center, num_slices, _ = self.raw_samples[index]
        adjacent = adjacent_slice_indices(
            center,
            num_slices,
            PROMPTMR_PLUS_RECIPE["architecture"]["num_adj_slices"],
        )
        with h5py.File(kspace_path, "r") as handle:
            dataset = handle[self.input_key]
            slices = [np.asarray(dataset[slice_index]) for slice_index in adjacent]
            if any(item.ndim != 3 for item in slices):
                raise ValueError(f"Invalid coil slice shape in {kspace_path.name}")
            kspace = np.concatenate(slices, axis=0)
            mask = np.asarray(handle["mask"])
        with h5py.File(target_path, "r") as handle:
            target = np.asarray(handle[self.target_key][center])
            maximum = float(handle.attrs[self.max_key])
        return self.transform(
            kspace,
            mask,
            target,
            maximum,
            kspace_path.name,
            center,
        )


def create_promptmr_data_loaders(args):
    if args.batch_size != 1:
        raise ValueError("Pinned PromptMR+ recipe requires batch size 1")
    train = PromptMRSliceDataset(
        args.data_path_train,
        input_key=args.input_key,
        target_key=args.target_key,
        max_key=args.max_key,
        transform=PromptMRDataTransform(training=True),
    )
    validation = PromptMRSliceDataset(
        args.data_path_val,
        input_key=args.input_key,
        target_key=args.target_key,
        max_key=args.max_key,
        transform=PromptMRDataTransform(training=False),
    )
    loader_options = {
        "batch_size": 1,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    return (
        DataLoader(train, shuffle=True, **loader_options),
        DataLoader(validation, shuffle=False, **loader_options),
    )
