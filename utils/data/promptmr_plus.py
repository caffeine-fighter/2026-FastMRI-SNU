"""Paired SNU challenge dataset adapter for pinned PromptMR+ training."""

from pathlib import Path
from typing import NamedTuple
import hashlib
import json
import os
import stat

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.learning.promptmr_plus_training import load_promptmr_training_recipe
from utils.model.promptmr_plus_adapter import (
    acceleration_from_filename,
    adjacent_slice_indices,
)


class PromptMRTrainingSample(NamedTuple):
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: torch.Tensor
    target: torch.Tensor
    max_value: torch.Tensor
    fname: str
    slice_num: int
    acceleration: int


class PromptMRPlusSliceData(Dataset):
    """Read five same-volume adjacent slices from paired read-only H5 files."""

    @staticmethod
    def _open_regular(path):
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError(f"PromptMR+ dataset entry is not regular: {path}")
        return os.fdopen(fd, "rb")

    @staticmethod
    def _digest(handle):
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
        return digest.hexdigest()

    def __init__(
        self,
        root,
        input_key="kspace",
        target_key="image_label",
        max_key="max",
    ):
        recipe = load_promptmr_training_recipe()
        self.uniform_resolution = tuple(recipe["uniform_resolution"])
        self.input_key = input_key
        self.target_key = target_key
        self.max_key = max_key
        root = Path(root)
        kspace_root = root / "kspace"
        image_root = root / "image"
        if not kspace_root.is_dir() or not image_root.is_dir():
            raise ValueError("PromptMR+ requires paired kspace/ and image/ directories")

        kspace_files = self._safe_h5_files(kspace_root)
        image_files = self._safe_h5_files(image_root)
        if set(kspace_files) != set(image_files):
            raise ValueError("PromptMR+ image and kspace filename sets must match exactly")

        self.examples = []
        self.files = []
        inventory = []
        for name in sorted(kspace_files):
            kspace_path = kspace_files[name]
            image_path = image_files[name]
            acceleration_from_filename(name)
            with self._open_regular(kspace_path) as kspace_source, self._open_regular(
                image_path
            ) as image_source:
                kspace_sha256 = self._digest(kspace_source)
                image_sha256 = self._digest(image_source)
                kspace_stat = os.fstat(kspace_source.fileno())
                image_stat = os.fstat(image_source.fileno())
                with h5py.File(kspace_source, "r") as kspace_h5, h5py.File(
                    image_source, "r"
                ) as image_h5:
                    if self.input_key not in kspace_h5 or "mask" not in kspace_h5:
                        raise ValueError(f"Missing PromptMR+ kspace or mask dataset: {name}")
                    if (
                        self.target_key not in image_h5
                        or self.max_key not in image_h5.attrs
                    ):
                        raise ValueError(f"Missing PromptMR+ target or max metadata: {name}")
                    kspace = kspace_h5[self.input_key]
                    target = image_h5[self.target_key]
                    if (
                        kspace.ndim != 4
                        or target.ndim != 3
                        or kspace.shape[0] != target.shape[0]
                        or kspace_h5["mask"].shape != (kspace.shape[-1],)
                    ):
                        raise ValueError(f"Invalid PromptMR+ paired shape contract: {name}")
                    if (
                        tuple(kspace.shape[-2:]) != self.uniform_resolution
                        or tuple(target.shape[-2:]) != self.uniform_resolution
                    ):
                        raise ValueError(
                            "PromptMR+ non-384 inputs require an approved stored-mask "
                            f"mask-mapping policy before training: {name}"
                        )
                    if not np.issubdtype(kspace.dtype, np.complexfloating):
                        raise ValueError(f"PromptMR+ kspace must be complex: {name}")
                    if np.issubdtype(target.dtype, np.complexfloating):
                        raise ValueError(f"PromptMR+ target must be real: {name}")
                    mask = np.asarray(kspace_h5["mask"])
                    if (
                        not np.isfinite(mask).all()
                        or not np.isin(mask, (0, 1)).all()
                    ):
                        raise ValueError(f"Mask must be finite and binary: {name}")
                    max_value = float(image_h5.attrs[self.max_key])
                    if not np.isfinite(max_value) or max_value <= 0:
                        raise ValueError(f"Invalid max value: {name}")
                    num_slices = int(kspace.shape[0])
                    inventory.append(
                        {
                            "name": name,
                            "kspace_shape": list(kspace.shape),
                            "kspace_dtype": str(kspace.dtype),
                            "target_shape": list(target.shape),
                            "target_dtype": str(target.dtype),
                            "mask_shape": list(mask.shape),
                            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                            "max_value": max_value,
                            "kspace_file_size": kspace_stat.st_size,
                            "image_file_size": image_stat.st_size,
                            "kspace_sha256": kspace_sha256,
                            "image_sha256": image_sha256,
                        }
                    )
                if (
                    self._digest(kspace_source) != kspace_sha256
                    or self._digest(image_source) != image_sha256
                ):
                    raise ValueError(
                        f"PromptMR+ source bytes changed during inventory: {name}"
                    )
            file_index = len(self.files)
            self.files.append(
                (kspace_path, image_path, name, kspace_sha256, image_sha256)
            )
            self.examples.extend((file_index, index) for index in range(num_slices))
        inventory_bytes = json.dumps(
            inventory, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()

    @staticmethod
    def _safe_h5_files(root):
        files = {}
        for path in root.iterdir():
            if path.suffix != ".h5" or path.is_symlink() or not path.is_file():
                raise ValueError(f"PromptMR+ dataset entry must be a regular .h5 file: {path.name}")
            files[path.name] = path
        if not files:
            raise ValueError("PromptMR+ dataset split is empty")
        return files

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file_index, slice_index = self.examples[item]
        (
            kspace_path,
            image_path,
            name,
            expected_kspace_sha256,
            expected_image_sha256,
        ) = self.files[file_index]
        with self._open_regular(kspace_path) as kspace_source, self._open_regular(
            image_path
        ) as image_source:
            if (
                self._digest(kspace_source) != expected_kspace_sha256
                or self._digest(image_source) != expected_image_sha256
            ):
                raise ValueError(f"PromptMR+ source bytes changed after inventory: {name}")
            with h5py.File(kspace_source, "r") as kspace_h5:
                dataset = kspace_h5[self.input_key]
                indices = adjacent_slice_indices(slice_index, dataset.shape[0], 5)
                slices = [np.asarray(dataset[index]) for index in indices]
                mask_array = np.asarray(kspace_h5["mask"])
            with h5py.File(image_source, "r") as image_handle:
                target_dataset = image_handle[self.target_key]
                target_array = np.asarray(target_dataset[slice_index])
                max_value = float(image_handle.attrs[self.max_key])
            if (
                self._digest(kspace_source) != expected_kspace_sha256
                or self._digest(image_source) != expected_image_sha256
            ):
                raise ValueError(f"PromptMR+ source bytes changed during sample read: {name}")

        adjacent = np.concatenate(slices, axis=0)
        if not np.isfinite(max_value) or max_value <= 0:
            raise ValueError(f"Invalid max value in {image_path.name}")
        if not np.isfinite(target_array).all():
            raise ValueError(f"Non-finite target values in {image_path.name}")
        if not np.isfinite(adjacent).all():
            raise ValueError(f"Non-finite k-space values in {kspace_path.name}")
        if (
            not np.isfinite(mask_array).all()
            or not np.isin(mask_array, (0, 1)).all()
        ):
            raise ValueError(f"Mask must be finite and binary in {kspace_path.name}")
        mask_array = mask_array.astype(bool, copy=False)

        pair = torch.from_numpy(
            np.stack((adjacent.real, adjacent.imag), axis=-1).astype(
                np.float32, copy=False
            )
        )
        mask_tensor = torch.from_numpy(mask_array.reshape(1, 1, -1, 1).copy())
        pair = pair * mask_tensor
        return PromptMRTrainingSample(
            masked_kspace=pair,
            mask=mask_tensor,
            num_low_frequencies=torch.tensor(-1, dtype=torch.int64),
            target=torch.from_numpy(target_array.astype(np.float32, copy=False)),
            max_value=torch.tensor(max_value, dtype=torch.float32),
            fname=name,
            slice_num=slice_index,
            acceleration=acceleration_from_filename(name),
        )
