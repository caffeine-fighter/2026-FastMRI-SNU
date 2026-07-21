"""Checkpointed, resumable FI-VarNet acc8 full-training primitives.

This module is deliberately separate from the one-step smoke path.  Its public
runner is gated by ``--fi-acc8-full-training`` and its checkpoints never carry
smoke authority, evaluation authority, or submission authority.
"""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import copy
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import sys
import tempfile
import time
import uuid

import h5py
import numpy as np
import torch

from utils.learning.fi_acc8_training import (
    FI_ACC8_PRODUCTION_ROOT,
    FI_ACTIVATION_CHECKPOINT_CONTRACT,
    _assert_finite_parameters,
    _center_crop_to_smallest,
    _cpu_snapshot,
    _h5_from_fd,
    _open_child_directory_nofollow,
    _open_directory_chain_nofollow,
    _open_regular_file_nofollow,
    _require_file_identity,
    _same_directory_identity,
    _select_smoke_device,
    _sha256_fd,
    _stat_signature,
    _validate_activation_checkpoint_contract,
    build_fi_scheduler,
    fi_lr_multiplier,
    inspect_acc8_training_data,
    preflight_smoke_gpu,
)
from utils.learning.train_part import (
    _cleanup_staged_directory,
    _create_staged_directory,
    _open_durably_created_directory,
    _publish_staged_directory_no_replace,
    _seal_staged_directory,
    _staged_directory_descriptor_path,
    build_model,
)
from utils.model.fi_varnet_adapter import (
    FI_DETERMINISTIC_REFLECT_PAD_CONTRACT,
    build_pinned_ssim_loss,
    enable_fi_activation_checkpointing,
    install_deterministic_reflect_pad_adapter,
    validate_deterministic_reflect_pad_receipt,
    verify_pinned_upstream_sources,
)


FI_ACC8_FULL_NAMESPACE = "EXP_FI_ACC8_CKPT_BASE_E30_R1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FI_ACC8_DETERMINISM_CONTRACT = {
    "schema": "fi-acc8-determinism-v2",
    "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "implementation": FI_DETERMINISTIC_REFLECT_PAD_CONTRACT["implementation"],
    "version": FI_DETERMINISTIC_REFLECT_PAD_CONTRACT["version"],
    "native_forward_exact": True,
    "state_dict_unchanged": True,
    "strict_deterministic_algorithms": True,
}
FI_ACC8_CHECKPOINT_BYTES = 1_479_000_000
FI_ACC8_RETAINED_CHECKPOINT_LIMIT = 32
FI_ACC8_RESOURCE_FIXED_RAM_MARGIN_BYTES = 512 * 1024 * 1024


def _available_ram_bytes():
    with Path("/proc/meminfo").open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
    raise RuntimeError("Cannot determine MemAvailable for FI acc8 resource preflight")


def _nearest_existing_ancestor(path):
    candidate = Path(os.path.abspath(os.fspath(path)))
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError("Cannot find existing output ancestor for disk preflight")
        candidate = candidate.parent
    return candidate


def preflight_full_training_resources(
    manifest,
    output_dir,
    *,
    available_ram_bytes=None,
    free_disk_bytes=None,
):
    """Fail before output reservation/CUDA unless conservative resources fit."""
    max_pair_bytes = max(
        int(record.kspace_size) + int(record.image_size)
        for record in manifest.records
    )
    cpu_model_bytes = FI_ACC8_CHECKPOINT_BYTES
    cpu_optimizer_bytes = FI_ACC8_CHECKPOINT_BYTES
    checkpoint_snapshot_bytes = FI_ACC8_CHECKPOINT_BYTES
    checkpoint_serialization_bytes = FI_ACC8_CHECKPOINT_BYTES
    required_ram_bytes = (
        max_pair_bytes
        + cpu_model_bytes
        + cpu_optimizer_bytes
        + checkpoint_snapshot_bytes
        + checkpoint_serialization_bytes
        + FI_ACC8_RESOURCE_FIXED_RAM_MARGIN_BYTES
    )
    retained_checkpoint_bytes = (
        FI_ACC8_RETAINED_CHECKPOINT_LIMIT * FI_ACC8_CHECKPOINT_BYTES
    )
    staging_checkpoint_bytes = FI_ACC8_CHECKPOINT_BYTES
    disk_margin_bytes = math.ceil(
        (retained_checkpoint_bytes + staging_checkpoint_bytes) * 0.05
    )
    required_disk_bytes = (
        retained_checkpoint_bytes + staging_checkpoint_bytes + disk_margin_bytes
    )
    if available_ram_bytes is None:
        available_ram_bytes = _available_ram_bytes()
    if free_disk_bytes is None:
        free_disk_bytes = shutil.disk_usage(
            _nearest_existing_ancestor(Path(output_dir).parent)
        ).free
    for value, description in (
        (available_ram_bytes, "available RAM"),
        (free_disk_bytes, "free disk"),
    ):
        if type(value) is not int or value < 0:
            raise RuntimeError(f"Invalid FI acc8 {description} measurement")
    result = {
        "schema": "fi-acc8-resource-preflight-v1",
        "checkpoint_bytes": FI_ACC8_CHECKPOINT_BYTES,
        "max_volume_bytes": max_pair_bytes,
        "max_pair_bytes": max_pair_bytes,
        "cpu_model_bytes": cpu_model_bytes,
        "cpu_optimizer_bytes": cpu_optimizer_bytes,
        "checkpoint_snapshot_bytes": checkpoint_snapshot_bytes,
        "checkpoint_serialization_bytes": checkpoint_serialization_bytes,
        "ram_margin_bytes": FI_ACC8_RESOURCE_FIXED_RAM_MARGIN_BYTES,
        "retained_checkpoint_limit": FI_ACC8_RETAINED_CHECKPOINT_LIMIT,
        "retained_checkpoint_bytes": retained_checkpoint_bytes,
        "staging_checkpoint_bytes": staging_checkpoint_bytes,
        "disk_margin_bytes": disk_margin_bytes,
        "available_ram_bytes": available_ram_bytes,
        "required_ram_bytes": required_ram_bytes,
        "free_disk_bytes": free_disk_bytes,
        "required_disk_bytes": required_disk_bytes,
    }
    if available_ram_bytes < required_ram_bytes:
        raise RuntimeError(
            "Insufficient available RAM for memory-safe FI acc8 checkpoint staging: "
            f"available={available_ram_bytes} required={required_ram_bytes}"
        )
    if free_disk_bytes < required_disk_bytes:
        raise RuntimeError(
            "Insufficient free disk for bounded FI acc8 checkpoint retention: "
            f"free={free_disk_bytes} required={required_disk_bytes}"
        )
    return result


def configure_determinism_pre_cuda():
    """Install and verify the frozen deterministic contract before CUDA selection."""
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif configured != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly "
            f"{CUBLAS_WORKSPACE_CONFIG!r}, got {configured!r}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG
        or not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cudnn.deterministic is not True
        or torch.backends.cudnn.benchmark is not False
    ):
        raise RuntimeError("FI acc8 deterministic execution contract did not apply")
    return dict(FI_ACC8_DETERMINISM_CONTRACT)


def _build_full_training_model_with_adapters(args):
    """Build the fresh CPU FI model and fail closed on either adapter receipt."""
    model = build_model(args)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("FI acc8 model factory must return fresh CPU parameters")
    reflect_padding = validate_deterministic_reflect_pad_receipt(
        install_deterministic_reflect_pad_adapter(model)
    )
    activation = _validate_activation_checkpoint_contract(
        enable_fi_activation_checkpointing(model)
    )
    return model, reflect_padding, activation


@dataclass(frozen=True)
class FIAcc8FullRecipe:
    schema: str = "fi-varnet-acc8-checkpointed-full-training-v2"
    model_family: str = "fi-varnet-acc8"
    namespace: str = FI_ACC8_FULL_NAMESPACE
    scope: str = "FULL_TRAINING_ONLY"
    scratch: bool = True
    external_learned_state: bool = False
    seed: int = 431
    batch_size: int = 1
    precision: str = "fp32"
    autocast: bool = False
    optimizer: str = "AdamW"
    lr: float = 3e-4
    weight_decay: float = 0.0
    loss: str = "upstream-fastmri-SSIMLoss"
    gradient_clipping: bool = False
    num_cascades: int = 12
    chans: int = 18
    pools: int = 4
    sens_chans: int = 8
    sens_pools: int = 4
    acceleration: int = 8
    train_files: int = 85
    slices_per_epoch: int = 2315
    base_epochs: int = 30
    base_max_steps: int = 69450
    scheduler_horizon_epochs: int = 40
    scheduler_max_steps: int = 92600
    ramp_steps: int = 3704
    cosine_decay_start: int = 46300
    checkpoint_file_cadence: int = 1
    status_interval_seconds: int = 300
    activation_checkpoint_feature_cascades: int = 12
    activation_checkpoint_image_cascades: int = 12
    reflect_padding_adapter_schema: str = FI_DETERMINISTIC_REFLECT_PAD_CONTRACT["schema"]
    reflect_padding_adapter_implementation: str = FI_DETERMINISTIC_REFLECT_PAD_CONTRACT[
        "implementation"
    ]
    reflect_padding_adapter_version: str = FI_DETERMINISTIC_REFLECT_PAD_CONTRACT[
        "version"
    ]
    reflect_padding_native_forward_exact: bool = True
    reflect_padding_state_dict_unchanged: bool = True
    reflect_padding_strict_deterministic_algorithms: bool = True

    def as_dict(self):
        return asdict(self)


FI_ACC8_FULL_RECIPE = FIAcc8FullRecipe()


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def recipe_sha256(reflect_padding_adapter):
    receipt = validate_deterministic_reflect_pad_receipt(reflect_padding_adapter)
    value = {
        "recipe": FI_ACC8_FULL_RECIPE.as_dict(),
        "reflect_padding_adapter": receipt,
    }
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def source_binding_sha256(source, reflect_padding_adapter):
    receipt = validate_deterministic_reflect_pad_receipt(reflect_padding_adapter)
    value = {"source": source, "reflect_padding_adapter": receipt}
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _seed_for(*parts):
    digest = hashlib.sha256()
    digest.update(str(FI_ACC8_FULL_RECIPE.seed).encode("ascii"))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def deterministic_file_order(manifest_or_records, epoch):
    if type(epoch) is not int or epoch < 1:
        raise ValueError("FI acc8 epoch must be a positive integer")
    records = getattr(manifest_or_records, "records", manifest_or_records)
    names = [record.name for record in records]
    if len(names) != len(set(names)) or not names:
        raise ValueError("FI acc8 records must have unique nonempty names")
    names.sort()
    random.Random(_seed_for("files", epoch)).shuffle(names)
    return names


def deterministic_slice_order(record, epoch):
    if type(record.slices) is not int or record.slices <= 0:
        raise ValueError("FI acc8 file slice count must be a positive integer")
    order = list(range(record.slices))
    random.Random(_seed_for("slices", epoch, record.name)).shuffle(order)
    return order


@dataclass
class FullSamplerCursor:
    epoch: int
    file_cursor: int
    global_step: int

    def as_dict(self):
        return {
            "epoch": int(self.epoch),
            "file_cursor": int(self.file_cursor),
            "global_step": int(self.global_step),
            "boundary": "verified-file",
        }


def _record_map(records):
    mapping = {record.name: record for record in records}
    if len(mapping) != len(tuple(records)):
        raise ValueError("FI acc8 record names must be unique")
    return mapping


def expected_global_step(records, epoch, file_cursor, epochs):
    if (
        type(epoch) is not int
        or type(file_cursor) is not int
        or type(epochs) is not int
        or epochs <= 0
        or epoch < 1
        or epoch > epochs + 1
        or file_cursor < 0
    ):
        raise ValueError("Invalid verified-file sampler cursor")
    records = tuple(records)
    if epoch == epochs + 1:
        if file_cursor != 0:
            raise ValueError("Completed sampler cursor must point to file zero")
        return sum(record.slices for record in records) * epochs
    order = deterministic_file_order(records, epoch)
    if file_cursor > len(order):
        raise ValueError("Sampler file cursor exceeds epoch inventory")
    mapping = _record_map(records)
    completed_epochs = sum(record.slices for record in records) * (epoch - 1)
    return completed_epochs + sum(mapping[name].slices for name in order[:file_cursor])


def validate_cursor(cursor, records, epochs):
    if isinstance(cursor, FullSamplerCursor):
        value = cursor.as_dict()
    else:
        value = cursor
    required = {"epoch", "file_cursor", "global_step", "boundary"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid full-training sampler cursor schema")
    if value["boundary"] != "verified-file":
        raise ValueError("Full-training resume is allowed only at verified file boundaries")
    for key in ("epoch", "file_cursor", "global_step"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"Invalid full-training sampler {key}")
    expected = expected_global_step(
        tuple(records), value["epoch"], value["file_cursor"], epochs
    )
    if value["global_step"] != expected:
        raise ValueError(
            "Full-training sampler next cursor does not match exact global step"
        )
    return FullSamplerCursor(
        value["epoch"], value["file_cursor"], value["global_step"]
    )


class _VerifiedAcc8FileTransactionMixin:
    """Shared sample/schema mechanics for the concrete transaction."""

    def __init__(self, manifest, record, slice_order):
        self.manifest = manifest
        self.record = record
        self.slice_order = list(slice_order)
        self.receipt = None
        self._stack = None
        self._fds = []
        self._consumed = 0
        self._iterator_started = False

    def __enter__(self):
        expected = list(range(self.record.slices))
        if sorted(self.slice_order) != expected or len(set(self.slice_order)) != len(expected):
            raise ValueError("FI acc8 transaction slice order must be an exact permutation")
        stack = ExitStack()
        self._stack = stack
        try:
            directory_fds, root_identities = _open_directory_chain_nofollow(
                self.manifest.root
            )
            for fd in directory_fds:
                stack.callback(os.close, fd)
            kspace_directory_fd, kspace_identity = _open_child_directory_nofollow(
                directory_fds[-1], "kspace", self.manifest.root / "kspace"
            )
            stack.callback(os.close, kspace_directory_fd)
            image_directory_fd, image_identity = _open_child_directory_nofollow(
                directory_fds[-1], "image", self.manifest.root / "image"
            )
            stack.callback(os.close, image_directory_fd)
            actual_directories = root_identities + (kspace_identity, image_identity)
            if actual_directories != self.manifest.directory_identities:
                raise ValueError("FI acc8 data directory identity changed after inventory")

            kspace_fd, kspace_open = _open_regular_file_nofollow(
                kspace_directory_fd, self.record.name
            )
            stack.callback(os.close, kspace_fd)
            image_fd, image_open = _open_regular_file_nofollow(
                image_directory_fd, self.record.name
            )
            stack.callback(os.close, image_fd)
            self._fds = [kspace_fd, image_fd]
            _require_file_identity(
                kspace_open,
                size=self.record.kspace_size,
                st_dev=self.record.kspace_st_dev,
                st_ino=self.record.kspace_st_ino,
                description="FI acc8 transaction kspace H5",
            )
            _require_file_identity(
                image_open,
                size=self.record.image_size,
                st_dev=self.record.image_st_dev,
                st_ino=self.record.image_st_ino,
                description="FI acc8 transaction image H5",
            )
            kspace_digest, self._kspace_before = _sha256_fd(
                kspace_fd, f"FI acc8 transaction kspace {self.record.name}"
            )
            image_digest, self._image_before = _sha256_fd(
                image_fd, f"FI acc8 transaction image {self.record.name}"
            )
            if (
                kspace_digest != self.record.kspace_sha256
                or image_digest != self.record.image_sha256
            ):
                raise ValueError("FI acc8 transaction H5 bytes changed after inventory")
            self._khf = stack.enter_context(_h5_from_fd(kspace_fd))
            self._ihf = stack.enter_context(_h5_from_fd(image_fd))
            self._validate_h5_contract()
            self._kspace_digest = kspace_digest
            self._image_digest = image_digest
            return self
        except BaseException:
            stack.close()
            self._stack = None
            raise

    def _validate_h5_contract(self):
        khf, ihf = self._khf, self._ihf
        if (
            self.manifest.input_key not in khf
            or "mask" not in khf
            or self.manifest.target_key not in ihf
            or self.manifest.max_key not in ihf.attrs
        ):
            raise ValueError(f"Missing required transaction H5 field in {self.record.name}")
        kspace = khf[self.manifest.input_key]
        target = ihf[self.manifest.target_key]
        mask = khf["mask"]
        if (
            tuple(kspace.shape) != tuple(self.record.kspace_shape)
            or tuple(target.shape) != tuple(self.record.target_shape)
            or kspace.dtype != np.dtype(np.complex64)
            or target.dtype != np.dtype(np.float32)
            or mask.dtype != np.dtype(np.float32)
            or mask.shape != (kspace.shape[-1],)
        ):
            raise ValueError(f"FI acc8 transaction H5 schema changed: {self.record.name}")
        mask_value = np.asarray(mask[...])
        if not np.all(np.isfinite(mask_value)) or not np.all(
            (mask_value == 0.0) | (mask_value == 1.0)
        ):
            raise ValueError(f"FI acc8 transaction stored mask is invalid: {self.record.name}")
        maximum = ihf.attrs[self.manifest.max_key]
        if not math.isfinite(float(maximum)):
            raise ValueError(f"FI acc8 transaction maximum is nonfinite: {self.record.name}")
        self._mask = mask_value
        self._maximum = float(maximum)

    def __iter__(self):
        if self._stack is None or self._iterator_started:
            raise RuntimeError("FI acc8 transaction can be iterated exactly once while open")
        self._iterator_started = True
        kspace_dataset = self._khf[self.manifest.input_key]
        target_dataset = self._ihf[self.manifest.target_key]
        for slice_index in self.slice_order:
            kspace = np.asarray(kspace_dataset[slice_index])
            target = np.asarray(target_dataset[slice_index])
            if (
                kspace.dtype != np.complex64
                or target.dtype != np.float32
                or not np.isfinite(kspace.real).all()
                or not np.isfinite(kspace.imag).all()
                or not np.isfinite(target).all()
            ):
                raise ValueError(
                    f"FI acc8 transaction slice has invalid dtype/value: {self.record.name}"
                )
            masked = np.ascontiguousarray(kspace * self._mask)
            self._consumed += 1
            yield {
                "mask": torch.from_numpy(self._mask.copy()).reshape(
                    1, 1, 1, kspace.shape[-1], 1
                ),
                "kspace": torch.view_as_real(torch.from_numpy(masked)).unsqueeze(0),
                "target": torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0),
                "maximum": torch.tensor([self._maximum], dtype=torch.float32),
                "fname": self.record.name,
                "slice": int(slice_index),
            }

def open_verified_acc8_file(manifest, record, slice_order):
    """Open one selected acc8 pair for an all-slices same-FD transaction."""
    return VerifiedAcc8FileTransaction(manifest, record, slice_order)


class VerifiedAcc8FileTransaction(_VerifiedAcc8FileTransactionMixin):
    """Retain one H5 pair's exact FDs through post-consumption verification."""

    def __enter__(self):
        expected = list(range(self.record.slices))
        if sorted(self.slice_order) != expected or len(set(self.slice_order)) != len(expected):
            raise ValueError("FI acc8 transaction slice order must be an exact permutation")
        self._directory_fds, root_identities = _open_directory_chain_nofollow(
            self.manifest.root
        )
        self._subdir_fds = []
        self._leaf_fds = []
        try:
            kdir, kid = _open_child_directory_nofollow(
                self._directory_fds[-1], "kspace", self.manifest.root / "kspace"
            )
            self._subdir_fds.append(kdir)
            idir, iid = _open_child_directory_nofollow(
                self._directory_fds[-1], "image", self.manifest.root / "image"
            )
            self._subdir_fds.append(idir)
            if root_identities + (kid, iid) != self.manifest.directory_identities:
                raise ValueError("FI acc8 data directory identity changed after inventory")
            kfd, kstat = _open_regular_file_nofollow(kdir, self.record.name)
            self._leaf_fds.append(kfd)
            ifd, istat = _open_regular_file_nofollow(idir, self.record.name)
            self._leaf_fds.append(ifd)
            _require_file_identity(kstat, size=self.record.kspace_size, st_dev=self.record.kspace_st_dev, st_ino=self.record.kspace_st_ino, description="FI acc8 transaction kspace H5")
            _require_file_identity(istat, size=self.record.image_size, st_dev=self.record.image_st_dev, st_ino=self.record.image_st_ino, description="FI acc8 transaction image H5")
            self._kspace_digest, self._kspace_before = _sha256_fd(kfd, f"FI acc8 transaction kspace {self.record.name}")
            self._image_digest, self._image_before = _sha256_fd(ifd, f"FI acc8 transaction image {self.record.name}")
            if self._kspace_digest != self.record.kspace_sha256 or self._image_digest != self.record.image_sha256:
                raise ValueError("FI acc8 transaction H5 bytes changed after inventory")
            self._h5_stack = ExitStack()
            self._khf = self._h5_stack.enter_context(_h5_from_fd(kfd))
            self._ihf = self._h5_stack.enter_context(_h5_from_fd(ifd))
            self._validate_h5_contract()
            self._stack = self._h5_stack
            return self
        except BaseException:
            self._close_all()
            raise

    def _close_all(self):
        if getattr(self, "_h5_stack", None) is not None:
            self._h5_stack.close()
            self._h5_stack = None
        for fd in reversed(getattr(self, "_leaf_fds", [])):
            try:
                os.close(fd)
            except OSError:
                pass
        self._leaf_fds = []
        for fd in reversed(getattr(self, "_subdir_fds", [])):
            try:
                os.close(fd)
            except OSError:
                pass
        self._subdir_fds = []
        for fd in reversed(getattr(self, "_directory_fds", [])):
            try:
                os.close(fd)
            except OSError:
                pass
        self._directory_fds = []
        self._stack = None

    def __exit__(self, exc_type, exc, traceback):
        pending = None
        try:
            if exc_type is None:
                if not self._iterator_started or self._consumed != self.record.slices:
                    raise ValueError(
                        "FI acc8 file transaction must consume all slices before acceptance"
                    )
                self._h5_stack.close()
                self._h5_stack = None
                kafter_digest, kafter = _sha256_fd(
                    self._leaf_fds[0], f"FI acc8 transaction kspace {self.record.name}"
                )
                iafter_digest, iafter = _sha256_fd(
                    self._leaf_fds[1], f"FI acc8 transaction image {self.record.name}"
                )
                if (
                    kafter_digest != self._kspace_digest
                    or iafter_digest != self._image_digest
                    or _stat_signature(kafter) != _stat_signature(self._kspace_before)
                    or _stat_signature(iafter) != _stat_signature(self._image_before)
                ):
                    raise ValueError(
                        f"FI acc8 transaction H5 bytes changed while consuming {self.record.name}"
                    )
                for fd, identity in zip(
                    self._directory_fds,
                    self.manifest.directory_identities[: len(self._directory_fds)],
                ):
                    if not _same_directory_identity(os.fstat(fd), identity):
                        raise ValueError("FI acc8 transaction root identity changed")
                self.receipt = {
                    "name": self.record.name,
                    "accepted": True,
                    "slices": self.record.slices,
                    "slice_order": list(self.slice_order),
                    "kspace_sha256": self._kspace_digest,
                    "image_sha256": self._image_digest,
                    "kspace_st_dev": self.record.kspace_st_dev,
                    "kspace_st_ino": self.record.kspace_st_ino,
                    "image_st_dev": self.record.image_st_dev,
                    "image_st_ino": self.record.image_st_ino,
                }
        except BaseException as error:
            pending = error
        finally:
            self._close_all()
        if pending is not None:
            raise pending
        return False


def run_boundary_engine(
    records,
    cursor,
    step_fn,
    boundary_fn,
    *,
    epoch_limit,
    stop_after_boundaries=None,
):
    """Execute deterministic file/slice order and commit only verified-file cursors."""
    records = tuple(records)
    validated = validate_cursor(cursor, records, epoch_limit)
    if isinstance(cursor, FullSamplerCursor):
        cursor.epoch = validated.epoch
        cursor.file_cursor = validated.file_cursor
        cursor.global_step = validated.global_step
    else:
        cursor = validated
    completed_boundaries = 0
    mapping = _record_map(records)
    while cursor.epoch <= epoch_limit:
        order = deterministic_file_order(records, cursor.epoch)
        while cursor.file_cursor < len(order):
            name = order[cursor.file_cursor]
            record = mapping[name]
            slice_order = deterministic_slice_order(record, cursor.epoch)
            for slice_index in slice_order:
                step_fn(record, slice_index)
                cursor.global_step += 1
            receipt = {
                "name": name,
                "accepted": True,
                "epoch": cursor.epoch,
                "slice_order": slice_order,
                "slices": record.slices,
            }
            cursor.file_cursor += 1
            epoch_completed = cursor.file_cursor == len(order)
            if epoch_completed:
                cursor.epoch += 1
                cursor.file_cursor = 0
            expected = expected_global_step(
                records, cursor.epoch, cursor.file_cursor, epoch_limit
            )
            if cursor.global_step != expected:
                raise RuntimeError("FI acc8 boundary engine global-step drift")
            boundary_fn(copy.deepcopy(cursor), receipt)
            completed_boundaries += 1
            if (
                stop_after_boundaries is not None
                and completed_boundaries >= stop_after_boundaries
            ):
                raise InterruptedError("synthetic interruption after verified file boundary")
            if epoch_completed:
                break
    expected_final = sum(record.slices for record in records) * epoch_limit
    if cursor.global_step != expected_final:
        raise RuntimeError("FI acc8 boundary engine did not reach exact final step")
    return cursor


def run_full_finite_optimizer_step(
    model,
    sample,
    loss_fn,
    optimizer,
    scheduler,
    device,
    *,
    global_step,
):
    """One nominal full-training step: no autocast, clipping, or smoke LR priming."""
    if type(global_step) is not int or global_step < 0:
        raise ValueError("Full-training global step must be a nonnegative integer")
    model.train()
    _assert_finite_parameters(model, "pre-step")
    applied_lr = float(optimizer.param_groups[0]["lr"])
    expected_lr = FI_ACC8_FULL_RECIPE.lr * fi_lr_multiplier(global_step)
    if not math.isfinite(applied_lr) or applied_lr != expected_lr:
        raise ValueError("Full-training optimizer LR does not match nominal scheduler")
    kspace = sample["kspace"].to(device=device, dtype=torch.float32)
    stored_mask = sample["mask"]
    target = sample["target"].to(device=device, dtype=torch.float32)
    maximum = sample["maximum"].to(device=device, dtype=torch.float32)
    if stored_mask.dtype != torch.float32:
        raise ValueError("Stored full-training mask must remain float32")
    expected_mask_shape = (kspace.shape[0], 1, 1, kspace.shape[-2], 1)
    if tuple(stored_mask.shape) != expected_mask_shape:
        raise ValueError("Stored full-training mask shape changed")
    if not torch.isfinite(stored_mask).all() or not torch.all(
        (stored_mask == 0.0) | (stored_mask == 1.0)
    ):
        raise ValueError("Stored full-training mask must be finite and binary")
    mask = stored_mask.to(device=device, dtype=torch.bool)
    if torch.count_nonzero(kspace.masked_select(~mask)).item() != 0:
        raise ValueError("Stored full-training mask was not applied to kspace")
    output = model(kspace, mask, crop_size=tuple(target.shape[-2:]))
    target, output = _center_crop_to_smallest(target, output)
    if not torch.isfinite(output).all():
        raise FloatingPointError("nonfinite full-training model output")
    loss = loss_fn(output.unsqueeze(1), target.unsqueeze(1), maximum)
    if loss.numel() != 1 or not torch.isfinite(loss.detach()).all():
        raise FloatingPointError("nonfinite full-training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient_count += 1
            if not torch.isfinite(parameter.grad.detach()).all():
                raise FloatingPointError(f"nonfinite full-training gradient: {name}")
    if gradient_count == 0:
        raise FloatingPointError("full training produced no gradients")
    optimizer.step()
    _assert_finite_parameters(model, "post-step")
    scheduler.step()
    post_step_lr = float(optimizer.param_groups[0]["lr"])
    expected_post = FI_ACC8_FULL_RECIPE.lr * fi_lr_multiplier(global_step + 1)
    if not math.isfinite(post_step_lr) or post_step_lr != expected_post:
        raise ValueError("Full-training scheduler advanced incorrectly")
    return {
        "loss": float(loss.detach().cpu()),
        "global_step": global_step + 1,
        "applied_lr": applied_lr,
        "post_step_lr": post_step_lr,
        "finite_loss": True,
        "finite_gradients": True,
        "finite_parameters": True,
        "gradient_parameter_count": gradient_count,
        "fname": sample["fname"],
        "slice": int(sample["slice"]),
    }


def _capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng(rng, device):
    random.setstate(rng["python"])
    numpy = rng["numpy"]
    np.random.set_state(
        (
            numpy["name"],
            numpy["keys"].cpu().numpy().astype(np.uint32, copy=False),
            numpy["position"],
            numpy["has_gauss"],
            numpy["cached_gaussian"],
        )
    )
    torch.set_rng_state(rng["torch_cpu"].cpu())
    device = torch.device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        if len(rng["torch_cuda"]) <= index:
            raise ValueError("Full checkpoint CUDA RNG topology does not match GPU")
        torch.cuda.set_rng_state(rng["torch_cuda"][index].cpu(), device=index)


def _validate_sha(value, description):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Invalid full checkpoint {description} SHA-256")


def _validate_bindings(bindings):
    required = {
        "source_sha256",
        "data_manifest_sha256",
        "recipe_sha256",
        "gpu_uuid",
        "reflect_padding_adapter",
    }
    if not isinstance(bindings, dict) or set(bindings) != required:
        raise ValueError("Invalid full checkpoint binding schema")
    for key in ("source_sha256", "data_manifest_sha256", "recipe_sha256"):
        _validate_sha(bindings[key], key)
    if not isinstance(bindings["gpu_uuid"], str) or not bindings["gpu_uuid"]:
        raise ValueError("Invalid full checkpoint GPU binding")
    validate_deterministic_reflect_pad_receipt(
        bindings["reflect_padding_adapter"]
    )


def _validate_state_mapping(value, description):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Full checkpoint {description} must be a nonempty mapping")


def _validate_finite_tensors(value, description):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"Full checkpoint {description} contains a nonfinite tensor")
    elif isinstance(value, dict):
        for nested in value.values():
            _validate_finite_tensors(nested, description)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_finite_tensors(nested, description)


def _validate_checkpoint_training_invariants(checkpoint, records, epochs):
    records = tuple(records)
    cursor = validate_cursor(checkpoint["sampler"], records, epochs)
    global_step = cursor.global_step
    metrics = checkpoint["metrics"]
    if metrics["loss_count"] != global_step:
        raise ValueError("Full checkpoint loss_count must equal sampler global_step")
    scheduler = checkpoint["scheduler"]
    if scheduler.get("last_epoch") != global_step:
        raise ValueError("Full checkpoint scheduler last_epoch must equal global_step")
    if scheduler.get("_step_count") != global_step + 1:
        raise ValueError("Full checkpoint scheduler _step_count must equal global_step + 1")

    optimizer = checkpoint["optimizer"]
    expected_lr = FI_ACC8_FULL_RECIPE.lr * fi_lr_multiplier(global_step)
    for group in optimizer["param_groups"]:
        lr = group.get("lr")
        if type(lr) is not float or not math.isfinite(lr) or lr != expected_lr:
            raise ValueError("Full checkpoint optimizer LR does not match nominal global_step")
    for parameter_state in optimizer["state"].values():
        step = parameter_state.get("step")
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise ValueError("Full checkpoint AdamW state.step must be scalar")
            step = step.item()
        if step != global_step:
            raise ValueError("Full checkpoint AdamW state.step must equal global_step")

    completed_files = (cursor.epoch - 1) * len(records) + cursor.file_cursor
    if len(checkpoint["transactions"]) != completed_files:
        raise ValueError("Full checkpoint transaction count does not match sampler cursor")
    expected_transactions = []
    remaining = completed_files
    mapping = _record_map(records)
    for epoch in range(1, cursor.epoch + 1):
        for name in deterministic_file_order(records, epoch):
            if remaining == 0:
                break
            record = mapping[name]
            expected_transactions.append(
                (epoch, name, record.slices, deterministic_slice_order(record, epoch))
            )
            remaining -= 1
        if remaining == 0:
            break
    for transaction, expected in zip(checkpoint["transactions"], expected_transactions):
        epoch, name, slices, slice_order = expected
        if (
            transaction.get("accepted") is not True
            or transaction.get("epoch") != epoch
            or transaction.get("name") != name
            or transaction.get("slices") != slices
            or transaction.get("slice_order") != slice_order
        ):
            raise ValueError("Full checkpoint transaction sequence does not match sampler cursor")
    _validate_finite_tensors(checkpoint["model"], "model")
    _validate_finite_tensors(checkpoint["optimizer"], "optimizer")


def validate_full_checkpoint(
    checkpoint,
    *,
    records=None,
    epochs=FI_ACC8_FULL_RECIPE.base_epochs,
):
    required = {
        "format_version",
        "kind",
        "scope",
        "resumable",
        "full_training_authorized",
        "evaluation_authorized",
        "submission_authorized",
        "recipe",
        "activation_checkpointing",
        "reflect_padding_adapter",
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "sampler",
        "bindings",
        "provenance",
        "transactions",
        "metrics",
        "runtime",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != required:
        raise ValueError("Invalid FI acc8 full checkpoint top-level schema")
    if (
        checkpoint["format_version"] != 2
        or checkpoint["kind"] != "fi-varnet-acc8-checkpointed-full-training"
        or checkpoint["scope"] != "FULL_TRAINING_ONLY"
        or checkpoint["resumable"] is not True
        or checkpoint["full_training_authorized"] is not True
        or checkpoint["evaluation_authorized"] is not False
        or checkpoint["submission_authorized"] is not False
        or checkpoint["recipe"] != FI_ACC8_FULL_RECIPE.as_dict()
    ):
        raise ValueError("Invalid FI acc8 full checkpoint scope or recipe")
    _validate_activation_checkpoint_contract(checkpoint["activation_checkpointing"])
    validate_deterministic_reflect_pad_receipt(checkpoint["reflect_padding_adapter"])
    _validate_state_mapping(checkpoint["model"], "model")
    if not isinstance(checkpoint["optimizer"], dict) or set(checkpoint["optimizer"]) != {"state", "param_groups"}:
        raise ValueError("Full checkpoint optimizer schema is invalid")
    if not isinstance(checkpoint["optimizer"]["param_groups"], list) or not checkpoint["optimizer"]["param_groups"]:
        raise ValueError("Full checkpoint optimizer groups are invalid")
    _validate_state_mapping(checkpoint["scheduler"], "scheduler")
    rng = checkpoint["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Full checkpoint RNG schema is invalid")
    if not torch.is_tensor(rng["torch_cpu"]) or rng["torch_cpu"].dtype != torch.uint8:
        raise ValueError("Full checkpoint CPU RNG is invalid")
    if not isinstance(rng["torch_cuda"], list) or any(
        not torch.is_tensor(item) or item.dtype != torch.uint8
        for item in rng["torch_cuda"]
    ):
        raise ValueError("Full checkpoint CUDA RNG is invalid")
    numpy = rng["numpy"]
    if not isinstance(numpy, dict) or set(numpy) != {"name", "keys", "position", "has_gauss", "cached_gaussian"}:
        raise ValueError("Full checkpoint NumPy RNG is invalid")
    _validate_bindings(checkpoint["bindings"])
    if (
        checkpoint["bindings"]["reflect_padding_adapter"]
        != checkpoint["reflect_padding_adapter"]
    ):
        raise ValueError("Full checkpoint reflect-padding adapter binding disagrees")
    provenance = checkpoint["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "source", "data", "recipe", "gpu", "reflect_padding_adapter"
    }:
        raise ValueError("Full checkpoint provenance schema is invalid")
    if (
        not isinstance(provenance["source"], dict)
        or not isinstance(provenance["data"], dict)
        or provenance["recipe"] != FI_ACC8_FULL_RECIPE.as_dict()
        or provenance["gpu"] != {"uuid": checkpoint["bindings"]["gpu_uuid"]}
        or provenance["reflect_padding_adapter"]
        != checkpoint["reflect_padding_adapter"]
    ):
        raise ValueError("Full checkpoint provenance does not match bindings/recipe")
    if not isinstance(checkpoint["transactions"], list):
        raise ValueError("Full checkpoint transactions must be a list")
    for transaction in checkpoint["transactions"]:
        if not isinstance(transaction, dict) or transaction.get("accepted") is not True or not isinstance(transaction.get("name"), str):
            raise ValueError("Full checkpoint contains an unaccepted file transaction")
    metrics = checkpoint["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {"loss_sum", "loss_count"}:
        raise ValueError("Full checkpoint metrics schema is invalid")
    if type(metrics["loss_sum"]) is not float or not math.isfinite(metrics["loss_sum"]):
        raise ValueError("Full checkpoint loss sum must be finite")
    if type(metrics["loss_count"]) is not int or metrics["loss_count"] < 0:
        raise ValueError("Full checkpoint loss count must be nonnegative")
    runtime = checkpoint["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"pid", "gpu_uuid", "peak_vram_bytes", "elapsed_seconds"}:
        raise ValueError("Full checkpoint runtime schema is invalid")
    if type(runtime["pid"]) is not int or runtime["pid"] <= 0:
        raise ValueError("Full checkpoint PID is invalid")
    if runtime["gpu_uuid"] != checkpoint["bindings"]["gpu_uuid"]:
        raise ValueError("Full checkpoint runtime GPU binding disagrees")
    if type(runtime["peak_vram_bytes"]) is not int or runtime["peak_vram_bytes"] < 0:
        raise ValueError("Full checkpoint peak VRAM is invalid")
    if type(runtime["elapsed_seconds"]) is not float or not math.isfinite(runtime["elapsed_seconds"]) or runtime["elapsed_seconds"] < 0:
        raise ValueError("Full checkpoint elapsed time is invalid")
    if records is not None:
        _validate_checkpoint_training_invariants(checkpoint, records, epochs)
    return checkpoint


def build_full_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    cursor,
    records,
    bindings,
    transactions,
    metrics,
    reflect_padding_adapter,
    provenance=None,
    runtime=None,
    activation_checkpointing=None,
):
    _validate_bindings(bindings)
    reflect_padding_adapter = validate_deterministic_reflect_pad_receipt(
        reflect_padding_adapter
    )
    if bindings["reflect_padding_adapter"] != reflect_padding_adapter:
        raise ValueError("Full checkpoint reflect-padding adapter binding disagrees")
    if isinstance(cursor, FullSamplerCursor):
        sampler = cursor.as_dict()
    else:
        sampler = dict(cursor)
    if runtime is None:
        runtime = {
            "pid": os.getpid(),
            "gpu_uuid": bindings["gpu_uuid"],
            "peak_vram_bytes": 0,
            "elapsed_seconds": 0.0,
        }
    if activation_checkpointing is None:
        activation_checkpointing = dict(FI_ACTIVATION_CHECKPOINT_CONTRACT)
    if provenance is None:
        provenance = {
            "source": {"sha256": bindings["source_sha256"]},
            "data": {"manifest_sha256": bindings["data_manifest_sha256"]},
            "recipe": FI_ACC8_FULL_RECIPE.as_dict(),
            "gpu": {"uuid": bindings["gpu_uuid"]},
            "reflect_padding_adapter": dict(reflect_padding_adapter),
        }
    checkpoint = {
        "format_version": 2,
        "kind": "fi-varnet-acc8-checkpointed-full-training",
        "scope": "FULL_TRAINING_ONLY",
        "resumable": True,
        "full_training_authorized": True,
        "evaluation_authorized": False,
        "submission_authorized": False,
        "recipe": FI_ACC8_FULL_RECIPE.as_dict(),
        "activation_checkpointing": _validate_activation_checkpoint_contract(
            activation_checkpointing
        ),
        "reflect_padding_adapter": dict(reflect_padding_adapter),
        "model": _cpu_snapshot(model.state_dict()),
        "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "scheduler": _cpu_snapshot(scheduler.state_dict()),
        "rng": _cpu_snapshot(_capture_rng()),
        "sampler": _cpu_snapshot(sampler),
        "bindings": _cpu_snapshot(dict(bindings)),
        "provenance": _cpu_snapshot(dict(provenance)),
        "transactions": _cpu_snapshot(list(transactions)),
        "metrics": _cpu_snapshot(dict(metrics)),
        "runtime": _cpu_snapshot(dict(runtime)),
    }
    return validate_full_checkpoint(checkpoint, records=records)


def _snapshot_add(digest, value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        _snapshot_add(digest, "tensor")
        _snapshot_add(digest, str(tensor.dtype))
        _snapshot_add(digest, list(tensor.shape))
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    elif isinstance(value, dict):
        _snapshot_add(digest, "dict")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _snapshot_add(digest, key)
            _snapshot_add(digest, value[key])
    elif isinstance(value, (list, tuple)):
        _snapshot_add(digest, type(value).__name__)
        for item in value:
            _snapshot_add(digest, item)
    elif value is None:
        _snapshot_add(digest, "none")
    elif isinstance(value, (str, bool, int, float)):
        payload = (type(value).__name__ + ":" + repr(value)).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    else:
        raise TypeError(f"Unsupported exact snapshot type: {type(value).__name__}")


def snapshot_bytes(value):
    digest = hashlib.sha256()
    _snapshot_add(digest, value)
    return digest.digest()


def _sha256_handle(handle):
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _secure_load_checkpoint(path, expected_sha256):
    _validate_sha(expected_sha256, "expected checkpoint")
    path = Path(path)
    directory_fds, _ = _open_directory_chain_nofollow(path.parent)
    directory_fd = directory_fds[-1]
    try:
        try:
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ValueError("Cannot securely open FI acc8 full checkpoint") from exc
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("FI acc8 full checkpoint must be regular")
            actual = _sha256_handle(handle)
            if actual != expected_sha256:
                raise ValueError(
                    f"Full checkpoint SHA-256 mismatch: expected={expected_sha256} actual={actual}"
                )
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    return validate_full_checkpoint(checkpoint)


def _require_fresh_cpu_training_state(model, optimizer, scheduler):
    if any(tensor.device.type != "cpu" for tensor in model.state_dict().values()):
        raise ValueError("Exact resume requires a fresh CPU model")
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    if optimizer_state["state"]:
        raise ValueError("Exact resume requires a fresh optimizer")
    if (
        scheduler_state.get("last_epoch") != 0
        or scheduler_state.get("_step_count") != 1
        or any(
            group.get("lr") != FI_ACC8_FULL_RECIPE.lr * fi_lr_multiplier(0)
            for group in optimizer_state["param_groups"]
        )
    ):
        raise ValueError("Exact resume requires a fresh scheduler and nominal LR0")


def _move_optimizer_state_to_device(optimizer, device):
    def move(value):
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: move(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [move(nested) for nested in value]
        if isinstance(value, tuple):
            return tuple(move(nested) for nested in value)
        return value

    for parameter, state in tuple(optimizer.state.items()):
        optimizer.state[parameter] = move(state)


def load_full_checkpoint(
    path,
    *,
    expected_sha256,
    expected_bindings,
    records,
    model,
    optimizer,
    scheduler,
    device,
    epochs=FI_ACC8_FULL_RECIPE.base_epochs,
):
    """Validate into fresh CPU objects, then perform the sole live-device move."""
    _require_fresh_cpu_training_state(model, optimizer, scheduler)
    checkpoint = _secure_load_checkpoint(path, expected_sha256)
    _validate_bindings(expected_bindings)
    if checkpoint["bindings"] != expected_bindings:
        raise ValueError("Full checkpoint binding does not match source/data/recipe/GPU")
    validate_full_checkpoint(checkpoint, records=records, epochs=epochs)

    live_model = model.state_dict()
    if set(live_model) != set(checkpoint["model"]):
        raise ValueError("Full checkpoint model keys do not match live model")
    if any(
        live_model[key].shape != checkpoint["model"][key].shape
        or live_model[key].dtype != checkpoint["model"][key].dtype
        for key in live_model
    ):
        raise ValueError("Full checkpoint model tensors do not match live model")
    live_groups = optimizer.state_dict()["param_groups"]
    saved_groups = checkpoint["optimizer"]["param_groups"]
    if [len(group["params"]) for group in live_groups] != [
        len(group["params"]) for group in saved_groups
    ]:
        raise ValueError("Full checkpoint optimizer topology does not match")

    try:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Full checkpoint state is not restorable into fresh CPU state") from exc

    device = torch.device(device)
    model.to(device=device, dtype=torch.float32)
    _move_optimizer_state_to_device(optimizer, device)
    _restore_rng(checkpoint["rng"], device)
    return checkpoint


def _write_file_fsync(path, payload, mode=0o444):
    with Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), mode)


def _validate_checkpoint_pointer(pointer):
    """Validate the complete bounded durable checkpoint authority graph."""
    required = {"format_version", "latest", "previous", "epoch_generations"}
    if (
        not isinstance(pointer, dict)
        or set(pointer) != required
        or type(pointer["format_version"]) is not int
        or pointer["format_version"] != 1
    ):
        raise ValueError("Invalid FI acc8 checkpoint pointer schema/version")

    epoch_generations = pointer["epoch_generations"]
    if not isinstance(epoch_generations, list) or len(epoch_generations) > 30:
        raise ValueError("Invalid FI acc8 checkpoint pointer epoch lineage length")

    entry_keys = {"generation", "checkpoint_sha256", "sampler"}
    sampler_keys = {"epoch", "file_cursor", "global_step", "boundary"}

    def validate_entry(entry):
        if not isinstance(entry, dict) or set(entry) != entry_keys:
            raise ValueError(
                "Invalid FI acc8 checkpoint pointer generation/SHA/sampler entry"
            )
        if not isinstance(entry["generation"], str) or re.fullmatch(
            r"generation-[0-9a-f]{32}", entry["generation"]
        ) is None:
            raise ValueError(
                "Invalid FI acc8 checkpoint pointer generation/SHA/sampler entry"
            )
        if not isinstance(entry["checkpoint_sha256"], str) or re.fullmatch(
            r"[0-9a-f]{64}", entry["checkpoint_sha256"]
        ) is None:
            raise ValueError(
                "Invalid FI acc8 checkpoint pointer generation/SHA/sampler entry"
            )
        sampler = entry["sampler"]
        if (
            not isinstance(sampler, dict)
            or set(sampler) != sampler_keys
            or sampler["boundary"] != "verified-file"
            or type(sampler["epoch"]) is not int
            or type(sampler["file_cursor"]) is not int
            or type(sampler["global_step"]) is not int
            or sampler["epoch"] < 1
            or sampler["file_cursor"] < 0
            or sampler["global_step"] < 0
        ):
            raise ValueError(
                "Invalid FI acc8 checkpoint pointer generation/SHA/sampler entry"
            )
        return entry

    latest = validate_entry(pointer["latest"])
    previous = pointer["previous"]
    if previous is not None:
        previous = validate_entry(previous)
        if previous["generation"] == latest["generation"]:
            raise ValueError(
                "Invalid FI acc8 checkpoint pointer duplicate current generation"
            )

    epoch_names = set()
    prior_epoch = None
    for entry in epoch_generations:
        validate_entry(entry)
        sampler = entry["sampler"]
        epoch = sampler["epoch"]
        if (
            sampler["file_cursor"] != 0
            or not 2 <= epoch <= 31
            or sampler["global_step"]
            != (epoch - 1) * FI_ACC8_FULL_RECIPE.slices_per_epoch
            or (prior_epoch is not None and epoch <= prior_epoch)
            or entry["generation"] in epoch_names
        ):
            raise ValueError("Invalid FI acc8 checkpoint pointer epoch lineage")
        prior_epoch = epoch
        epoch_names.add(entry["generation"])

    all_entries = [latest]
    if previous is not None:
        all_entries.append(previous)
    all_entries.extend(epoch_generations)
    entries_by_generation = {}
    for entry in all_entries:
        generation = entry["generation"]
        if generation in entries_by_generation and entries_by_generation[generation] != entry:
            raise ValueError(
                "Invalid FI acc8 checkpoint pointer conflicting generation entry"
            )
        entries_by_generation[generation] = entry
    if len(entries_by_generation) > FI_ACC8_RETAINED_CHECKPOINT_LIMIT:
        raise ValueError("Invalid FI acc8 checkpoint pointer retained generation count")
    return pointer


def _load_pointer(root):
    path = Path(root) / "checkpoint-current.json"
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return None
    pointer = json.loads(payload)
    return _validate_checkpoint_pointer(pointer)


def _same_open_inode(fd, metadata):
    opened = os.fstat(fd)
    return (
        opened.st_dev == metadata.st_dev
        and opened.st_ino == metadata.st_ino
        and stat.S_IFMT(opened.st_mode) == stat.S_IFMT(metadata.st_mode)
    )


def _identity_bound_rename_unlink(
    directory_fd, source_name, retained_fd, *, alias_prefix
):
    """Rename one still-owned name private, rebind its inode, then unlink it."""
    retained = os.fstat(retained_fd)
    try:
        current = os.stat(source_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Cleanup source {source_name!r} disappeared; retained inode preserved"
        ) from exc
    if not _same_open_inode(retained_fd, current):
        raise RuntimeError(
            f"Cleanup source {source_name!r} changed identity; replacement preserved"
        )
    alias = f"{alias_prefix}{uuid.uuid4().hex}"
    os.rename(
        source_name,
        alias,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    try:
        alias_fd = os.open(
            alias,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Private cleanup alias {alias!r} could not be identity-bound; preserved"
        ) from exc
    try:
        if not _same_open_inode(alias_fd, retained):
            raise RuntimeError(
                f"Private cleanup alias {alias!r} changed identity; replacement preserved"
            )
    finally:
        os.close(alias_fd)
    os.unlink(alias, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _cleanup_failed_atomic_stage(directory_fd, stage_name, stage_fd):
    _identity_bound_rename_unlink(
        directory_fd,
        stage_name,
        stage_fd,
        alias_prefix=".cleanup-",
    )


def _atomic_replace_bytes(path, payload):
    path = Path(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=f"/proc/self/fd/{directory_fd}",
            prefix=f".{path.name}-unpublished-",
            delete=False,
        ) as handle:
            stage_name = Path(handle.name).name
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(
                    stage_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
                os.fsync(directory_fd)
            except BaseException as error:
                try:
                    _cleanup_failed_atomic_stage(
                        directory_fd, stage_name, handle.fileno()
                    )
                except BaseException as cleanup_error:
                    note = (
                        "Atomic stage cleanup was not identity-safe; pathname preserved: "
                        f"{cleanup_error}"
                    )
                    setattr(error, "__notes__", [*getattr(error, "__notes__", ()), note])
                raise
    finally:
        os.close(directory_fd)


def _purge_directory_fd(directory_fd):
    """Delete only the two known files from an already-private retired generation."""
    expected = {"checkpoint.pt", "metadata.json"}
    names = set(os.listdir(directory_fd))
    if names != expected:
        raise ValueError(
            "Checkpoint generation must contain exactly checkpoint.pt and metadata.json"
        )
    retained = {}
    try:
        for name in sorted(expected):
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise ValueError(
                    f"Checkpoint generation {name} must be a nofollow regular file"
                ) from exc
            retained[name] = fd
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(
                    f"Checkpoint generation {name} must be a regular file"
                )
        for name in sorted(expected):
            _identity_bound_rename_unlink(
                directory_fd,
                name,
                retained[name],
                alias_prefix=f".delete-{name}-",
            )
    finally:
        for fd in retained.values():
            os.close(fd)


def _retire_generation_locked(generations_dir, generation):
    """Under the private-root flock, rename first; never unlink its old name."""
    if (
        not isinstance(generation, str)
        or not generation.startswith("generation-")
        or "/" in generation
        or generation in {".", ".."}
    ):
        raise ValueError("Invalid checkpoint generation basename")
    generations_fd = os.open(
        generations_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    retired_name = f".retired-{uuid.uuid4().hex}"
    try:
        try:
            os.rename(
                generation,
                retired_name,
                src_dir_fd=generations_fd,
                dst_dir_fd=generations_fd,
            )
        except FileNotFoundError:
            return
        os.fsync(generations_fd)
        fd = os.open(
            retired_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=generations_fd,
        )
        try:
            os.fchmod(fd, 0o700)
            _purge_directory_fd(fd)
        finally:
            os.close(fd)
        os.rmdir(retired_name, dir_fd=generations_fd)
        os.fsync(generations_fd)
    finally:
        os.close(generations_fd)


def publish_full_checkpoint(root, checkpoint, *, epoch_end):
    """Commit an immutable generation, then atomically switch latest/previous."""
    checkpoint = validate_full_checkpoint(checkpoint)
    if type(epoch_end) is not bool:
        raise ValueError("epoch_end must be a bool")
    if epoch_end and (
        checkpoint["sampler"]["file_cursor"] != 0
        or checkpoint["sampler"]["epoch"] <= 1
    ):
        raise ValueError("epoch_end requires a completed epoch cursor")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generations = root / "checkpoint-generations"
    generations.mkdir(exist_ok=True)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    staged = None
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        previous_pointer = _load_pointer(root)
        if epoch_end and previous_pointer is not None:
            completed_epoch_cursor = checkpoint["sampler"]["epoch"]
            if any(
                isinstance(item, dict)
                and isinstance(item.get("sampler"), dict)
                and item["sampler"].get("epoch") == completed_epoch_cursor
                for item in previous_pointer["epoch_generations"]
            ):
                raise ValueError(
                    f"Epoch {completed_epoch_cursor - 1} checkpoint is already retained"
                )
        generation = f"generation-{uuid.uuid4().hex}"
        final_generation = generations / generation
        staged = _create_staged_directory(final_generation, "FI acc8 checkpoint")
        stage_path = _staged_directory_descriptor_path(staged)
        checkpoint_path = stage_path / "checkpoint.pt"
        with checkpoint_path.open("x+b") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            reloaded = torch.load(handle, map_location="cpu", weights_only=True)
            validate_full_checkpoint(reloaded)
            if snapshot_bytes(reloaded) != snapshot_bytes(checkpoint):
                raise ValueError("Full checkpoint restricted round-trip mismatch")
            os.fchmod(handle.fileno(), 0o444)
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        metadata = {
            "schema": "fi-varnet-acc8-checkpoint-generation-v1",
            "generation": generation,
            "checkpoint_sha256": digest,
            "sampler": checkpoint["sampler"],
            "epoch_end": epoch_end,
            "metrics": checkpoint["metrics"],
        }
        _write_file_fsync(
            stage_path / "metadata.json", _canonical_json(metadata) + b"\n"
        )
        _seal_staged_directory(staged)
        _publish_staged_directory_no_replace(
            staged, final_generation, "FI acc8 checkpoint"
        )
        staged = None
        entry = {
            "generation": generation,
            "checkpoint_sha256": digest,
            "sampler": checkpoint["sampler"],
        }
        old_latest = previous_pointer["latest"] if previous_pointer else None
        epoch_generations = (
            list(previous_pointer["epoch_generations"]) if previous_pointer else []
        )
        if epoch_end:
            epoch_generations.append(entry)
        pointer = {
            "format_version": 1,
            "latest": entry,
            "previous": old_latest,
            "epoch_generations": epoch_generations,
        }
        _validate_checkpoint_pointer(pointer)
        _atomic_replace_bytes(
            root / "checkpoint-current.json", _canonical_json(pointer) + b"\n"
        )
        keep = {entry["generation"]}
        if old_latest is not None:
            keep.add(old_latest["generation"])
        keep.update(item["generation"] for item in epoch_generations)
        for child in list(generations.iterdir()):
            if child.name.startswith("generation-") and child.name not in keep:
                _retire_generation_locked(generations, child.name)
        os.fsync(root_fd)
    finally:
        if staged is not None:
            _cleanup_staged_directory(staged)
        os.close(root_fd)
    final_checkpoint = final_generation / "checkpoint.pt"
    return {
        "generation": generation,
        "checkpoint_path": final_checkpoint,
        "sha256": digest,
        "pointer_path": root / "checkpoint-current.json",
    }


_STATUS_KEYS = {
    "schema",
    "authoritative",
    "phase",
    "pid",
    "gpu_uuid",
    "gpu_name",
    "gpu_index",
    "vram_allocated_bytes",
    "vram_reserved_bytes",
    "vram_peak_bytes",
    "epoch",
    "file_cursor",
    "file",
    "slice",
    "global_step",
    "nominal_lr",
    "moving_loss",
    "throughput_steps_per_second",
    "eta_seconds",
    "last_checkpoint_path",
    "last_checkpoint_sha256",
    "command_argv",
    "deterministic_contract",
    "reflect_padding_adapter",
    "updated_unix_seconds",
}


def validate_status(status):
    if not isinstance(status, dict) or set(status) != _STATUS_KEYS:
        raise ValueError("Invalid FI acc8 full-training status schema")
    if status["schema"] != "fi-varnet-acc8-full-training-status-v2":
        raise ValueError("Invalid FI acc8 full-training status version")
    if status["authoritative"] is not False:
        raise ValueError("FI acc8 full-training status must be nonauthoritative")
    if status["phase"] not in {"training", "checkpointed", "complete"}:
        raise ValueError("Invalid FI acc8 full-training status phase")
    for key in (
        "pid",
        "gpu_index",
        "vram_allocated_bytes",
        "vram_reserved_bytes",
        "vram_peak_bytes",
        "epoch",
        "file_cursor",
        "slice",
        "global_step",
    ):
        minimum = 1 if key in {"pid", "epoch"} else 0
        if type(status[key]) is not int or status[key] < minimum:
            raise ValueError(f"Invalid FI acc8 full-training status {key}")
    for key in ("gpu_uuid", "gpu_name", "file"):
        if not isinstance(status[key], str) or not status[key]:
            raise ValueError(f"Invalid FI acc8 full-training status {key}")
    for key in (
        "nominal_lr",
        "moving_loss",
        "throughput_steps_per_second",
        "eta_seconds",
        "updated_unix_seconds",
    ):
        if (
            type(status[key]) is not float
            or not math.isfinite(status[key])
            or status[key] < 0
        ):
            raise ValueError(f"Invalid FI acc8 full-training status {key}")
    checkpoint_path = status["last_checkpoint_path"]
    checkpoint_sha = status["last_checkpoint_sha256"]
    if not isinstance(checkpoint_path, str) or not isinstance(checkpoint_sha, str):
        raise ValueError("Invalid FI acc8 full-training status checkpoint")
    if bool(checkpoint_path) != bool(checkpoint_sha):
        raise ValueError("FI acc8 status checkpoint path and SHA must be paired")
    if checkpoint_path:
        if not Path(checkpoint_path).is_absolute():
            raise ValueError("FI acc8 status checkpoint path must be absolute")
        try:
            _validate_sha(checkpoint_sha, "status checkpoint")
        except ValueError as exc:
            raise ValueError("Invalid FI acc8 full-training status checkpoint SHA") from exc
    command = status["command_argv"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("Invalid FI acc8 full-training status command argv")
    if status["deterministic_contract"] != FI_ACC8_DETERMINISM_CONTRACT:
        raise ValueError("Invalid FI acc8 full-training status deterministic contract")
    validate_deterministic_reflect_pad_receipt(
        status["reflect_padding_adapter"]
    )
    return status


def atomic_write_status(path, status):
    validate_status(status)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace_bytes(path, _canonical_json(status) + b"\n")


# The orchestration entrypoint is appended below the independently testable
# integrity/resume/publication primitives. Heavy model/CUDA work remains lazy.


def _check_full_run_root_state(output_dir, *, resume):
    """Read-only collision/resume gate; it never reserves output bytes."""
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    try:
        current = os.lstat(output_dir)
    except FileNotFoundError:
        current = None
    if resume:
        if current is None or not stat.S_ISDIR(current.st_mode):
            raise FileNotFoundError("Exact resume requires the existing run directory")
        names = set(os.listdir(output_dir))
        if "INCOMPLETE" not in names or "COMPLETE" in names:
            raise ValueError("Exact resume requires INCOMPLETE and forbids COMPLETE")
    elif current is not None:
        raise FileExistsError(f"FI acc8 full-training output already exists: {output_dir}")
    return output_dir


def _check_resume_checkpoint_reference(output_dir, checkpoint_path, checkpoint_sha256):
    """Read-only gate for one pointer-authorized, in-root durable generation."""
    _validate_sha(checkpoint_sha256, "expected checkpoint")
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    checkpoint_path = Path(os.path.abspath(os.fspath(checkpoint_path)))
    expected_generations = output_dir / "checkpoint-generations"
    generation = checkpoint_path.parent.name
    if (
        checkpoint_path.name != "checkpoint.pt"
        or checkpoint_path.parent.parent != expected_generations
        or not generation.startswith("generation-")
        or generation == "generation-"
    ):
        raise ValueError("Resume checkpoint must be a generation in this run root")
    pointer = _load_pointer(output_dir)
    if pointer is None:
        raise ValueError("Exact resume requires a durable checkpoint pointer")
    entries = [pointer["latest"], pointer["previous"]]
    authorized_entry = next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("sampler"), dict)
            and entry.get("generation") == generation
            and entry.get("checkpoint_sha256") == checkpoint_sha256
        ),
        None,
    )
    if authorized_entry is None:
        raise ValueError(
            "Exact resume checkpoint must match pointer latest or previous generation/SHA/sampler"
        )
    metadata_path = checkpoint_path.with_name("metadata.json")
    try:
        generation_metadata = json.loads(metadata_path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("Pointer-authorized resume generation metadata is invalid") from exc
    if (
        not isinstance(generation_metadata, dict)
        or generation_metadata.get("generation") != authorized_entry.get("generation")
        or generation_metadata.get("checkpoint_sha256")
        != authorized_entry.get("checkpoint_sha256")
        or generation_metadata.get("sampler") != authorized_entry.get("sampler")
    ):
        raise ValueError(
            "Exact resume checkpoint must match pointer latest or previous generation/SHA/sampler"
        )
    try:
        metadata = os.lstat(checkpoint_path)
    except FileNotFoundError as exc:
        raise ValueError("Pointer-authorized resume checkpoint is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Pointer-authorized resume checkpoint must be regular")
    return checkpoint_path


def _prepare_full_run_root(output_dir, *, resume):
    output_dir = _check_full_run_root_state(output_dir, resume=resume)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    parent_fds, _ = _open_directory_chain_nofollow(output_dir.parent)
    try:
        parent_fd = parent_fds[-1]
        try:
            current = os.stat(output_dir.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if resume:
            if current is None or not stat.S_ISDIR(current.st_mode):
                raise FileNotFoundError("Exact resume requires the existing run directory")
        elif current is not None:
            raise FileExistsError(f"FI acc8 full-training output already exists: {output_dir}")
        else:
            os.mkdir(output_dir.name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
    finally:
        for fd in reversed(parent_fds):
            os.close(fd)
    run_fds, _ = _open_directory_chain_nofollow(output_dir)
    try:
        names = set(os.listdir(run_fds[-1]))
        if resume:
            if "INCOMPLETE" not in names or "COMPLETE" in names:
                raise ValueError("Exact resume requires INCOMPLETE and forbids COMPLETE")
        else:
            _write_file_fsync(
                output_dir / "INCOMPLETE",
                b"FI-VARNET-ACC8-FULL-TRAINING-INCOMPLETE\n",
            )
            os.fsync(run_fds[-1])
    finally:
        for fd in reversed(run_fds):
            os.close(fd)
    return output_dir


def validate_run_provenance(provenance):
    required = {
        "schema",
        "source",
        "data",
        "recipe",
        "gpu_preflight",
        "scope",
        "reflect_padding_adapter",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("Invalid FI acc8 full-training provenance schema")
    if provenance["schema"] != "fi-varnet-acc8-full-training-provenance-v2":
        raise ValueError("Invalid FI acc8 full-training provenance version")
    if (
        not isinstance(provenance["source"], dict)
        or not isinstance(provenance["data"], dict)
        or not isinstance(provenance["gpu_preflight"], dict)
        or provenance["recipe"] != FI_ACC8_FULL_RECIPE.as_dict()
        or provenance["scope"]
        != {"training": True, "evaluation": False, "submission": False}
    ):
        raise ValueError("Invalid FI acc8 full-training provenance contract")
    validate_deterministic_reflect_pad_receipt(
        provenance["reflect_padding_adapter"]
    )
    return provenance


def _manifest_provenance(manifest):
    def selected(record):
        return {
            "name": record.name,
            "slices": record.slices,
            "kspace_shape": list(record.kspace_shape),
            "target_shape": list(record.target_shape),
            "mask_sha256": record.mask_sha256,
            "kspace_sha256": record.kspace_sha256,
            "image_sha256": record.image_sha256,
            "kspace_size": record.kspace_size,
            "kspace_st_dev": record.kspace_st_dev,
            "kspace_st_ino": record.kspace_st_ino,
            "image_size": record.image_size,
            "image_st_dev": record.image_st_dev,
            "image_st_ino": record.image_st_ino,
        }
    return {
        "root": str(manifest.root),
        "manifest_sha256": manifest.manifest_sha256,
        "organizer_total_entries": manifest.total_entries,
        "selected_acc8_count": manifest.selected_acc8_count,
        "ignored_acc4_count": manifest.ignored_acc4_count,
        "ignored_acc4_identity_sha256": manifest.ignored_acc4_identity_sha256,
        "selected_slice_count": manifest.slice_count,
        "ignored_acc4_access": {
            "method": "nofollow-stat-only", "payload_opened": False,
            "payload_hashed": False, "h5_read": False,
        },
        "selected_acc8_files": [selected(record) for record in manifest.records],
        "ignored_acc4_files": [asdict(record) for record in manifest.ignored_acc4_records],
        "directories": [asdict(item) for item in manifest.directory_identities],
    }


def _runtime(gpu_uuid, device, started):
    return {
        "pid": os.getpid(),
        "gpu_uuid": gpu_uuid,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _cuda_status_runtime(device):
    """Read counters only from the already-selected CUDA runtime/device."""
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _status_payload(
    cursor,
    *,
    phase,
    gpu,
    device_runtime,
    file_name,
    slice_index,
    started,
    starting_global_step,
    nominal_lr,
    moving_loss,
    checkpoint_path,
    checkpoint_sha256,
    command_argv,
    reflect_padding_adapter,
    now_monotonic=None,
    updated_unix_seconds=None,
):
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    updated_unix_seconds = (
        float(time.time())
        if updated_unix_seconds is None
        else updated_unix_seconds
    )
    elapsed = max(float(now_monotonic) - float(started), 0.0)
    session_steps = max(cursor.global_step - starting_global_step, 0)
    throughput = session_steps / elapsed if elapsed > 0.0 else 0.0
    remaining = max(FI_ACC8_FULL_RECIPE.base_max_steps - cursor.global_step, 0)
    eta = remaining / throughput if throughput > 0.0 else 0.0
    checkpoint_path = str(checkpoint_path) if checkpoint_path else ""
    status = {
        "schema": "fi-varnet-acc8-full-training-status-v2",
        "authoritative": False,
        "phase": phase,
        "pid": os.getpid(),
        "gpu_uuid": gpu["uuid"],
        "gpu_name": gpu["name"],
        "gpu_index": gpu["index"],
        "vram_allocated_bytes": device_runtime["allocated_bytes"],
        "vram_reserved_bytes": device_runtime["reserved_bytes"],
        "vram_peak_bytes": device_runtime["peak_bytes"],
        "epoch": cursor.epoch,
        "file_cursor": cursor.file_cursor,
        "file": file_name,
        "slice": slice_index,
        "global_step": cursor.global_step,
        "nominal_lr": float(nominal_lr),
        "moving_loss": float(moving_loss),
        "throughput_steps_per_second": float(throughput),
        "eta_seconds": float(eta),
        "last_checkpoint_path": checkpoint_path,
        "last_checkpoint_sha256": checkpoint_sha256,
        "command_argv": list(command_argv),
        "deterministic_contract": dict(FI_ACC8_DETERMINISM_CONTRACT),
        "reflect_padding_adapter": dict(
            validate_deterministic_reflect_pad_receipt(reflect_padding_adapter)
        ),
        "updated_unix_seconds": float(updated_unix_seconds),
    }
    return validate_status(status)


def validate_full_training_summary(summary):
    required = {
        "schema",
        "namespace",
        "scope",
        "training_complete",
        "evaluation_authorized",
        "submission_authorized",
        "completed_epoch",
        "global_step",
        "optimizer_steps",
        "scheduler_steps",
        "file_transactions",
        "loss_sum",
        "loss_count",
        "mean_loss",
        "last_checkpoint",
        "last_checkpoint_sha256",
        "bindings",
        "reflect_padding_adapter",
        "pid",
        "gpu_uuid",
        "peak_vram_bytes",
        "elapsed_seconds",
    }
    if not isinstance(summary, dict) or set(summary) != required:
        raise ValueError("Invalid FI acc8 full-training summary schema")
    if (
        summary["schema"] != "fi-varnet-acc8-full-training-summary-v2"
        or summary["namespace"] != FI_ACC8_FULL_NAMESPACE
        or summary["scope"] != "FULL_TRAINING_ONLY"
        or summary["training_complete"] is not True
        or summary["evaluation_authorized"] is not False
        or summary["submission_authorized"] is not False
        or summary["completed_epoch"] != FI_ACC8_FULL_RECIPE.base_epochs
        or summary["global_step"] != FI_ACC8_FULL_RECIPE.base_max_steps
        or summary["optimizer_steps"] != FI_ACC8_FULL_RECIPE.base_max_steps
        or summary["scheduler_steps"] != FI_ACC8_FULL_RECIPE.base_max_steps
        or summary["file_transactions"]
        != FI_ACC8_FULL_RECIPE.base_epochs * FI_ACC8_FULL_RECIPE.train_files
        or summary["loss_count"] != FI_ACC8_FULL_RECIPE.base_max_steps
    ):
        raise ValueError("Invalid FI acc8 full-training summary contract")
    for key in ("loss_sum", "mean_loss", "elapsed_seconds"):
        if type(summary[key]) is not float or not math.isfinite(summary[key]):
            raise ValueError(f"Invalid FI acc8 full-training summary {key}")
    if summary["loss_sum"] < 0 or summary["mean_loss"] < 0 or summary["elapsed_seconds"] < 0:
        raise ValueError("Invalid FI acc8 full-training summary finite values")
    for key in ("pid", "peak_vram_bytes"):
        minimum = 1 if key == "pid" else 0
        if type(summary[key]) is not int or summary[key] < minimum:
            raise ValueError(f"Invalid FI acc8 full-training summary {key}")
    if (
        not isinstance(summary["gpu_uuid"], str)
        or not summary["gpu_uuid"]
        or summary["gpu_uuid"] != summary["bindings"].get("gpu_uuid")
        or not isinstance(summary["last_checkpoint"], str)
        or not Path(summary["last_checkpoint"]).is_absolute()
    ):
        raise ValueError("Invalid FI acc8 full-training summary runtime binding")
    _validate_sha(summary["last_checkpoint_sha256"], "summary checkpoint")
    _validate_bindings(summary["bindings"])
    receipt = validate_deterministic_reflect_pad_receipt(
        summary["reflect_padding_adapter"]
    )
    if summary["bindings"]["reflect_padding_adapter"] != receipt:
        raise ValueError("FI acc8 full-training summary adapter binding disagrees")
    return summary


def _complete_full_run(output_dir, summary):
    validate_full_training_summary(summary)
    _atomic_replace_bytes(output_dir / "training-summary.json", _canonical_json(summary) + b"\n")
    _atomic_replace_bytes(
        output_dir / "COMPLETE",
        b"FI-VARNET-ACC8-FULL-TRAINING-COMPLETE-E30-STEP69450\n",
    )
    run_fds, _ = _open_directory_chain_nofollow(output_dir)
    try:
        os.unlink("INCOMPLETE", dir_fd=run_fds[-1])
        os.fsync(run_fds[-1])
    finally:
        for fd in reversed(run_fds):
            os.close(fd)


def run_fi_acc8_full_training(args, output_dir):
    """Train only scratch epochs 1-30; never evaluate or authorize submission."""
    if not getattr(args, "fi_acc8_full_training", False):
        raise ValueError("FI acc8 full training requires its explicit gate")
    if Path(args.data_path_train) != FI_ACC8_PRODUCTION_ROOT:
        raise ValueError(f"Production data root must be {FI_ACC8_PRODUCTION_ROOT}")
    resume_path = getattr(args, "resume_checkpoint", None)
    resume_sha = getattr(args, "resume_checkpoint_sha256", None)
    if (resume_path is None) != (resume_sha is None):
        raise ValueError("Exact resume requires checkpoint path and SHA-256")

    resume = resume_path is not None
    output_dir = _check_full_run_root_state(output_dir, resume=resume)
    if resume:
        resume_path = _check_resume_checkpoint_reference(
            output_dir, resume_path, resume_sha
        )
    source = verify_pinned_upstream_sources()
    manifest = inspect_acc8_training_data(
        args.data_path_train, input_key=args.input_key, target_key=args.target_key,
        max_key=args.max_key, expected_root=FI_ACC8_PRODUCTION_ROOT,
    )
    gpu = preflight_smoke_gpu(args.GPU_NUM, args.expected_gpu_uuid)
    resource_preflight = preflight_full_training_resources(manifest, output_dir)
    determinism = configure_determinism_pre_cuda()

    # Build and validate the exact CPU model adapters before binding provenance or
    # selecting/transferring to CUDA.  A malformed/missing receipt therefore fails
    # closed without creating GPU state.
    random.seed(FI_ACC8_FULL_RECIPE.seed)
    np.random.seed(FI_ACC8_FULL_RECIPE.seed)
    torch.manual_seed(FI_ACC8_FULL_RECIPE.seed)
    model, reflect_padding, activation = _build_full_training_model_with_adapters(args)
    reflect_padding = dict(
        validate_deterministic_reflect_pad_receipt(reflect_padding)
    )

    bindings = {
        "source_sha256": source_binding_sha256(source, reflect_padding),
        "data_manifest_sha256": manifest.manifest_sha256,
        "recipe_sha256": recipe_sha256(reflect_padding),
        "gpu_uuid": gpu["uuid"],
        "reflect_padding_adapter": reflect_padding,
    }
    provenance = {
        "schema": "fi-varnet-acc8-full-training-provenance-v2",
        "source": source, "data": _manifest_provenance(manifest),
        "recipe": FI_ACC8_FULL_RECIPE.as_dict(), "gpu_preflight": gpu,
        "scope": {"training": True, "evaluation": False, "submission": False},
        "reflect_padding_adapter": reflect_padding,
    }
    validate_run_provenance(provenance)
    output_dir = _prepare_full_run_root(output_dir, resume=resume)
    provenance_path = output_dir / "provenance.json"
    checkpoint_provenance = {
        "source": source,
        "data": provenance["data"],
        "recipe": FI_ACC8_FULL_RECIPE.as_dict(),
        "gpu": {"uuid": gpu["uuid"]},
        "reflect_padding_adapter": reflect_padding,
    }
    if provenance_path.exists():
        if json.loads(provenance_path.read_bytes()) != provenance:
            raise ValueError("Resume provenance does not match current exact gates")
    else:
        _write_file_fsync(provenance_path, _canonical_json(provenance) + b"\n")

    device = _select_smoke_device(args.GPU_NUM)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FI_ACC8_FULL_RECIPE.lr,
        weight_decay=FI_ACC8_FULL_RECIPE.weight_decay,
    )
    scheduler = build_fi_scheduler(optimizer)
    records = tuple(manifest.records)
    if resume_path is None:
        torch.cuda.manual_seed_all(FI_ACC8_FULL_RECIPE.seed)
        model.to(device=device, dtype=torch.float32)
        cursor, transactions = FullSamplerCursor(1, 0, 0), []
        metrics = {"loss_sum": 0.0, "loss_count": 0}
    else:
        state = load_full_checkpoint(
            resume_path, expected_sha256=resume_sha, expected_bindings=bindings,
            records=records, model=model, optimizer=optimizer, scheduler=scheduler,
            device=device,
        )
        cursor = FullSamplerCursor(
            state["sampler"]["epoch"], state["sampler"]["file_cursor"],
            state["sampler"]["global_step"],
        )
        transactions, metrics = list(state["transactions"]), dict(state["metrics"])

    loss_fn = build_pinned_ssim_loss()
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device=device)
    started = time.monotonic()
    last_status = started
    starting_global_step = cursor.global_step
    last_checkpoint = str(resume_path) if resume_path is not None else ""
    last_checkpoint_sha256 = resume_sha if resume_path is not None else ""
    moving_loss = (
        float(metrics["loss_sum"] / metrics["loss_count"])
        if metrics["loss_count"]
        else 0.0
    )
    command_argv = list(sys.argv)

    def publish_status(phase, record_name, slice_index, *, now_monotonic=None):
        nonlocal last_status
        now = time.monotonic() if now_monotonic is None else now_monotonic
        atomic_write_status(
            output_dir / "status.json",
            _status_payload(
                cursor,
                phase=phase,
                gpu=gpu,
                device_runtime=_cuda_status_runtime(device),
                file_name=record_name,
                slice_index=slice_index,
                started=started,
                now_monotonic=now,
                starting_global_step=starting_global_step,
                nominal_lr=float(optimizer.param_groups[0]["lr"]),
                moving_loss=moving_loss,
                checkpoint_path=last_checkpoint,
                checkpoint_sha256=last_checkpoint_sha256,
                command_argv=command_argv,
                reflect_padding_adapter=reflect_padding,
            ),
        )
        last_status = now

    mapping = _record_map(records)
    last_record_name = ""
    last_slice = 0
    while cursor.epoch <= FI_ACC8_FULL_RECIPE.base_epochs:
        epoch = cursor.epoch
        file_order = deterministic_file_order(records, epoch)
        while cursor.file_cursor < len(file_order):
            record = mapping[file_order[cursor.file_cursor]]
            last_record_name = record.name
            order = deterministic_slice_order(record, epoch)
            file_started, last_slice = time.monotonic(), 0
            with open_verified_acc8_file(manifest, record, order) as transaction:
                for sample in transaction:
                    result = run_full_finite_optimizer_step(
                        model, sample, loss_fn, optimizer, scheduler, device,
                        global_step=cursor.global_step,
                    )
                    cursor.global_step = result["global_step"]
                    metrics["loss_sum"] += result["loss"]
                    metrics["loss_count"] += 1
                    moving_loss = (
                        float(result["loss"])
                        if metrics["loss_count"] == 1
                        else 0.98 * moving_loss + 0.02 * float(result["loss"])
                    )
                    if not math.isfinite(moving_loss):
                        raise FloatingPointError("nonfinite FI acc8 moving loss")
                    last_slice = result["slice"]
                    now = time.monotonic()
                    if now - last_status >= FI_ACC8_FULL_RECIPE.status_interval_seconds:
                        publish_status(
                            "training", record.name, last_slice, now_monotonic=now
                        )
            receipt = dict(transaction.receipt)
            receipt.update(epoch=epoch, elapsed_seconds=float(time.monotonic() - file_started))
            transactions.append(receipt)
            cursor.file_cursor += 1
            epoch_end = cursor.file_cursor == len(file_order)
            if epoch_end:
                cursor.epoch, cursor.file_cursor = cursor.epoch + 1, 0
            validate_cursor(cursor, records, FI_ACC8_FULL_RECIPE.base_epochs)
            state = build_full_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, cursor=cursor,
                records=records, bindings=bindings, transactions=transactions, metrics=metrics,
                provenance=checkpoint_provenance,
                runtime=_runtime(gpu["uuid"], device, started),
                activation_checkpointing=activation,
                reflect_padding_adapter=reflect_padding,
            )
            publication = publish_full_checkpoint(output_dir, state, epoch_end=epoch_end)
            last_checkpoint = str(publication["checkpoint_path"])
            last_checkpoint_sha256 = publication["sha256"]
            publish_status("checkpointed", record.name, last_slice)
            if epoch_end:
                break

    if (cursor.epoch, cursor.file_cursor, cursor.global_step) != (31, 0, 69450):
        raise RuntimeError("Base run did not stop exactly at epoch 30/global step 69450")
    if metrics["loss_count"] != 69450 or len(transactions) != 2550:
        raise RuntimeError("Base run did not cover each slice/file exactly once")
    runtime = _runtime(gpu["uuid"], device, started)
    summary = {
        "schema": "fi-varnet-acc8-full-training-summary-v2",
        "namespace": FI_ACC8_FULL_NAMESPACE, "scope": "FULL_TRAINING_ONLY",
        "training_complete": True, "evaluation_authorized": False,
        "submission_authorized": False, "completed_epoch": 30,
        "global_step": 69450, "optimizer_steps": 69450,
        "scheduler_steps": 69450, "file_transactions": 2550,
        "loss_sum": metrics["loss_sum"], "loss_count": metrics["loss_count"],
        "mean_loss": metrics["loss_sum"] / metrics["loss_count"],
        "last_checkpoint": last_checkpoint,
        "last_checkpoint_sha256": last_checkpoint_sha256,
        "bindings": bindings,
        "reflect_padding_adapter": reflect_padding,
        **runtime,
    }
    publish_status("complete", last_record_name, last_slice)
    _complete_full_run(output_dir, summary)
    return {
        "output_dir": output_dir, "summary": summary,
        "checkpoint_path": Path(last_checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(last_checkpoint).read_bytes()).hexdigest(),
    }
