import ctypes
import copy
import errno
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import warnings
import uuid
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_MANIFEST_FORMAT_VERSION = 2
_LEGACY_CHECKPOINT_MANIFEST_FORMAT_VERSION = 1
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
_AT_EMPTY_PATH = 0x1000
_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400
_GENERATION_RE = re.compile(r"[0-9a-f]{32}")


def _open_checkpoint_directory(directory):
    return os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def _open_anonymous_file(directory_fd, mode="w+b"):
    """Open an unnamed same-filesystem inode suitable for descriptor publication."""
    if not hasattr(os, "O_TMPFILE"):
        raise OSError(errno.ENOTSUP, "Linux O_TMPFILE descriptor publication unavailable")
    try:
        fd = os.open(
            ".",
            os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"Linux O_TMPFILE descriptor publication unavailable: {exc.strerror}",
        ) from exc
    return os.fdopen(fd, mode)


def _publish_fd_without_overwrite(source, directory_fd, destination_name):
    """Hard-link an open inode directly, never resolving a temporary pathname."""
    if not isinstance(destination_name, str) or Path(destination_name).name != destination_name:
        raise ValueError("Descriptor publication destination must be a basename")
    source_fd = source if isinstance(source, int) else source.fileno()
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    encoded_name = os.fsencode(destination_name)
    if linkat(source_fd, b"", directory_fd, encoded_name, _AT_EMPTY_PATH) == 0:
        return
    first_errno = ctypes.get_errno()
    if first_errno not in (errno.ENOENT, errno.EPERM):
        raise OSError(first_errno, os.strerror(first_errno), destination_name)

    # Some containers remove CAP_DAC_READ_SEARCH, which makes AT_EMPTY_PATH
    # return ENOENT/EPERM. /proc/self/fd still names this exact descriptor and
    # AT_SYMLINK_FOLLOW asks linkat to link its inode, not a caller-owned path.
    proc_fd = os.fsencode(f"/proc/self/fd/{source_fd}")
    if linkat(_AT_FDCWD, proc_fd, directory_fd, encoded_name, _AT_SYMLINK_FOLLOW) == 0:
        return
    fallback_errno = ctypes.get_errno()
    raise OSError(
        fallback_errno,
        f"Linux descriptor publication unavailable: {os.strerror(fallback_errno)}",
        destination_name,
    )


def _replace_from_open_descriptor(source, directory, directory_fd, destination_name, prefix):
    """Replace a compatibility name via one bounded, random staging link."""
    staging_name = f"{prefix}{uuid.uuid4().hex}"
    _publish_fd_without_overwrite(source, directory_fd, staging_name)
    # POSIX cannot rename an unnamed inode over an existing name directly.
    # A successful replace consumes this sole staging name. If replace fails or
    # is interrupted, leave at most this one link: cleanup cannot atomically
    # prove that a non-cooperating directory writer did not substitute it.
    os.replace(Path(directory) / staging_name, Path(directory) / destination_name)


def _open_regular_at(directory_fd, name, description):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(f"Invalid {description} name: {name!r}")
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"Cannot securely open {description}: {name}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{description} is not a regular file: {name}")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _validate_checkpoint_manifest_payload(manifest, manifest_path):
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid checkpoint publication manifest: {manifest_path}")
    format_version = manifest.get("format_version")
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise ValueError(f"Unsupported checkpoint publication manifest: {manifest_path}")
    legacy_required = {"format_version", "generation", "model", "best"}
    current_required = legacy_required | {"artifacts"}
    if format_version == _LEGACY_CHECKPOINT_MANIFEST_FORMAT_VERSION:
        valid_keys = (legacy_required, legacy_required | {"history"})
    elif format_version == CHECKPOINT_MANIFEST_FORMAT_VERSION:
        valid_keys = (
            current_required,
            current_required | {"history"},
            current_required | {"retained_epochs"},
            current_required | {"history", "retained_epochs"},
        )
    else:
        raise ValueError(f"Unsupported checkpoint publication manifest: {manifest_path}")
    if set(manifest) not in valid_keys:
        raise ValueError(f"Invalid checkpoint publication manifest: {manifest_path}")
    generation = manifest["generation"]
    if not isinstance(generation, str) or _GENERATION_RE.fullmatch(generation) is None:
        raise ValueError(f"Invalid checkpoint generation in manifest: {manifest_path}")
    expected_model = f".checkpoint-generation-{generation}-model.pt"
    if manifest["model"] != expected_model:
        raise ValueError(f"Invalid model artifact in checkpoint manifest: {manifest_path}")
    best_name = manifest["best"]
    if best_name is not None:
        best_match = re.fullmatch(
            r"\.checkpoint-generation-([0-9a-f]{32})-(model|best)\.pt",
            best_name if isinstance(best_name, str) else "",
        )
        if best_match is None:
            raise ValueError(f"Invalid best artifact in checkpoint manifest: {manifest_path}")
    if "history" in manifest:
        expected_history = f".checkpoint-generation-{generation}-history.npy"
        if manifest["history"] != expected_history:
            raise ValueError(
                f"Invalid history artifact in checkpoint manifest: {manifest_path}"
            )
    if format_version == _LEGACY_CHECKPOINT_MANIFEST_FORMAT_VERSION:
        return manifest
    artifact_names = {manifest["model"], manifest["best"]} - {None}
    if "history" in manifest:
        artifact_names.add(manifest["history"])
    artifact_hashes = manifest["artifacts"]
    if (
        not isinstance(artifact_hashes, dict)
        or set(artifact_hashes) != artifact_names
        or any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in artifact_hashes.values()
        )
    ):
        raise ValueError(
            f"Invalid artifact digests in checkpoint manifest: {manifest_path}"
        )
    if "retained_epochs" in manifest:
        records = manifest["retained_epochs"]
        if not isinstance(records, list) or not records:
            raise ValueError(f"Invalid retained epoch ledger in manifest: {manifest_path}")
        previous_epoch = 0
        seen_generations = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "epoch", "generation", "digest"
            }:
                raise ValueError(
                    f"Invalid retained epoch record in manifest: {manifest_path}"
                )
            epoch = record["epoch"]
            record_generation = record["generation"]
            digest = record["digest"]
            if (
                not isinstance(epoch, int)
                or isinstance(epoch, bool)
                or epoch <= previous_epoch
                or not isinstance(record_generation, str)
                or _GENERATION_RE.fullmatch(record_generation) is None
                or record_generation in seen_generations
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(
                    f"Invalid retained epoch provenance in manifest: {manifest_path}"
                )
            previous_epoch = epoch
            seen_generations.add(record_generation)
        if records[-1]["generation"] != generation:
            raise ValueError(
                f"Retained epoch ledger does not end at manifest generation: {manifest_path}"
            )
    return manifest


def _validate_retention_policy_transition(previous, retained_epoch_record):
    """Keep an existing retained ledger contiguous across new generations."""
    if (
        previous is not None
        and previous.get("retained_epochs")
        and retained_epoch_record is None
    ):
        raise ValueError(
            "cannot disable retained epoch publication after a retained ledger exists"
        )


def _read_checkpoint_manifest(directory, directory_fd=None):
    manifest_path = Path(directory) / CHECKPOINT_MANIFEST_NAME
    owns_directory_fd = directory_fd is None
    if owns_directory_fd:
        directory_fd = _open_checkpoint_directory(directory)
    try:
        try:
            manifest_fd = os.open(
                CHECKPOINT_MANIFEST_NAME,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError(f"Cannot securely open checkpoint manifest: {manifest_path}") from exc
        with os.fdopen(manifest_fd, "r", encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError(f"Checkpoint manifest is not a regular file: {manifest_path}")
            manifest = json.load(handle)
    finally:
        if owns_directory_fd:
            os.close(directory_fd)
    return _validate_checkpoint_manifest_payload(manifest, manifest_path)


def _open_manifest_artifact(directory_fd, manifest, field):
    name = manifest[field]
    if name is None:
        raise FileNotFoundError("Authoritative checkpoint manifest contains no best artifact")
    return _open_regular_at(directory_fd, name, f"manifest {field} artifact")


def _manifest_artifact_sha256(manifest, field):
    artifact_hashes = manifest.get("artifacts")
    if artifact_hashes is None:
        return None
    return artifact_hashes[manifest[field]]


def _verify_manifest_artifact_handle(handle, manifest, field, description):
    expected_sha256 = _manifest_artifact_sha256(manifest, field)
    if expected_sha256 is not None:
        _verify_open_handle_sha256(handle, expected_sha256, description)


def _verify_open_handle_sha256(handle, expected_sha256, description):
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{description} SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    handle.seek(0)


def _load_checkpoint_from_handle(handle, expected_sha256=None):
    if expected_sha256 is not None:
        _verify_open_handle_sha256(handle, expected_sha256, "Checkpoint")
    return torch.load(handle, map_location="cpu", weights_only=True)


def _load_requested_checkpoint(checkpoint_path, expected_sha256=None):
    checkpoint_path = Path(checkpoint_path)
    field = {"model.pt": "model", "best_model.pt": "best"}.get(checkpoint_path.name)
    directory_fd = _open_checkpoint_directory(checkpoint_path.parent)
    try:
        if field is None:
            with _open_regular_at(directory_fd, checkpoint_path.name, "checkpoint") as handle:
                return _load_checkpoint_from_handle(handle, expected_sha256)
        manifest = _read_checkpoint_manifest(checkpoint_path.parent, directory_fd)
        if manifest is None:
            with _open_regular_at(directory_fd, checkpoint_path.name, "checkpoint") as handle:
                return _load_checkpoint_from_handle(handle, expected_sha256)
        artifact_name = manifest[field]
        if artifact_name is None:
            raise FileNotFoundError(
                "Authoritative checkpoint manifest contains no best artifact"
            )
        manifest_sha256 = _manifest_artifact_sha256(manifest, field)
        if (
            manifest_sha256 is not None
            and expected_sha256 is not None
            and expected_sha256 != manifest_sha256
        ):
            raise ValueError(
                "Checkpoint SHA-256 mismatch: "
                f"requested={expected_sha256} authoritative={manifest_sha256}"
            )
        with _open_manifest_artifact(directory_fd, manifest, field) as handle:
            return _load_checkpoint_from_handle(
                handle, manifest_sha256 or expected_sha256
            )
    finally:
        os.close(directory_fd)


def _replace_alias_from_artifact(directory, directory_fd, artifact_name, alias_name):
    with _open_regular_at(directory_fd, artifact_name, "manifest artifact") as source_handle:
        with _open_anonymous_file(directory_fd) as output_handle:
            shutil.copyfileobj(source_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
            _replace_from_open_descriptor(
                output_handle,
                directory,
                directory_fd,
                alias_name,
                ".checkpoint-alias-",
            )


def _replace_external_alias_from_artifact(
    source_directory_fd, artifact_name, destination
):
    destination = Path(destination)
    destination_directory_fd = _open_checkpoint_directory(destination.parent)
    try:
        with _open_regular_at(
            source_directory_fd, artifact_name, "manifest history artifact"
        ) as source_handle:
            with _open_anonymous_file(destination_directory_fd) as output_handle:
                shutil.copyfileobj(source_handle, output_handle)
                output_handle.flush()
                os.fsync(output_handle.fileno())
                _replace_from_open_descriptor(
                    output_handle,
                    destination.parent,
                    destination_directory_fd,
                    destination.name,
                    ".val-loss-alias-",
                )
        os.fsync(destination_directory_fd)
    finally:
        os.close(destination_directory_fd)


def recover_checkpoint_publication(directory):
    """Materialize stable aliases from the single authoritative manifest."""
    directory = Path(directory)
    directory_fd = _open_checkpoint_directory(directory)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        manifest = _read_checkpoint_manifest(directory, directory_fd)
        if manifest is None:
            return False
        with _open_manifest_artifact(directory_fd, manifest, "model") as handle:
            model_state = _load_checkpoint_from_handle(
                handle, _manifest_artifact_sha256(manifest, "model")
            )
        validate_training_checkpoint(model_state)
        if "history" in manifest:
            with _open_manifest_artifact(
                directory_fd, manifest, "history"
            ) as history_handle:
                _verify_manifest_artifact_handle(
                    history_handle, manifest, "history", "Checkpoint history"
                )
                history_state = np.load(history_handle, allow_pickle=False)
            _validate_val_loss_history(
                history_state,
                model_state["epoch"],
                directory / manifest["history"],
            )
        if manifest["best"] is not None:
            with _open_manifest_artifact(directory_fd, manifest, "best") as handle:
                best_state = _load_checkpoint_from_handle(
                    handle, _manifest_artifact_sha256(manifest, "best")
                )
            validate_training_checkpoint(best_state)
            validate_checkpoint_pair(model_state, best_state)
            _replace_alias_from_artifact(
                directory, directory_fd, manifest["best"], "best_model.pt"
            )
        if "history" in manifest and directory.name == "checkpoints":
            _replace_external_alias_from_artifact(
                directory_fd,
                manifest["history"],
                directory.parent / "val_loss_log.npy",
            )
        _replace_alias_from_artifact(
            directory, directory_fd, manifest["model"], "model.pt"
        )
        os.fsync(directory_fd)
        return True
    finally:
        os.close(directory_fd)


def _capture_rng_state():
    python_version, python_state, python_gauss = random.getstate()
    numpy_name, numpy_state, numpy_pos, numpy_gauss, numpy_cached = np.random.get_state()
    return {
        "python_version": python_version,
        "python_state": list(python_state),
        "python_gauss": python_gauss,
        "numpy_name": numpy_name,
        "numpy_state": torch.from_numpy(
            np.asarray(numpy_state, dtype=np.int64).copy()
        ),
        "numpy_pos": numpy_pos,
        "numpy_gauss": numpy_gauss,
        "numpy_cached": numpy_cached,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state, cuda_device_index=None):
    random.setstate(
        (
            int(state["python_version"]),
            tuple(int(value) for value in state["python_state"]),
            state["python_gauss"],
        )
    )
    np.random.set_state(
        (
            state["numpy_name"],
            state["numpy_state"].cpu().numpy().astype(np.uint32, copy=False),
            int(state["numpy_pos"]),
            int(state["numpy_gauss"]),
            float(state["numpy_cached"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_device_index is not None and len(state["torch_cuda"]) > cuda_device_index:
        torch.cuda.set_rng_state(
            state["torch_cuda"][cuda_device_index].cpu(),
            device=cuda_device_index,
        )


def _selected_cuda_device_index(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


def _validated_epoch(value, label="Checkpoint"):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} epoch must be a nonnegative integer")
    return int(value)


def _validate_rng_tensor(value, label):
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise ValueError(f"Checkpoint RNG {label} must be a nonempty uint8 tensor")


def _validate_rng_state(state):
    if state is None:
        return
    required = {
        "python_version",
        "python_state",
        "python_gauss",
        "numpy_name",
        "numpy_state",
        "numpy_pos",
        "numpy_gauss",
        "numpy_cached",
        "torch_cpu",
        "torch_cuda",
    }
    if not isinstance(state, Mapping) or not required.issubset(state):
        missing = required.difference(state if isinstance(state, Mapping) else {})
        raise ValueError(f"Invalid checkpoint RNG state; missing keys: {sorted(missing)}")
    if isinstance(state["python_version"], bool) or not isinstance(
        state["python_version"], Integral
    ):
        raise ValueError("Checkpoint RNG python_version must be an integer")
    if not isinstance(state["python_state"], list) or not state["python_state"]:
        raise ValueError("Checkpoint RNG python_state must be a nonempty integer list")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in state["python_state"]
    ):
        raise ValueError("Checkpoint RNG python_state must be a nonempty integer list")
    if state["python_gauss"] is not None and (
        not isinstance(state["python_gauss"], Real)
        or isinstance(state["python_gauss"], bool)
        or not math.isfinite(float(state["python_gauss"]))
    ):
        raise ValueError("Checkpoint RNG python_gauss must be finite or None")
    try:
        probe = random.Random()
        probe.setstate(
            (
                int(state["python_version"]),
                tuple(int(value) for value in state["python_state"]),
                state["python_gauss"],
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Checkpoint RNG Python state is not restorable") from exc
    if not isinstance(state["numpy_name"], str) or not state["numpy_name"]:
        raise ValueError("Checkpoint RNG numpy_name must be a nonempty string")
    numpy_state = state["numpy_state"]
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        not torch.is_tensor(numpy_state)
        or numpy_state.dtype not in integer_dtypes
        or numpy_state.ndim != 1
        or numpy_state.numel() == 0
    ):
        raise ValueError("Checkpoint RNG numpy_state must be an integer tensor")
    if isinstance(state["numpy_pos"], bool) or not isinstance(
        state["numpy_pos"], Integral
    ):
        raise ValueError("Checkpoint RNG numpy_pos must be an integer")
    if state["numpy_gauss"] not in (0, 1) or isinstance(
        state["numpy_gauss"], bool
    ):
        raise ValueError("Checkpoint RNG numpy_gauss must be 0 or 1")
    if not isinstance(state["numpy_cached"], Real) or isinstance(
        state["numpy_cached"], bool
    ) or not math.isfinite(float(state["numpy_cached"])):
        raise ValueError("Checkpoint RNG numpy_cached must be finite")
    try:
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(
            (
                state["numpy_name"],
                numpy_state.cpu().numpy().astype(np.uint32, copy=False),
                int(state["numpy_pos"]),
                int(state["numpy_gauss"]),
                float(state["numpy_cached"]),
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Checkpoint RNG NumPy state is not restorable") from exc
    _validate_rng_tensor(state["torch_cpu"], "torch_cpu")
    try:
        torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint RNG torch_cpu state is not restorable") from exc
    if not isinstance(state["torch_cuda"], list):
        raise ValueError("Checkpoint RNG torch_cuda must be a list")
    for cuda_state in state["torch_cuda"]:
        _validate_rng_tensor(cuda_state, "torch_cuda entry")


def build_training_state(
    epoch,
    model,
    optimizer,
    best_val_loss,
    scheduler=None,
    scaler=None,
    model_contract=None,
):
    """Build a safe, complete checkpoint for an exact training continuation."""
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": _validated_epoch(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": float(best_val_loss),
        "rng_state": _capture_rng_state(),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if model_contract is not None:
        state["model_contract"] = dict(model_contract)
    validate_training_checkpoint(state)
    return state


def sanitize_legacy_training_state(checkpoint):
    """Convert an already-loaded trusted legacy checkpoint to the safe format."""
    required = {"epoch", "model", "optimizer", "best_val_loss"}
    if not isinstance(checkpoint, Mapping) or not required.issubset(checkpoint):
        missing = required.difference(
            checkpoint if isinstance(checkpoint, Mapping) else {}
        )
        raise ValueError(f"Invalid legacy checkpoint; missing keys: {sorted(missing)}")
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": _validated_epoch(checkpoint["epoch"], "Legacy checkpoint"),
        "model": checkpoint["model"],
        "optimizer": checkpoint["optimizer"],
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "rng_state": None,
    }
    validate_training_checkpoint(state)
    return state


def _validate_model_state_dict(state):
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint model must be a nonempty state-dict mapping")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("Checkpoint model state-dict must map string names to tensors")


def _validate_optimizer_state_dict(state):
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint optimizer must be a state-dict mapping")
    if "state" not in state or "param_groups" not in state:
        raise ValueError("Checkpoint optimizer must contain state and param_groups")
    if not isinstance(state["state"], Mapping):
        raise ValueError("Checkpoint optimizer state must be a mapping")
    groups = state["param_groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("Checkpoint optimizer param_groups must be a nonempty list")
    parameter_ids = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("Checkpoint optimizer param_groups entries must contain params lists")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in group["params"]):
            raise ValueError("Checkpoint optimizer parameter identifiers must be integers")
        parameter_ids.extend(int(value) for value in group["params"])
    if not parameter_ids or len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("Checkpoint optimizer parameter identifiers must be nonempty and unique")
    if any(
        isinstance(key, bool) or not isinstance(key, Integral) or int(key) not in parameter_ids
        for key in state["state"]
    ):
        raise ValueError("Checkpoint optimizer state contains an unknown parameter identifier")


def _model_state_signature(state):
    return {
        key: (tuple(value.shape), value.dtype)
        for key, value in state.items()
    }


def _validate_model_compatibility(checkpoint_state, target_state):
    checkpoint_keys = set(checkpoint_state)
    target_keys = set(target_state)
    if checkpoint_keys != target_keys:
        raise ValueError("Checkpoint model keys do not match the target model")
    if _model_state_signature(checkpoint_state) != _model_state_signature(target_state):
        raise ValueError(
            "Checkpoint model tensor shapes or dtypes do not match the target model"
        )


def _optimizer_topology(state):
    return tuple(len(group["params"]) for group in state["param_groups"])


def _validate_optimizer_compatibility(checkpoint_state, target_state):
    if _optimizer_topology(checkpoint_state) != _optimizer_topology(target_state):
        raise ValueError(
            "Checkpoint optimizer parameter-group topology does not match the target optimizer"
        )


def validate_training_checkpoint(checkpoint):
    required = {
        "format_version",
        "epoch",
        "model",
        "optimizer",
        "best_val_loss",
        "rng_state",
    }
    if not isinstance(checkpoint, Mapping) or not required.issubset(checkpoint):
        missing = required.difference(
            checkpoint if isinstance(checkpoint, Mapping) else {}
        )
        raise ValueError(f"Invalid training checkpoint; missing keys: {sorted(missing)}")
    if (
        isinstance(checkpoint["format_version"], bool)
        or not isinstance(checkpoint["format_version"], Integral)
        or checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            f"Unsupported training checkpoint format {checkpoint['format_version']}"
        )
    _validated_epoch(checkpoint["epoch"])
    _validate_model_state_dict(checkpoint["model"])
    _validate_optimizer_state_dict(checkpoint["optimizer"])
    best_val_loss = checkpoint["best_val_loss"]
    if isinstance(best_val_loss, bool) or (
        torch.is_tensor(best_val_loss)
        and (best_val_loss.numel() != 1 or best_val_loss.dtype == torch.bool)
    ):
        raise ValueError("Checkpoint best validation loss must be a finite scalar")
    try:
        finite_best_loss = math.isfinite(float(best_val_loss))
    except (TypeError, ValueError):
        finite_best_loss = False
    if not finite_best_loss:
        raise ValueError("Checkpoint best validation loss must be finite")
    _validate_rng_state(checkpoint["rng_state"])
    for field in ("scheduler", "scaler", "model_contract"):
        if field in checkpoint and not isinstance(checkpoint[field], Mapping):
            raise ValueError(f"Checkpoint {field} state must be a mapping")
    if "model_contract" in checkpoint:
        for key, value in checkpoint["model_contract"].items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Checkpoint model contract must contain string pairs")


def validate_checkpoint_pair(model_checkpoint, best_checkpoint):
    """Reject model/best states that cannot describe one training lineage."""
    validate_training_checkpoint(model_checkpoint)
    validate_training_checkpoint(best_checkpoint)
    if best_checkpoint["epoch"] > model_checkpoint["epoch"]:
        raise ValueError("best checkpoint epoch must not exceed model checkpoint epoch")
    model_best = float(model_checkpoint["best_val_loss"])
    best_best = float(best_checkpoint["best_val_loss"])
    if model_best != best_best:
        raise ValueError(
            "Model and best checkpoints must carry the same best validation loss"
        )
    if _model_state_signature(model_checkpoint["model"]) != _model_state_signature(
        best_checkpoint["model"]
    ):
        raise ValueError(
            "Model and best checkpoints must have the same model architecture"
        )
    if _optimizer_topology(model_checkpoint["optimizer"]) != _optimizer_topology(
        best_checkpoint["optimizer"]
    ):
        raise ValueError(
            "Model and best checkpoints must have the same optimizer topology"
        )
    if model_checkpoint.get("model_contract") != best_checkpoint.get("model_contract"):
        raise ValueError("Model and best checkpoints must have the same model contract")


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _capture_live_rng_state(cuda_device_index=None):
    live = {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
    }
    if cuda_device_index is not None:
        live["torch_cuda"] = torch.cuda.get_rng_state(cuda_device_index).cpu().clone()
    return live


def _restore_live_rng_state(state, cuda_device_index=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if cuda_device_index is not None and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device=cuda_device_index)


def _prevalidate_selected_cuda_rng(rng_state, cuda_device_index):
    if cuda_device_index is None:
        return
    previous = torch.cuda.get_rng_state(cuda_device_index).cpu().clone()
    try:
        torch.cuda.set_rng_state(
            rng_state["torch_cuda"][cuda_device_index].cpu(),
            device=cuda_device_index,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint CUDA RNG state for cuda:{cuda_device_index} is not restorable"
        ) from exc
    finally:
        torch.cuda.set_rng_state(previous, device=cuda_device_index)


def load_training_state(
    checkpoint_path,
    model,
    optimizer,
    device,
    allow_inexact=False,
    learning_rate_override=None,
    expected_sha256=None,
    scheduler=None,
    scaler=None,
    expected_model_contract=None,
):
    """Transactionally restore a complete checkpoint or leave live state untouched."""
    checkpoint = _load_requested_checkpoint(checkpoint_path, expected_sha256)
    validate_training_checkpoint(checkpoint)
    checkpoint_contract = checkpoint.get("model_contract")
    if expected_model_contract is not None:
        expected_model_contract = dict(expected_model_contract)
        legacy_varnet = expected_model_contract == {"model_family": "varnet"}
        if not (checkpoint_contract is None and legacy_varnet):
            if checkpoint_contract != expected_model_contract:
                raise ValueError(
                    "Checkpoint model family or PromptMR+ recipe is incompatible "
                    "with this run"
                )
    if scheduler is not None and "scheduler" not in checkpoint:
        raise ValueError("Checkpoint is missing scheduler state required for exact resume")
    if scaler is not None and "scaler" not in checkpoint:
        raise ValueError("Checkpoint is missing AMP scaler state required for exact resume")
    if checkpoint["rng_state"] is None and not allow_inexact:
        raise ValueError(
            "Checkpoint has no RNG state; pass --allow-inexact-resume to accept "
            "a non-bit-exact continuation"
        )
    cuda_device_index = _selected_cuda_device_index(device)
    missing_selected_cuda_rng = (
        checkpoint["rng_state"] is not None
        and cuda_device_index is not None
        and len(checkpoint["rng_state"]["torch_cuda"]) <= cuda_device_index
    )
    if missing_selected_cuda_rng and not allow_inexact:
        raise ValueError(
            f"Checkpoint CUDA RNG topology cannot restore selected device "
            f"cuda:{cuda_device_index}; pass --allow-inexact-resume to accept "
            "a non-bit-exact continuation"
        )

    _validate_model_compatibility(checkpoint["model"], model.state_dict())
    _validate_optimizer_compatibility(
        checkpoint["optimizer"], optimizer.state_dict()
    )

    # Exercise restoration on an isolated object graph first. Copy the scheduler
    # together with its optimizer so their internal reference remains coherent.
    staged_model, staged_optimizer, staged_scheduler, staged_scaler = copy.deepcopy(
        (model, optimizer, scheduler, scaler)
    )
    try:
        staged_model.load_state_dict(checkpoint["model"])
        staged_optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state(staged_optimizer, device)
        if staged_scheduler is not None:
            staged_scheduler.load_state_dict(checkpoint["scheduler"])
        if staged_scaler is not None:
            staged_scaler.load_state_dict(checkpoint["scaler"])
        if learning_rate_override is not None:
            set_optimizer_learning_rate(staged_optimizer, learning_rate_override)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint training state is not restorable") from exc

    applicable_cuda_index = (
        cuda_device_index
        if checkpoint["rng_state"] is not None and not missing_selected_cuda_rng
        else None
    )
    if applicable_cuda_index is not None:
        _prevalidate_selected_cuda_rng(
            checkpoint["rng_state"], applicable_cuda_index
        )

    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
    scaler_before = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
    rng_before = _capture_live_rng_state(applicable_cuda_index)
    try:
        model.load_state_dict(staged_model.state_dict())
        optimizer.load_state_dict(staged_optimizer.state_dict())
        if scheduler is not None:
            scheduler.load_state_dict(staged_scheduler.state_dict())
        if scaler is not None:
            scaler.load_state_dict(staged_scaler.state_dict())
        if checkpoint["rng_state"] is not None:
            _restore_rng_state(checkpoint["rng_state"], applicable_cuda_index)
    except BaseException:
        model.load_state_dict(model_before)
        optimizer.load_state_dict(optimizer_before)
        if scheduler is not None:
            scheduler.load_state_dict(scheduler_before)
        if scaler is not None:
            scaler.load_state_dict(scaler_before)
        _restore_live_rng_state(rng_before, applicable_cuda_index)
        raise

    if checkpoint["rng_state"] is None:
        warnings.warn(
            "Resuming without RNG state; data order may differ from an uninterrupted run",
            RuntimeWarning,
            stacklevel=2,
        )
    elif missing_selected_cuda_rng:
        warnings.warn(
            f"Resuming without CUDA RNG state for selected device "
            f"cuda:{cuda_device_index}; continuation is not bit-exact",
            RuntimeWarning,
            stacklevel=2,
        )
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def set_optimizer_learning_rate(optimizer, learning_rate):
    """Override all optimizer parameter-group learning rates after resume."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Resume learning rate must be a positive finite value")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _validate_val_loss_history(history, start_epoch, history_path):
    enough_rows = (
        isinstance(history, np.ndarray)
        and history.ndim == 2
        and history.shape[1] == 2
        and history.shape[0] >= start_epoch
    )
    prefix = history[:start_epoch] if enough_rows else np.empty((0, 2))
    valid_epochs = enough_rows and np.array_equal(
        prefix[:, 0], np.arange(start_epoch)
    )
    if not valid_epochs or not np.isfinite(prefix).all():
        raise ValueError(
            f"Invalid validation history at {history_path}; expected at least "
            f"{start_epoch} rows with epochs 0 through {start_epoch - 1}"
        )
    return prefix.copy()


def load_val_loss_history(checkpoint_path, start_epoch):
    """Load history bound to the checkpoint manifest, with legacy fallback."""
    checkpoint_path = Path(checkpoint_path)
    manifest = None
    directory_fd = None
    if checkpoint_path.name in {"model.pt", "best_model.pt"}:
        directory_fd = _open_checkpoint_directory(checkpoint_path.parent)
        try:
            manifest = _read_checkpoint_manifest(
                checkpoint_path.parent, directory_fd
            )
            if manifest is not None and "history" in manifest:
                with _open_manifest_artifact(
                    directory_fd, manifest, "history"
                ) as history_handle:
                    _verify_manifest_artifact_handle(
                        history_handle, manifest, "history", "Checkpoint history"
                    )
                    history = np.load(history_handle, allow_pickle=False)
                return _validate_val_loss_history(
                    history,
                    start_epoch,
                    checkpoint_path.parent / manifest["history"],
                )
        finally:
            os.close(directory_fd)

    # Pre-manifest checkpoints had only this compatibility file.
    history_path = checkpoint_path.parent.parent / "val_loss_log.npy"
    history = np.load(history_path, allow_pickle=False)
    return _validate_val_loss_history(history, start_epoch, history_path)


def _copy_handle_without_overwrite(source_handle, destination):
    destination = Path(destination)
    directory_fd = _open_checkpoint_directory(destination.parent)
    try:
        with _open_anonymous_file(directory_fd) as output_handle:
            source_handle.seek(0)
            shutil.copyfileobj(source_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
            _publish_fd_without_overwrite(
                output_handle, directory_fd, destination.name
            )
            os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _copy_without_overwrite(source, destination):
    source = Path(source)
    source_directory_fd = _open_checkpoint_directory(source.parent)
    try:
        with _open_regular_at(
            source_directory_fd, source.name, "source checkpoint"
        ) as source_handle:
            _copy_handle_without_overwrite(source_handle, destination)
    finally:
        os.close(source_directory_fd)


def preserve_best_checkpoint(checkpoint_path, destination_dir):
    """Validate and seed a resumed experiment from its authoritative best."""
    checkpoint_path = Path(checkpoint_path)
    best_name = (
        "best_model.safe.pt"
        if checkpoint_path.name.endswith(".safe.pt")
        else "best_model.pt"
    )
    destination_dir = Path(destination_dir)
    destination = destination_dir / "best_model.pt"
    source_directory = checkpoint_path.parent
    source_directory_fd = _open_checkpoint_directory(source_directory)
    best_handle = None
    try:
        fcntl.flock(source_directory_fd, fcntl.LOCK_EX)
        manifest = _read_checkpoint_manifest(source_directory, source_directory_fd)
        if manifest is None:
            source = source_directory / best_name
            with _open_regular_at(
                source_directory_fd, checkpoint_path.name, "resume checkpoint"
            ) as model_handle:
                model_state = _load_checkpoint_from_handle(model_handle)
            best_handle = _open_regular_at(
                source_directory_fd, best_name, "source best checkpoint"
            )
        else:
            source = source_directory / "best_model.pt"
            with _open_manifest_artifact(
                source_directory_fd, manifest, "model"
            ) as model_handle:
                model_state = _load_checkpoint_from_handle(
                    model_handle, _manifest_artifact_sha256(manifest, "model")
                )
            if "history" in manifest:
                with _open_manifest_artifact(
                    source_directory_fd, manifest, "history"
                ) as history_handle:
                    _verify_manifest_artifact_handle(
                        history_handle, manifest, "history", "Checkpoint history"
                    )
                    history_state = np.load(history_handle, allow_pickle=False)
                _validate_val_loss_history(
                    history_state,
                    model_state["epoch"],
                    source_directory / manifest["history"],
                )
            best_handle = _open_manifest_artifact(
                source_directory_fd, manifest, "best"
            )

        best_state = _load_checkpoint_from_handle(
            best_handle,
            None if manifest is None else _manifest_artifact_sha256(manifest, "best"),
        )
        validate_checkpoint_pair(model_state, best_state)

        if manifest is not None:
            _replace_alias_from_artifact(
                source_directory,
                source_directory_fd,
                manifest["best"],
                "best_model.pt",
            )
            _replace_alias_from_artifact(
                source_directory,
                source_directory_fd,
                manifest["model"],
                "model.pt",
            )
            if "history" in manifest and source_directory.name == "checkpoints":
                _replace_external_alias_from_artifact(
                    source_directory_fd,
                    manifest["history"],
                    source_directory.parent / "val_loss_log.npy",
                )
            os.fsync(source_directory_fd)

        if os.path.abspath(source) == os.path.abspath(destination):
            return destination
        destination_dir.mkdir(parents=True, exist_ok=True)
        try:
            _copy_handle_without_overwrite(best_handle, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing best-checkpoint destination: {destination}"
            ) from exc
        return destination
    finally:
        if best_handle is not None:
            best_handle.close()
        os.close(source_directory_fd)
