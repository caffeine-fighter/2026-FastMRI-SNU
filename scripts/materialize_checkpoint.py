#!/usr/bin/env python3
"""Materialize one immutable checkpoint for the fixed inference loaders.

The candidate experiment directory must be nonexistent or an existing empty,
real directory whose parent already exists. The source and candidate paths may
not contain ``..`` or any symlink component. The source is safely loaded with
``weights_only=True`` and copied, never linked. Publication atomically adds a
complete, fsynced ``checkpoints`` directory without overwriting an existing
one. A failed/interrupted pre-publication attempt may preserve one hidden
``.checkpoints-unpublished-orphan-*`` tree; it never exposes ``checkpoints`` or
mutates a collision winner, and that candidate is no longer empty/reusable.
"""

import argparse
import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODEL_ROOT = REPO_ROOT / "utils" / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from utils.learning.resume import (  # noqa: E402
    _open_anonymous_file,
    _open_regular_at,
    _publish_fd_without_overwrite,
    validate_training_checkpoint,
)
from utils.learning.train_part import (  # noqa: E402
    _cleanup_staged_directory,
    _create_staged_directory,
    _publish_staged_directory_no_replace,
    _seal_staged_directory,
)

PROVENANCE_NAME = "materialization.json"
SHA256_LENGTH = 64


def _reject_traversal(path, description):
    path = Path(path)
    if not path.name or ".." in path.parts:
        raise ValueError(f"{description} path must name one location without traversal: {path}")
    return path


def _directory_flags():
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _open_parent_without_symlinks(path, description):
    """Open every parent component relative to descriptors without following links."""
    path = _reject_traversal(path, description)
    absolute = Path(os.path.abspath(path))
    directory_fd = os.open("/", _directory_flags())
    try:
        for component in absolute.parent.parts[1:]:
            next_fd = os.open(component, _directory_flags(), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except OSError as exc:
        os.close(directory_fd)
        raise ValueError(
            f"{description} parent must exist and contain no symlinks: {path}"
        ) from exc
    return absolute, directory_fd


def _sha256_handle(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _source_identity(file_stat):
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _validate_checkpoint(state):
    if not isinstance(state, Mapping):
        raise ValueError("Source is not a checkpoint mapping")
    model = state.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("Source checkpoint model must be a nonempty state-dict mapping")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in model.items()):
        raise ValueError("Source checkpoint model state-dict must map string names to tensors")
    if "optimizer" in state or "rng_state" in state:
        validate_training_checkpoint(state)
        return "training"
    return "inference"


def _open_validate_and_hash_source(source, expected_sha256):
    source, directory_fd = _open_parent_without_symlinks(source, "Source checkpoint")
    try:
        handle = _open_regular_at(directory_fd, source.name, "source checkpoint")
    finally:
        os.close(directory_fd)
    try:
        before = _source_identity(os.fstat(handle.fileno()))
        digest = _sha256_handle(handle)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"Source checkpoint SHA256 mismatch: expected {expected_sha256}, got {digest}"
            )
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=True)
        checkpoint_kind = _validate_checkpoint(state)
        after = _source_identity(os.fstat(handle.fileno()))
        if after != before:
            raise ValueError("Source checkpoint changed while it was being validated")
        handle.seek(0)
        return source, handle, digest, checkpoint_kind, before
    except BaseException:
        handle.close()
        raise


def _normalize_expected_sha256(value):
    if value is None:
        return None
    normalized = str(value).lower()
    if len(normalized) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("Expected SHA256 must contain exactly 64 hexadecimal characters")
    return normalized


def _prepare_candidate_directory(candidate_exp_dir):
    candidate, parent_fd = _open_parent_without_symlinks(
        candidate_exp_dir, "Candidate experiment"
    )
    try:
        try:
            candidate_stat = os.stat(
                candidate.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            os.mkdir(candidate.name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            candidate_stat = os.stat(
                candidate.name, dir_fd=parent_fd, follow_symlinks=False
            )
        if not stat.S_ISDIR(candidate_stat.st_mode):
            raise ValueError(
                f"Candidate experiment must be nonexistent or an empty real directory: {candidate}"
            )
        candidate_fd = os.open(candidate.name, _directory_flags(), dir_fd=parent_fd)
        try:
            opened_stat = os.fstat(candidate_fd)
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                candidate_stat.st_dev,
                candidate_stat.st_ino,
            ):
                raise ValueError("Candidate experiment changed identity while opening")
            with os.scandir(candidate_fd) as entries:
                if next(entries, None) is not None:
                    raise ValueError(
                        "Candidate experiment must be nonexistent or completely empty: "
                        f"{candidate}"
                    )
        finally:
            os.close(candidate_fd)
    except OSError as exc:
        raise ValueError(
            f"Candidate experiment must be a real directory with a real parent: {candidate}"
        ) from exc
    finally:
        os.close(parent_fd)
    return candidate


def _copy_source_to_descriptor(source_handle, output_handle, original_digest, identity):
    source_handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
        output_handle.write(chunk)
        digest.update(chunk)
    if digest.hexdigest() != original_digest or _source_identity(
        os.fstat(source_handle.fileno())
    ) != identity:
        raise ValueError("Source checkpoint changed while it was being copied")
    output_handle.flush()
    os.fsync(output_handle.fileno())


def _open_publication_temporary(directory_fd):
    try:
        return _open_anonymous_file(directory_fd)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        # The staging directory is private and atomically published as a tree.
        # Bind the securely-created named fallback to the already-open directory
        # descriptor, and keep it open until renameat2 consumes its random name.
        return tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".materialize-checkpoint-",
            dir=f"/proc/self/fd/{directory_fd}",
            delete=False,
        )


def _publish_temporary_without_overwrite(temporary, directory_fd, destination_name):
    temporary_name = temporary.name
    if not isinstance(temporary_name, str):
        _publish_fd_without_overwrite(temporary, directory_fd, destination_name)
        return
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        directory_fd,
        os.fsencode(Path(temporary_name).name),
        directory_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    ) != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                f"Refusing to overwrite staged {destination_name}",
                destination_name,
            )
        raise OSError(error_number, os.strerror(error_number), destination_name)


def materialize_checkpoint(source, candidate_exp_dir, expected_sha256=None):
    """Publish an exact, independently copied best checkpoint and provenance."""
    expected_sha256 = _normalize_expected_sha256(expected_sha256)
    source, source_handle, digest, _checkpoint_kind, identity = (
        _open_validate_and_hash_source(source, expected_sha256)
    )
    try:
        candidate = _prepare_candidate_directory(candidate_exp_dir)
        final_checkpoints = candidate / "checkpoints"
        staged = _create_staged_directory(final_checkpoints, "Candidate checkpoints")
        try:
            with _open_publication_temporary(staged.directory_fd) as best_temporary:
                _copy_source_to_descriptor(
                    source_handle, best_temporary, digest, identity
                )
                best_temporary.seek(0)
                reloaded = torch.load(
                    best_temporary, map_location="cpu", weights_only=True
                )
                _validate_checkpoint(reloaded)
                _publish_temporary_without_overwrite(
                    best_temporary, staged.directory_fd, "best_model.pt"
                )

            provenance = {
                "artifact": {"path": "best_model.pt", "sha256": digest},
                "format_version": 1,
                "operation": "materialize_immutable_checkpoint",
                "source": {"path": str(source), "sha256": digest},
            }
            provenance_bytes = (
                json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8")
            with _open_publication_temporary(
                staged.directory_fd
            ) as provenance_temporary:
                provenance_temporary.write(provenance_bytes)
                provenance_temporary.flush()
                os.fsync(provenance_temporary.fileno())
                _publish_temporary_without_overwrite(
                    provenance_temporary, staged.directory_fd, PROVENANCE_NAME
                )

            _seal_staged_directory(staged)
            _publish_staged_directory_no_replace(
                staged, final_checkpoints, "Candidate checkpoints"
            )
        except BaseException:
            _cleanup_staged_directory(staged)
            raise
        return final_checkpoints / "best_model.pt"
    finally:
        source_handle.close()


def _build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", metavar="SOURCE", help="regular weights-only-compatible checkpoint"
    )
    parser.add_argument(
        "--candidate-exp-dir",
        required=True,
        metavar="DIRECTORY",
        help="new or empty candidate experiment directory (parent must exist)",
    )
    parser.add_argument(
        "--expected-sha256",
        metavar="HEX",
        help="optional expected SHA256 of the exact source checkpoint bytes",
    )
    return parser


def _main(argv=None):
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        best = materialize_checkpoint(
            args.source, args.candidate_exp_dir, args.expected_sha256
        )
    except Exception as exc:
        parser.error(f"materialization failed: {exc}")
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
