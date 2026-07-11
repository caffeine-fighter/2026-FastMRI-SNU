#!/usr/bin/env python3
"""Convert explicitly trusted legacy checkpoints to the safe resume format."""

import argparse
import fcntl
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.learning.resume import (
    _open_anonymous_file,
    _open_checkpoint_directory,
    _open_regular_at,
    _publish_fd_without_overwrite,
    sanitize_legacy_training_state,
    validate_checkpoint_pair,
    validate_training_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trusted legacy model.pt")
    parser.add_argument("output", type=Path, help="Safe resume checkpoint to create")
    parser.add_argument(
        "--trusted-input",
        action="store_true",
        help="Confirm that both legacy checkpoints are trusted",
    )
    return parser.parse_args()


def _temporary_path(directory):
    directory_fd = _open_checkpoint_directory(directory)
    try:
        return _open_anonymous_file(directory_fd)
    finally:
        os.close(directory_fd)


def _convert_trusted(source, temporary_output):
    legacy = torch.load(source, map_location="cpu", weights_only=False)
    temporary_output.seek(0)
    temporary_output.truncate()
    torch.save(sanitize_legacy_training_state(legacy), temporary_output)
    temporary_output.flush()
    os.fsync(temporary_output.fileno())
    temporary_output.seek(0)
    safe_state = torch.load(temporary_output, map_location="cpu", weights_only=True)
    validate_training_checkpoint(safe_state)
    return safe_state


def _publish_without_overwrite(source, destination):
    """Atomically publish an open inode without clobbering or path reopening."""
    destination = Path(destination)
    directory_fd = _open_checkpoint_directory(destination.parent)
    source_handle = None
    try:
        if hasattr(source, "fileno"):
            source_handle = source
        else:
            source_path = Path(source)
            source_directory_fd = _open_checkpoint_directory(source_path.parent)
            try:
                source_handle = _open_regular_at(
                    source_directory_fd, source_path.name, "converted checkpoint"
                )
            finally:
                os.close(source_directory_fd)
        _publish_fd_without_overwrite(
            source_handle, directory_fd, destination.name
        )
    finally:
        if source_handle is not source and source_handle is not None:
            source_handle.close()
        os.close(directory_fd)


def _publish_pair_without_overwrite(
    temporary_model,
    output_model,
    temporary_best,
    output_best,
):
    output_directory = output_model.parent
    if output_best.parent != output_directory:
        raise ValueError("Checkpoint outputs must share a directory")
    directory_fd = os.open(output_directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        _publish_without_overwrite(temporary_model, output_model)
        _publish_without_overwrite(temporary_best, output_best)
        os.fsync(directory_fd)
    finally:
        # POSIX has no atomic compare-and-unlink.  A rollback unlink could delete
        # a non-cooperating writer's replacement, so published names are never
        # removed here. A caller treats any partial publication as a failed,
        # non-overwriting conversion and reports it for explicit recovery.
        os.close(directory_fd)


def main():
    args = parse_args()
    if not args.trusted_input:
        raise SystemExit(
            "Refusing unsafe legacy deserialization without --trusted-input"
        )

    if not args.output.name.endswith(".safe.pt"):
        raise SystemExit("Output filename must end with .safe.pt")

    source_best = args.input.parent / "best_model.pt"
    output_best = args.output.parent / "best_model.safe.pt"
    if args.output.resolve() == output_best.resolve():
        raise SystemExit("Model and best-model outputs must be distinct paths")
    if not source_best.is_file():
        raise SystemExit(f"Legacy best checkpoint not found: {source_best}")
    for output in (args.output, output_best):
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing output: {output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_handle = _temporary_path(args.output.parent)
    temporary_best_handle = _temporary_path(args.output.parent)
    try:
        safe_model = _convert_trusted(args.input, temporary_model_handle)
        safe_best = _convert_trusted(source_best, temporary_best_handle)
        validate_checkpoint_pair(safe_model, safe_best)
        try:
            _publish_pair_without_overwrite(
                temporary_model_handle,
                args.output,
                temporary_best_handle,
                output_best,
            )
        except FileExistsError as exc:
            raise SystemExit(
                f"Refusing to overwrite output created concurrently: {exc.filename}"
            )
    finally:
        temporary_model_handle.close()
        temporary_best_handle.close()

    print(
        f"Created safe inexact-resume checkpoint: {args.output}\n"
        f"Created safe best checkpoint: {output_best}\n"
        "Use --allow-inexact-resume because legacy RNG state was unavailable."
    )


if __name__ == "__main__":
    main()
