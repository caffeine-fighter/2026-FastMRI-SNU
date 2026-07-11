#!/usr/bin/env python3
"""Safely average compatible same-basin training checkpoints for inference."""

import argparse
import copy
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from numbers import Real
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
    validate_training_checkpoint,
)


AVERAGING_MANIFEST_FORMAT_VERSION = 1


class PublicationIndeterminateError(OSError):
    """A required publication namespace update could not be confirmed durable."""


def _make_parent_chain_durable(directory):
    """Create missing directories and durably record each name in its parent."""
    directory = Path(os.path.abspath(directory))
    missing = []
    ancestor = directory
    while not ancestor.exists():
        missing.append(ancestor)
        parent = ancestor.parent
        if parent == ancestor:
            raise FileNotFoundError(errno.ENOENT, "No existing directory ancestor", str(directory))
        ancestor = parent

    ancestor_fd = _open_checkpoint_directory(ancestor)
    os.close(ancestor_fd)
    for child in reversed(missing):
        parent_fd = _open_checkpoint_directory(child.parent)
        try:
            try:
                os.mkdir(child.name, dir_fd=parent_fd)
            except FileExistsError:
                child_fd = _open_checkpoint_directory(child)
                os.close(child_fd)
            try:
                os.fsync(parent_fd)
            except OSError as exc:
                raise PublicationIndeterminateError(
                    exc.errno,
                    "Publication parent creation directory sync failed",
                    str(child.parent),
                ) from exc
        finally:
            os.close(parent_fd)


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Average two or more compatible training checkpoints into a sanitized "
            "inference-only checkpoint. Existing outputs are never overwritten."
        )
    )
    parser.add_argument(
        "sources",
        nargs="+",
        metavar="SOURCE",
        help="source training checkpoints (at least two)",
    )
    parser.add_argument(
        "--output", required=True, metavar="CHECKPOINT", help="output checkpoint path"
    )
    parser.add_argument(
        "--manifest", required=True, metavar="JSON", help="output manifest path"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        metavar="WEIGHT",
        help="optional nonnegative source weights (one per source)",
    )
    parser.add_argument(
        "--template",
        metavar="SOURCE",
        help="source checkpoint whose non-training metadata is retained",
    )
    return parser


def _load_and_hash(path):
    directory_fd = _open_checkpoint_directory(path.parent)
    try:
        handle = _open_regular_at(directory_fd, path.name, "source checkpoint")
    finally:
        os.close(directory_fd)
    with handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=True)
    return state, digest.hexdigest()


def _model_signature(state):
    return {key: (tuple(tensor.shape), tensor.dtype) for key, tensor in state.items()}


def _optimizer_topology(state):
    return tuple(len(group["params"]) for group in state["param_groups"])


def _sha256_regular_at(directory_fd, name, description):
    with _open_regular_at(directory_fd, name, description) as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_publication_temporary(directory_fd, directory):
    try:
        return _open_anonymous_file(directory_fd)
    except OSError as exc:
        unsupported = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}
        if exc.errno not in unsupported:
            raise
        # This capability-restricted fallback keeps a securely-created named
        # inode open. Publication still uses a no-overwrite hard link while the
        # directory lock coordinates cooperating writers.
        return tempfile.NamedTemporaryFile(
            mode="w+b", prefix=".average-checkpoint-", dir=directory, delete=False
        )


def _publish_temporary_without_overwrite(temporary, directory_fd, destination_name):
    temporary_name = temporary.name
    if not isinstance(temporary_name, str):
        _publish_fd_without_overwrite(temporary, directory_fd, destination_name)
        return
    source_name = Path(temporary_name).name
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            directory_fd,
            os.fsencode(source_name),
            directory_fd,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                f"Refusing to overwrite existing output {destination_name}",
                destination_name,
            )
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _publish_bytes_without_overwrite(data, directory_fd, directory, destination_name):
    """Durably stage bytes and attempt one no-overwrite namespace publication."""
    temporary = _open_publication_temporary(directory_fd, directory)
    temporary_name = temporary.name
    named_source = isinstance(temporary_name, str)
    publication_attempted = False
    try:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        publication_attempted = True
        _publish_temporary_without_overwrite(
            temporary, directory_fd, destination_name
        )
    finally:
        try:
            temporary.close()
        finally:
            if named_source and not publication_attempted:
                try:
                    os.unlink(Path(temporary_name).name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass


def _publish_output_and_manifest(output, output_path, manifest, manifest_path):
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    if output_path.parent.resolve() != manifest_path.parent.resolve():
        raise ValueError("Output checkpoint and manifest must share a directory")
    if output_path.name == manifest_path.name:
        raise ValueError("Output checkpoint and manifest paths must be distinct")
    _make_parent_chain_durable(output_path.parent)

    checkpoint_buffer = io.BytesIO()
    torch.save(output, checkpoint_buffer)
    checkpoint_buffer.seek(0)
    validated_output = torch.load(
        checkpoint_buffer, map_location="cpu", weights_only=True
    )
    if (
        validated_output.get("inference_only") is not True
        or "optimizer" in validated_output
        or "rng_state" in validated_output
    ):
        raise ValueError("Serialized output is not a sanitized inference checkpoint")
    checkpoint_bytes = checkpoint_buffer.getvalue()
    artifact_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    manifest = {
        **manifest,
        "artifact": {
            "path": output_path.name,
            "sha256": artifact_digest,
        },
        "committed": True,
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

    directory_fd = _open_checkpoint_directory(output_path.parent)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        try:
            existing_digest = _sha256_regular_at(
                directory_fd, output_path.name, "averaged generation artifact"
            )
        except FileNotFoundError:
            output_exists = False
        else:
            output_exists = True
            if existing_digest != artifact_digest:
                raise FileExistsError(
                    errno.EEXIST,
                    "Refusing to overwrite different generation artifact",
                    str(output_path),
                )
        try:
            existing_manifest_digest = _sha256_regular_at(
                directory_fd, manifest_path.name, "averaging manifest"
            )
        except FileNotFoundError:
            manifest_exists = False
        else:
            manifest_exists = True
            if existing_manifest_digest != manifest_digest:
                raise FileExistsError(
                    errno.EEXIST,
                    "Refusing to overwrite different authoritative manifest",
                    str(manifest_path),
                )

        if not output_exists:
            _publish_bytes_without_overwrite(
                checkpoint_bytes,
                directory_fd,
                output_path.parent,
                output_path.name,
            )
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                raise PublicationIndeterminateError(
                    exc.errno,
                    "Averaged generation was published but its directory sync failed",
                    str(output_path),
                ) from exc
        if not manifest_exists:
            _publish_bytes_without_overwrite(
                manifest_bytes,
                directory_fd,
                output_path.parent,
                manifest_path.name,
            )
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                raise PublicationIndeterminateError(
                    exc.errno,
                    "Averaging manifest was published but its directory sync failed",
                    str(manifest_path),
                ) from exc
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            raise PublicationIndeterminateError(
                exc.errno,
                "Publication matched, but its directory sync failed",
                str(output_path.parent),
            ) from exc
    finally:
        os.close(directory_fd)


def _normalized_weights(weights, count):
    if count < 2:
        raise ValueError("Averaging requires at least two source checkpoints")
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError("Number of weights must match source checkpoints")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not math.isfinite(float(weight))
        or float(weight) < 0.0
        for weight in weights
    ):
        raise ValueError("All weights must be finite nonnegative numbers")
    total = math.fsum(float(weight) for weight in weights)
    if total <= 0.0:
        raise ValueError("Weights must have a positive sum")
    return [float(weight) / total for weight in weights]


def average_checkpoints(source_paths, output_path, weights=None, template_path=None, manifest_path=None):
    """Average floating model tensors and write an inference checkpoint."""
    source_paths = [Path(path) for path in source_paths]
    normalized_weights = _normalized_weights(weights, len(source_paths))
    resolved_sources = [path.resolve() for path in source_paths]
    resolved_template = (
        resolved_sources[0] if template_path is None else Path(template_path).resolve()
    )
    try:
        template_index = resolved_sources.index(resolved_template)
    except ValueError as exc:
        raise ValueError("Template checkpoint must be one of the source checkpoints") from exc
    loaded = [_load_and_hash(path) for path in source_paths]
    states = [state for state, _digest in loaded]
    source_hashes = [digest for _state, digest in loaded]
    for state in states:
        validate_training_checkpoint(state)
    template_signature = _model_signature(states[0]["model"])
    template_topology = _optimizer_topology(states[0]["optimizer"])
    for state in states[1:]:
        if _model_signature(state["model"]) != template_signature:
            raise ValueError("Source checkpoint model keys, shapes, or dtypes are incompatible")
        if _optimizer_topology(state["optimizer"]) != template_topology:
            raise ValueError("Source checkpoint optimizer topology is incompatible")
    averaged_model = {}
    for key, template_tensor in states[0]["model"].items():
        if not template_tensor.is_floating_point():
            if any(
                not torch.equal(state["model"][key], template_tensor)
                for state in states[1:]
            ):
                raise ValueError(
                    f"Source checkpoint nonfloating model tensor {key!r} is not identical"
                )
            averaged_model[key] = template_tensor.clone()
            continue
        accumulator = torch.zeros_like(template_tensor, dtype=torch.float64)
        for source_index, (state, weight) in enumerate(
            zip(states, normalized_weights)
        ):
            source_tensor = state["model"][key]
            if not torch.isfinite(source_tensor).all().item():
                raise ValueError(
                    f"Source checkpoint {source_index} has nonfinite floating model tensor {key!r}"
                )
            accumulator.add_(source_tensor.to(torch.float64), alpha=weight)
        averaged_tensor = accumulator.to(template_tensor.dtype)
        if not torch.isfinite(averaged_tensor).all().item():
            raise ValueError(f"Averaged floating model tensor {key!r} is nonfinite")
        averaged_model[key] = averaged_tensor
    manifest = {
        "format_version": AVERAGING_MANIFEST_FORMAT_VERSION,
        "operation": "same_basin_model_weight_average",
        "template": str(resolved_template),
        "sources": [
            {
                "path": str(path),
                "sha256": digest,
                "weight": weight,
            }
            for path, digest, weight in zip(
                resolved_sources, source_hashes, normalized_weights
            )
        ],
    }
    output = copy.deepcopy(states[template_index])
    output.pop("optimizer", None)
    output.pop("rng_state", None)
    output["model"] = averaged_model
    output["checkpoint_type"] = "inference_only_model_average"
    output["inference_only"] = True
    output["training_state_removed"] = ["optimizer", "rng_state"]
    output["averaging_provenance"] = copy.deepcopy(manifest)
    if manifest_path is None:
        manifest_path = Path(str(output_path) + ".manifest.json")
    _publish_output_and_manifest(
        output, output_path, manifest, manifest_path
    )
    return output


def _main(argv=None):
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    if len(arguments.sources) < 2:
        parser.error("at least two source checkpoints are required")
    try:
        average_checkpoints(
            arguments.sources,
            arguments.output,
            weights=arguments.weights,
            template_path=arguments.template,
            manifest_path=arguments.manifest,
        )
    except Exception as exc:
        parser.error(f"averaging failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
