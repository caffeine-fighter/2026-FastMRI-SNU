"""Metadata-only PromptMR+ run planning; never imports torch or a model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shlex
import shutil
from importlib import metadata as importlib_metadata
from pathlib import Path

from packaging.version import InvalidVersion, Version

from utils.promptmr.contracts import (
    PROMPTMR_PLUS_RECIPE,
    parse_acceleration_filename,
    run_name_component,
)

_GIB = 1024 ** 3


def _read_h5_metadata(
    path: Path,
    input_key: str = "kspace",
    target_key: str = "image_label",
    max_key: str = "max",
) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        if input_key not in handle or "mask" not in handle:
            raise ValueError(
                f"{path.name} must contain {input_key!r} and 'mask'"
            )
        dataset = handle[input_key]
        kspace_shape = tuple(dataset.shape)
        if (
            len(kspace_shape) != 4
            or any(dimension <= 0 for dimension in kspace_shape)
            or dataset.dtype.kind != "c"
        ):
            raise ValueError(
                f"Expected non-empty complex [slice, coil, height, width] "
                f"{input_key!r} in {path.name}"
            )
        slices = int(kspace_shape[0])
        sample_bytes = int(5 * math.prod(kspace_shape[1:]) * dataset.dtype.itemsize)
        volume_bytes = int(math.prod(kspace_shape) * dataset.dtype.itemsize)
        mask = handle["mask"]
        mask_shape = tuple(mask.shape)
        if len(mask_shape) == 1:
            mask_width = mask_shape[0]
        elif len(mask_shape) == 2 and mask_shape[0] == 1:
            mask_width = mask_shape[1]
        elif len(mask_shape) == 3 and mask_shape[0] == mask_shape[2] == 1:
            mask_width = mask_shape[1]
        elif (
            len(mask_shape) == 4
            and mask_shape[0] == mask_shape[1] == mask_shape[3] == 1
        ):
            mask_width = mask_shape[2]
        else:
            raise ValueError(f"Unsupported sampling mask shape in {path.name}: {mask_shape}")
        if mask_width <= 0 or mask_width != kspace_shape[-1]:
            raise ValueError(
                f"Sampling mask width mismatch in {path.name}: "
                f"mask={mask_width}, kspace={kspace_shape[-1]}"
            )
        mask_bytes = int(math.prod(mask_shape) * mask.dtype.itemsize)

    target_path = path.parent.parent / "image" / path.name
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing target volume: {target_path}")
    with h5py.File(target_path, "r") as handle:
        if target_key not in handle or max_key not in handle.attrs:
            raise ValueError(
                f"{target_path.name} must contain {target_key!r} and {max_key!r} attr"
            )
        target = handle[target_key]
        if (
            len(target.shape) != 3
            or target.shape[0] != slices
            or tuple(target.shape[1:]) != kspace_shape[-2:]
        ):
            raise ValueError(
                f"Target slice/geometry contract mismatch in {target_path.name}"
            )
        try:
            maximum = float(handle.attrs[max_key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid {max_key!r} attr in {target_path.name}") from exc
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError(f"Invalid {max_key!r} attr in {target_path.name}")
        target_bytes = int(math.prod(target.shape[1:]) * target.dtype.itemsize)
        target_volume_bytes = int(math.prod(target.shape) * target.dtype.itemsize)

    return {
        "slices": slices,
        "sample_bytes": sample_bytes + mask_bytes,
        "target_bytes": target_bytes,
        "volume_bytes": volume_bytes + target_volume_bytes + mask_bytes,
    }


def collect_dataset_stats(
    data_path: Path,
    *,
    input_key: str = "kspace",
    target_key: str = "image_label",
    metadata_reader=None,
) -> dict:
    """Count exact acc4/acc8 volumes and slices without loading array payloads."""
    data_path = Path(data_path)
    kspace_path = data_path / "kspace"
    if not kspace_path.is_dir():
        raise FileNotFoundError(f"Missing kspace directory: {kspace_path}")
    files = sorted(kspace_path.glob("*.h5"))
    if not files:
        raise ValueError(f"No HDF5 volumes found in {kspace_path}")
    reader = metadata_reader or (
        lambda path: _read_h5_metadata(path, input_key=input_key, target_key=target_key)
    )
    volumes = {4: 0, 8: 0}
    slices = {4: 0, 8: 0}
    max_sample_bytes = 0
    total_volume_bytes = 0
    for path in files:
        acceleration = parse_acceleration_filename(path.name)
        metadata = reader(path)
        count = metadata.get("slices")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Invalid slice count in {path.name}")
        volumes[acceleration] += 1
        slices[acceleration] += count
        max_sample_bytes = max(
            max_sample_bytes,
            int(metadata.get("sample_bytes", 0)) + int(metadata.get("target_bytes", 0)),
        )
        total_volume_bytes += int(metadata.get("volume_bytes", 0))
    if not all(volumes.values()):
        raise ValueError(
            f"PromptMR+ requires both acc4 and acc8 routes; observed volumes={volumes}"
        )
    return {
        "path": str(data_path.absolute()),
        "volumes": volumes,
        "slices": slices,
        "max_sample_bytes": max_sample_bytes,
        "total_volume_bytes": total_volume_bytes,
    }


def estimate_run(
    stats: dict,
    *,
    batch_size: int,
    epochs: int,
    retain_val_epochs: bool,
    checkpoint_reserve_gib: float,
    retained_stats: dict | None = None,
) -> dict:
    if batch_size <= 0 or epochs <= 0 or checkpoint_reserve_gib <= 0:
        raise ValueError("Batch size, epochs, and checkpoint reserve must be positive")
    total_slices = sum(stats["slices"].values())
    steps = math.ceil(total_slices / batch_size)
    raw_batch = int(stats.get("max_sample_bytes", 0)) * batch_size
    cpu_ram = 512 * 1024 ** 2 + 4 * raw_batch
    retained_source = retained_stats if retained_stats is not None else stats
    retained = (
        int(retained_source.get("total_volume_bytes", 0)) * epochs
        if retain_val_epochs
        else 0
    )
    checkpoint = math.ceil(checkpoint_reserve_gib * _GIB)
    metadata = epochs * 1024 ** 2
    one_arm_disk = (epochs + 2) * checkpoint + retained + metadata
    disk = 2 * one_arm_disk
    return {
        "steps_per_epoch": steps,
        "total_steps": steps * epochs,
        "cpu_ram_estimate_bytes": cpu_ram,
        "disk_estimate_bytes": disk,
        "assumptions": {
            "cpu_ram": "512 MiB process reserve plus 4x one raw adjacent-slice batch",
            "disk": (
                "matched candidate and control arms, each with one immutable "
                "checkpoint per epoch, two stable checkpoint aliases, retained "
                "validation payloads, and 1 MiB metadata reserve per epoch"
            ),
            "checkpoint_reserve_gib": checkpoint_reserve_gib,
        },
    }


def _dependency_status() -> dict[str, bool]:
    status = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "einops", "h5py", "numpy", "skimage", "cv2")
    }
    if status["torch"]:
        try:
            status["torch"] = (
                Version(importlib_metadata.version("torch")) >= Version("2.3")
            )
        except (importlib_metadata.PackageNotFoundError, InvalidVersion):
            status["torch"] = False
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan a pinned PromptMR+ Vessl run from local HDF5 metadata only."
    )
    parser.add_argument("--train-data-path", type=Path, required=True)
    parser.add_argument("--val-data-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1,), default=1)
    parser.add_argument(
        "--epochs", type=int, default=PROMPTMR_PLUS_RECIPE["training"]["max_epochs"]
    )
    parser.add_argument("--input-key", default="kspace")
    parser.add_argument("--target-key", default="image_label")
    parser.add_argument("--retain-val-epochs", action="store_true")
    parser.add_argument("--checkpoint-reserve-gib", type=float, default=8.0)
    parser.add_argument("--output-parent", type=Path, default=Path("../result"))
    parser.add_argument(
        "--control-run-name", type=run_name_component,
        default="EXP036_varnet_control_e5_seed430",
    )
    parser.add_argument(
        "--candidate-run-name", type=run_name_component,
        default="EXP036_promptmr_plus_default_e5_seed430",
    )
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--report-interval", type=int, default=10)
    parser.add_argument("--license-confirmed", action="store_true")
    return parser


def _command(parts) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    train = collect_dataset_stats(
        args.train_data_path, input_key=args.input_key, target_key=args.target_key
    )
    validation = collect_dataset_stats(
        args.val_data_path, input_key=args.input_key, target_key=args.target_key
    )
    estimate = estimate_run(
        train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        retain_val_epochs=True,
        checkpoint_reserve_gib=args.checkpoint_reserve_gib,
        retained_stats=validation,
    )
    disk_probe = args.output_parent.absolute()
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    common = [
        "--GPU-NUM", args.gpu_number,
        "--batch-size", args.batch_size,
        "--num-epochs", args.epochs,
        "--report-interval", args.report_interval,
        "--result-root", args.output_parent,
        "--data-path-train", args.train_data_path,
        "--data-path-val", args.val_data_path,
        "--input-key", args.input_key,
        "--target-key", args.target_key,
        "--max-key", "max",
        "--seed", 430,
        "--retain-val-epochs",
    ]
    control_command = _command([
        "python", "-u", "train.py",
        "--model-family", "varnet",
        "--net-name", args.control_run_name,
        "--lr", 1e-3,
        "--cascade", 8,
        "--chans", 12,
        "--sens_chans", 8,
        *common,
    ])
    candidate_parts = [
        "python", "-u", "train.py",
        "--model-family", "promptmr_plus",
        "--net-name", args.candidate_run_name,
        "--lr", PROMPTMR_PLUS_RECIPE["optimizer"]["lr"],
        *common,
    ]
    if args.license_confirmed:
        candidate_parts.append("--confirm-promptmr-noncommercial-use")
    dependencies_present = _dependency_status()
    disk_sufficient = free_bytes >= estimate["disk_estimate_bytes"]
    plan = {
        "recipe_id": PROMPTMR_PLUS_RECIPE["recipe_id"],
        "train": train,
        "validation": validation,
        "estimate": estimate,
        "dependencies_present": dependencies_present,
        "disk_free_bytes": free_bytes,
        "disk_sufficient": disk_sufficient,
        "license_confirmed": args.license_confirmed,
        "candidate_launch_ready": (
            args.license_confirmed
            and disk_sufficient
            and all(dependencies_present.values())
        ),
        "commands": {
            "control": control_command,
            "candidate": _command(candidate_parts),
        },
        "creates_output_directories": False,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0
