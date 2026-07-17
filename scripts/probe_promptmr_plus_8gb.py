#!/usr/bin/env python3
"""Bounded, no-training PromptMR+ FP32 inference feasibility probe."""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import resource
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psutil
import torch

from utils.model.promptmr_plus_adapter import (
    PromptMRContractError,
    PromptMRInput,
    PromptMRNonFiniteError,
    PromptMRPlusAdapter,
    acceleration_from_filename,
    build_promptmr_plus,
    load_promptmr_plus_checkpoint,
    prepare_promptmr_input,
    verify_promptmr_plus_source,
)

EXPECTED_SLICE_SHAPES = {4: (15, 640, 480), 8: (15, 640, 400)}
EXPECTED_VOLUME_SHAPES = {4: (38, 640, 480), 8: (40, 640, 368)}
OFFICIAL_CHECKPOINT_SHA256 = {
    "brain": "42722018604944c567c598ddf5c488d135793ef337359a1da03ad3d4301e177e",
    "knee": "3f931e9fd5eed3f755c580760de04bf7b870bc5d0a9b39c5164955359f385a86",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceleration", type=int, choices=(4, 8), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/Data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-kind", choices=("brain", "knee"))
    parser.add_argument("--compute-sens-per-coil", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument(
        "--input-source", choices=("actual-h5", "synthetic"), default="actual-h5"
    )
    parser.add_argument(
        "--telemetry-mode",
        choices=("process-attributed", "device-level-unattributed"),
        default="process-attributed",
    )
    parser.add_argument("--expected-gpu-index", type=int, default=0)
    parser.add_argument("--expected-gpu-uuid")
    return parser.parse_args()


def validate_probe_mode(args):
    if args.telemetry_mode != "device-level-unattributed":
        return
    if (
        args.input_source != "synthetic"
        or args.expected_gpu_index != 0
        or not isinstance(args.expected_gpu_uuid, str)
        or not re.fullmatch(r"GPU-[0-9A-Za-z-]+", args.expected_gpu_uuid)
        or args.checkpoint is not None
        or args.checkpoint_kind is not None
    ):
        raise RuntimeError("device-level override contract violation")


def initial_probe_state(args):
    if args.checkpoint is None:
        weights_requested = "random_untrained"
    elif args.checkpoint_kind is None:
        weights_requested = "checkpoint_supplied_without_required_kind"
    else:
        weights_requested = (
            f"official_{args.checkpoint_kind}_namespace_feasibility_only"
        )
    return {
        "weights_requested": weights_requested,
        "checkpoint_loaded": False,
        "required_process_gpu_memory_evidence_present": False,
    }


def load_official_checkpoint(path, expected_hash):
    """Hash and deserialize one regular checkpoint through the same no-follow FD."""
    absolute = Path(os.path.abspath(path))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(absolute.anchor, directory_flags)
    file_fd = None
    try:
        for component in absolute.parts[1:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise RuntimeError("unsafe official checkpoint path") from exc
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = os.open(
                absolute.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
        except OSError as exc:
            raise RuntimeError("unsafe official checkpoint path") from exc
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise RuntimeError("unsafe official checkpoint path")
        with os.fdopen(file_fd, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError("official checkpoint checksum mismatch")
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
        return checkpoint, actual_hash
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def choose_cases(cases):
    if not cases:
        raise RuntimeError("no eligible PromptMR+ data case")
    return {
        "maximum_slice_input": max(
            cases,
            key=lambda item: item["shape"][1]
            * (((item["shape"][2] - 1) | 7) + 1)
            * (((item["shape"][3] - 1) | 7) + 1),
        ),
        "maximum_retained_volume": max(
            cases,
            key=lambda item: item["shape"][0]
            * item["shape"][2]
            * item["shape"][3],
        ),
    }


def find_actual_cases(data_root, acceleration):
    cases = []
    for root, _, names in os.walk(data_root):
        for name in names:
            if not name.endswith(".h5"):
                continue
            try:
                if acceleration_from_filename(name) != acceleration:
                    continue
            except ValueError:
                raise
            path = Path(root) / name
            with h5py.File(path, "r") as handle:
                if "kspace" not in handle or "mask" not in handle:
                    continue
                shape = tuple(handle["kspace"].shape)
                if len(shape) != 4:
                    continue
                mask = np.asarray(handle["mask"]).astype(bool, copy=False).reshape(-1)
                if mask.shape != (shape[3],):
                    raise RuntimeError(f"mask width mismatch for {name}")
            cases.append(
                {
                    "path": path,
                    "name": name,
                    "shape": shape,
                    "split": path.relative_to(data_root).parts[0],
                    "effective_acceleration": len(mask) / max(int(mask.sum()), 1),
                }
            )
    selected = choose_cases(cases)
    selected["eligible_case_count"] = len(cases)
    slice_shape = selected["maximum_slice_input"]["shape"][1:]
    volume_shape = (
        selected["maximum_retained_volume"]["shape"][0],
        *selected["maximum_retained_volume"]["shape"][2:],
    )
    if slice_shape != EXPECTED_SLICE_SHAPES[acceleration]:
        raise RuntimeError(
            f"maximum slice shape drift: {slice_shape} != {EXPECTED_SLICE_SHAPES[acceleration]}"
        )
    if volume_shape != EXPECTED_VOLUME_SHAPES[acceleration]:
        raise RuntimeError(
            f"maximum retained-volume shape drift: {volume_shape} != {EXPECTED_VOLUME_SHAPES[acceleration]}"
        )
    return selected


def synthetic_cases(acceleration):
    """Return official-shape synthetic cases without touching a dataset."""
    if acceleration not in EXPECTED_SLICE_SHAPES:
        raise ValueError("unsupported synthetic acceleration")
    coils, slice_height, slice_width = EXPECTED_SLICE_SHAPES[acceleration]
    volume_slices, volume_height, volume_width = EXPECTED_VOLUME_SHAPES[
        acceleration
    ]

    def case(name, shape):
        return {
            "path": None,
            "name": name,
            "shape": shape,
            "split": "synthetic",
            "effective_acceleration": None,
            "synthetic": True,
        }

    return {
        "maximum_slice_input": case(
            f"promptmr_synthetic_acc{acceleration}_maximum.h5",
            (5, coils, slice_height, slice_width),
        ),
        "maximum_retained_volume": case(
            f"promptmr_synthetic_acc{acceleration}_full_volume.h5",
            (volume_slices, coils, volume_height, volume_width),
        ),
        "eligible_case_count": 2,
    }


def make_synthetic_volume(shape, acceleration):
    """Create one deterministic slice and broadcast it across a legal volume."""
    if len(shape) != 4 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in shape
    ):
        raise PromptMRContractError("synthetic volume shape must be positive rank four")
    slices, coils, height, width = shape
    mask = np.zeros(width, dtype=np.bool_)
    mask[::acceleration] = True
    center = width // 2
    mask[max(0, center - 12) : min(width, center + 12)] = True
    base = np.zeros((1, coils, height, width), dtype=np.complex64)
    base[..., mask] = np.complex64(1.0 + 0.25j)
    volume = np.broadcast_to(base, (slices, coils, height, width))
    return volume, mask


def load_case_volume(case):
    if case.get("synthetic") is True:
        acceleration = acceleration_from_filename(case["name"])
        return make_synthetic_volume(case["shape"], acceleration)
    return load_volume(case["path"])


def load_volume(path):
    with h5py.File(path, "r") as handle:
        kspace = np.asarray(handle["kspace"])
        mask = np.asarray(handle["mask"]).astype(bool, copy=False).reshape(-1)
    return kspace, mask


def to_device(prepared, device):
    return PromptMRInput(
        kspace=prepared.kspace.to(device),
        mask=prepared.mask.to(device),
        num_low_frequencies=prepared.num_low_frequencies.to(device),
        acceleration=prepared.acceleration,
    )


def classify_failure(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "FAIL_CUDA_OOM"
    if isinstance(error, MemoryError):
        return "FAIL_HOST_OOM"
    if isinstance(error, OSError):
        return "FAIL_DATA_IO"
    if isinstance(error, PromptMRNonFiniteError):
        return "FAIL_NONFINITE"
    if isinstance(error, PromptMRContractError):
        return "FAIL_SHAPE_CONTRACT"
    message = str(error).lower()
    if (
        "gpu idle evidence unavailable" in message
        or "driver-version evidence unavailable" in message
        or "device-level gpu evidence unavailable" in message
    ):
        return "FAIL_ENVIRONMENT_EVIDENCE"
    if "gpu is not exclusive" in message or "gpu is not idle" in message:
        return "FAIL_GPU_BUSY"
    if "gpu memory evidence unavailable" in message:
        return "FAIL_MEMORY_EVIDENCE"
    if "non-finite" in message:
        return "FAIL_NONFINITE"
    if any(
        token in message
        for token in (
            "shape",
            "mask width mismatch",
            "schema",
            "contract",
        )
    ):
        return "FAIL_SHAPE_CONTRACT"
    if isinstance(error, NotImplementedError) or any(
        token in message for token in ("not implemented", "unsupported operation")
    ):
        return "FAIL_UNSUPPORTED_OPERATION"
    return "FAIL_RUNTIME"


def write_report_exclusive(path, report):
    """Publish one private report without following any directory symlink."""
    serialized = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    absolute = Path(os.path.abspath(path))
    if absolute.name in {"", ".", ".."}:
        raise ValueError("unsafe output path")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ValueError("unsafe output path") from exc
            os.close(directory_fd)
            directory_fd = next_fd
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        file_fd = os.open(absolute.name, flags, 0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(file_fd, "w", encoding="utf-8", closefd=False) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(file_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def compute_app_memory_mib():
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
        processes = {}
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                raise ValueError("unexpected compute-process field count")
            pid = int(fields[0])
            memory_mib = float(fields[1])
            if pid <= 0 or not math.isfinite(memory_mib) or memory_mib < 0:
                raise ValueError("invalid compute-process evidence value")
            if pid in processes:
                raise ValueError("duplicate compute-process PID evidence")
            processes[pid] = memory_mib
        return processes
    except (OSError, subprocess.CalledProcessError, ValueError):
        raise RuntimeError(
            "required current-process GPU memory evidence unavailable"
        ) from None


def assert_exclusive_gpu_process():
    other_pids = sorted(set(compute_app_memory_mib()) - {os.getpid()})
    if other_pids:
        raise RuntimeError("GPU is not exclusive to the probe")


def assert_gpu_idle_before_probe():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        lines = output.splitlines()
        if len(lines) != 1:
            raise ValueError("unexpected GPU count")
        fields = [field.strip() for field in lines[0].split(",")]
        if len(fields) != 2:
            raise ValueError("unexpected GPU idle field count")
        utilization, memory_used = map(float, fields)
        if (
            not math.isfinite(utilization)
            or not math.isfinite(memory_used)
            or not 0.0 <= utilization <= 100.0
            or memory_used < 0.0
        ):
            raise ValueError("invalid GPU idle evidence value")
    except (OSError, subprocess.CalledProcessError, ValueError):
        raise RuntimeError("required GPU idle evidence unavailable") from None
    if utilization != 0.0 or memory_used > 32.0:
        raise RuntimeError("GPU is not idle before probe")
    return {
        "utilization_percent": utilization,
        "memory_used_mib": memory_used,
    }


def device_level_gpu_snapshot(
    *, expected_gpu_index, expected_gpu_uuid, expected_compute_rows
):
    """Collect strict aggregate telemetry without claiming process attribution."""
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_result = subprocess.run(
            gpu_command, text=True, capture_output=True, check=False
        )
        process_result = subprocess.run(
            process_command, text=True, capture_output=True, check=False
        )
        if gpu_result.returncode != 0 or process_result.returncode != 0:
            raise ValueError("telemetry command failed")
        gpu_lines = gpu_result.stdout.strip().splitlines()
        if len(gpu_lines) != 1:
            raise ValueError("unexpected GPU count")
        fields = [field.strip() for field in gpu_lines[0].split(",")]
        if len(fields) != 5:
            raise ValueError("unexpected GPU field count")
        gpu_index = int(fields[0])
        gpu_uuid = fields[1]
        total_mib, used_mib, utilization = map(float, fields[2:])
        if (
            gpu_index < 0
            or gpu_index != expected_gpu_index
            or gpu_uuid != expected_gpu_uuid
            or not re.fullmatch(r"GPU-[0-9A-Za-z-]+", gpu_uuid)
            or not math.isfinite(total_mib)
            or not math.isfinite(used_mib)
            or not math.isfinite(utilization)
            or total_mib <= 0.0
            or used_mib < 0.0
            or used_mib > total_mib
            or not 0.0 <= utilization <= 100.0
            or not isinstance(expected_compute_rows, int)
            or isinstance(expected_compute_rows, bool)
            or expected_compute_rows < 0
        ):
            raise ValueError("invalid aggregate GPU evidence")
        process_rows = []
        for line in process_result.stdout.strip().splitlines():
            row = [field.strip() for field in line.split(",")]
            if len(row) != 2:
                raise ValueError("unexpected compute-process field count")
            row_uuid = row[0]
            memory_mib = float(row[1])
            if (
                row_uuid != expected_gpu_uuid
                or not math.isfinite(memory_mib)
                or memory_mib < 0.0
            ):
                raise ValueError("invalid aggregate compute evidence")
            process_rows.append(memory_mib)
        if len(process_rows) != expected_compute_rows:
            raise ValueError("unexpected compute-process row count")
    except (OSError, ValueError):
        raise RuntimeError("required device-level GPU evidence unavailable") from None
    return {
        "evidence_label": "DEVICE_LEVEL_UNATTRIBUTED",
        "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "memory_total_mib": total_mib,
        "memory_used_mib": used_mib,
        "utilization_percent": utilization,
        "compute_row_count": len(process_rows),
        "aggregate_compute_memory_mib": sum(process_rows),
        "command_exit_codes": [gpu_result.returncode, process_result.returncode],
    }


def preflight_telemetry(args):
    validate_probe_mode(args)
    if args.telemetry_mode == "device-level-unattributed":
        snapshot = device_level_gpu_snapshot(
            expected_gpu_index=args.expected_gpu_index,
            expected_gpu_uuid=args.expected_gpu_uuid,
            expected_compute_rows=0,
        )
        if (
            snapshot["memory_total_mib"] != 8192.0
            or snapshot["utilization_percent"] != 0.0
            or snapshot["memory_used_mib"] > 32.0
        ):
            raise RuntimeError("GPU is not idle before probe")
        return snapshot
    assert_exclusive_gpu_process()
    return assert_gpu_idle_before_probe()


def process_gpu_memory_mib():
    memory = compute_app_memory_mib()
    if os.getpid() in memory:
        return memory[os.getpid()]
    raise RuntimeError("required current-process GPU memory evidence unavailable")


def environment():
    properties = torch.cuda.get_device_properties(0)
    try:
        driver_version = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver_version):
            raise ValueError("unexpected driver-version evidence")
    except (OSError, subprocess.CalledProcessError, ValueError):
        raise RuntimeError(
            "required NVIDIA driver-version evidence unavailable"
        ) from None
    return {
        "platform": "VESSL",
        "gpu": properties.name,
        "gpu_total_memory_mib": properties.total_memory / 2**20,
        "driver_version": driver_version,
        "compute_capability": [properties.major, properties.minor],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "host_total_gib": psutil.virtual_memory().total / 2**30,
        "host_available_gib_before": psutil.virtual_memory().available / 2**30,
    }


def checkpoint_metadata(args, model, manifest, state):
    if args.checkpoint is None:
        if args.checkpoint_kind is not None:
            raise RuntimeError("checkpoint kind requires a checkpoint")
        return {"weights": "random_untrained", "quality_evidence": False}
    if args.checkpoint_kind is None:
        raise RuntimeError("official checkpoint kind is required")
    entry_name = f"fastmri_{args.checkpoint_kind}_promptmr_plus"
    expected = manifest["checkpoints"][entry_name]
    if expected["sha256"] != OFFICIAL_CHECKPOINT_SHA256[args.checkpoint_kind]:
        raise RuntimeError("manifest checkpoint trust anchor mismatch")
    checkpoint, actual_hash = load_official_checkpoint(
        args.checkpoint, expected["sha256"]
    )
    load_promptmr_plus_checkpoint(model, checkpoint)
    state.update(
        {
            "checkpoint_loaded": True,
            "checkpoint_anatomy": args.checkpoint_kind,
            "checkpoint_sha256": actual_hash,
        }
    )
    hyperparameters = checkpoint.get("hyper_parameters", {})
    return {
        "weights": "official_pretrained_namespace_feasibility_only",
        "quality_evidence": False,
        "anatomy": args.checkpoint_kind,
        "filename": args.checkpoint.name,
        "official_url": expected["url"],
        "sha256": actual_hash,
        "trust_anchor": "immutable SOURCE_MANIFEST.json checkpoint SHA-256",
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "model_version": hyperparameters.get("model_version"),
        "strict_core_load": True,
        "training_data_use": "initialization/namespace feasibility only",
    }


def validate_output_shape(output, expected_shape, label):
    if tuple(output.shape) != tuple(expected_shape):
        raise PromptMRContractError(f"{label} output shape contract failure")


def case_report(case, data_root):
    if case.get("synthetic") is True:
        return {
            "filename": case["name"],
            "dataset_split": "synthetic",
            "volume_shape_slices_coils_height_width": list(case["shape"]),
            "effective_acceleration_observed_only": None,
            "nominal_acceleration_source": "unique acc4/acc8 synthetic filename token",
            "data_open_mode": "NOT_OPENED_SYNTHETIC",
        }
    return {
        "filename": case["name"],
        "dataset_split": case["split"],
        "volume_shape_slices_coils_height_width": list(case["shape"]),
        "effective_acceleration_observed_only": case["effective_acceleration"],
        "nominal_acceleration_source": "unique acc4/acc8 filename token",
        "data_open_mode": "read_only",
    }


def run_probe(args, state):
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0") not in ("0", ""):
        raise RuntimeError("probe requires exactly the assigned CUDA device")
    preflight_gpu_idle = preflight_telemetry(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one visible CUDA device")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce GTX 1080":
        raise RuntimeError("probe requires the VESSL NVIDIA GeForce GTX 1080")

    manifest = verify_promptmr_plus_source()
    selected = (
        synthetic_cases(args.acceleration)
        if args.input_source == "synthetic"
        else find_actual_cases(args.data_root, args.acceleration)
    )
    slice_case = selected["maximum_slice_input"]
    volume_case = selected["maximum_retained_volume"]
    report = {
        "schema": "promptmr_plus_vessl_feasibility_v1",
        "status": "RUNNING",
        "source": {
            "url": manifest["repository"],
            "commit": manifest["commit"],
            "implementation": "models/promptmr_v2.py::PromptMR",
            "implementation_sha256": manifest["files"][
                "upstream/models/promptmr_v2.py"
            ],
            "license": manifest["license"],
            "upstream_algorithm_modified": False,
            "runtime_compatibility_shim": "mirror missing PromptMRBlock.n_buffer from its upstream model.n_buffer",
        },
        "config": {
            **manifest["model_config"],
            "batch_size": 1,
            "dtype": "float32",
            "bf16_used": False,
            "fp16_used": False,
            "input_source": args.input_source,
            "telemetry_mode": args.telemetry_mode,
            "evidence_label": (
                "DEVICE_LEVEL_UNATTRIBUTED"
                if args.telemetry_mode == "device-level-unattributed"
                else "PROCESS_LEVEL_ATTRIBUTED"
            ),
            "pid_attribution_status": (
                "UNAVAILABLE"
                if args.telemetry_mode == "device-level-unattributed"
                else "REQUIRED"
            ),
            "grad_mode": "torch.inference_mode",
            "warmup_slices": 1,
            "latency_measurement_repeats": 3,
            "compute_sens_per_coil": args.compute_sens_per_coil,
            "use_checkpoint": args.use_checkpoint,
            "adapter_forward_latency_scope": "synchronized PromptMRPlusAdapter forward after H2D; includes output validation",
            "single_slice_harness_latency_scope": "host adapter + H2D + synchronized adapter forward + D2H; external host finite check excluded",
            "full_volume_retention_latency_scope": "host adapter + H2D + synchronized adapter forward + exact output-shape gate + D2H + GPU retention append; external host finite check excluded",
        },
        "actual_case": {
            "selection": "maximum coils * spatial shape after upstream 8-pixel padding",
            "eligible_case_count": selected["eligible_case_count"],
            **case_report(slice_case, args.data_root),
            "raw_slice_input_shape": [
                1,
                slice_case["shape"][1],
                slice_case["shape"][2],
                slice_case["shape"][3],
                2,
            ],
            "effective_adjacent_model_input_shape": [
                1,
                5 * slice_case["shape"][1],
                slice_case["shape"][2],
                slice_case["shape"][3],
                2,
            ],
        },
        "full_volume_case": {
            "selection": "maximum slices * height * width retained output elements",
            **case_report(volume_case, args.data_root),
            "retained_output_shape": [
                volume_case["shape"][0],
                volume_case["shape"][2],
                volume_case["shape"][3],
            ],
        },
        "environment": {
            **environment(),
            "preflight_gpu_idle": preflight_gpu_idle,
            "telemetry_evidence_label": (
                "DEVICE_LEVEL_UNATTRIBUTED"
                if args.telemetry_mode == "device-level-unattributed"
                else "PROCESS_LEVEL_ATTRIBUTED"
            ),
            "pid_attribution_status": (
                "UNAVAILABLE"
                if args.telemetry_mode == "device-level-unattributed"
                else "REQUIRED"
            ),
        },
        "h5_accessed": args.input_source == "actual-h5",
        "dataset_loaded": args.input_source == "actual-h5",
        "model_inference_executed": True,
        "training_executed": False,
        "official_evaluation_executed": False,
        "leaderboard_submission_executed": False,
    }

    device = torch.device("cuda:0")
    torch.manual_seed(430)
    model = build_promptmr_plus()
    report["checkpoint"] = checkpoint_metadata(args, model, manifest, state)
    report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    report["parameter_bytes_fp32"] = report["parameter_count"] * 4
    adapter = PromptMRPlusAdapter(model).eval().to(device)

    slice_volume, slice_mask = load_case_volume(slice_case)
    prepared_cpu = prepare_promptmr_input(
        slice_volume,
        slice_mask,
        slice_index=slice_volume.shape[0] // 2,
        filename=slice_case["name"],
    )
    prepared_gpu = to_device(prepared_cpu, device)
    expected_slice_output_shape = [
        1,
        slice_case["shape"][2],
        slice_case["shape"][3],
    ]
    with torch.inference_mode():
        warmup = adapter(
            prepared_gpu,
            compute_sens_per_coil=args.compute_sens_per_coil,
            use_checkpoint=args.use_checkpoint,
        )
        torch.cuda.synchronize()
        validate_output_shape(
            warmup, expected_slice_output_shape, "maximum-slice warmup"
        )
        del warmup
        torch.cuda.reset_peak_memory_stats(device)
        forward_latencies = []
        output_shape = None
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = adapter(
                prepared_gpu,
                compute_sens_per_coil=args.compute_sens_per_coil,
                use_checkpoint=args.use_checkpoint,
            )
            torch.cuda.synchronize()
            forward_latencies.append((time.perf_counter() - started) * 1000)
            output_shape = list(output.shape)
            validate_output_shape(
                output, expected_slice_output_shape, "maximum-slice"
            )
            del output
        repeat_allocated = torch.cuda.max_memory_allocated(device) / 2**20
        repeat_reserved = torch.cuda.max_memory_reserved(device) / 2**20
        if args.telemetry_mode == "device-level-unattributed":
            repeat_device_snapshot = device_level_gpu_snapshot(
                expected_gpu_index=args.expected_gpu_index,
                expected_gpu_uuid=args.expected_gpu_uuid,
                expected_compute_rows=1,
            )
            repeat_process_memory = None
        else:
            repeat_device_snapshot = None
            repeat_process_memory = process_gpu_memory_mib()
            state["required_process_gpu_memory_evidence_present"] = True

        torch.cuda.synchronize()
        harness_started = time.perf_counter()
        harness_prepared = prepare_promptmr_input(
            slice_volume,
            slice_mask,
            slice_index=slice_volume.shape[0] // 2,
            filename=slice_case["name"],
        )
        harness_output_gpu = adapter(
            to_device(harness_prepared, device),
            compute_sens_per_coil=args.compute_sens_per_coil,
            use_checkpoint=args.use_checkpoint,
        )
        validate_output_shape(
            harness_output_gpu,
            expected_slice_output_shape,
            "maximum-slice harness",
        )
        harness_output = harness_output_gpu.cpu()
        harness_latency = (time.perf_counter() - harness_started) * 1000
        if not torch.isfinite(harness_output).all():
            raise RuntimeError("non-finite harness-like slice output")
        del harness_output_gpu, harness_output, harness_prepared, prepared_cpu, prepared_gpu
        del slice_volume, slice_mask
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        volume, volume_mask = load_case_volume(volume_case)
        retained_outputs = []
        expected_item_output_shape = (1, volume.shape[2], volume.shape[3])
        volume_forward_latencies = []
        volume_harness_latencies = []
        item_output = None
        for index in range(volume.shape[0]):
            torch.cuda.synchronize()
            end_to_end_started = time.perf_counter()
            item_cpu = prepare_promptmr_input(
                volume,
                volume_mask,
                slice_index=index,
                filename=volume_case["name"],
            )
            item_gpu = to_device(item_cpu, device)
            torch.cuda.synchronize()
            forward_started = time.perf_counter()
            item_output = adapter(
                item_gpu,
                compute_sens_per_coil=args.compute_sens_per_coil,
                use_checkpoint=args.use_checkpoint,
            )
            torch.cuda.synchronize()
            volume_forward_latencies.append(
                (time.perf_counter() - forward_started) * 1000
            )
            validate_output_shape(
                item_output, expected_item_output_shape, "full-volume slice"
            )
            item_output_host = item_output.detach().cpu()
            retained_outputs.append(item_output)
            volume_harness_latencies.append(
                (time.perf_counter() - end_to_end_started) * 1000
            )
            if not torch.isfinite(item_output_host).all():
                raise PromptMRNonFiniteError(
                    f"non-finite full-volume output at slice {index}"
                )
            del item_cpu, item_gpu, item_output_host
        retained = torch.cat(retained_outputs, dim=0)
        expected_output_shape = (
            volume.shape[0],
            volume.shape[2],
            volume.shape[3],
        )
        if tuple(retained.shape) != expected_output_shape:
            raise PromptMRContractError(
                f"unexpected retained output shape {tuple(retained.shape)}"
            )
        if not torch.isfinite(retained).all():
            raise PromptMRNonFiniteError("non-finite retained full-volume output")
        retained_host = retained.detach().cpu()
        if tuple(retained_host.shape) != expected_output_shape:
            raise PromptMRContractError("host-retained output shape contract failure")
        volume_allocated = torch.cuda.max_memory_allocated(device) / 2**20
        volume_reserved = torch.cuda.max_memory_reserved(device) / 2**20
        if args.telemetry_mode == "device-level-unattributed":
            volume_device_snapshot = device_level_gpu_snapshot(
                expected_gpu_index=args.expected_gpu_index,
                expected_gpu_uuid=args.expected_gpu_uuid,
                expected_compute_rows=1,
            )
            volume_process_memory = None
        else:
            volume_device_snapshot = None
            volume_process_memory = process_gpu_memory_mib()

    report.update(
        {
            "status": "PASS",
            "output": {
                "shape": output_shape,
                "dtype": "float32",
                "finite": True,
            },
            "latency_ms_per_slice": {
                "repeated_adapter_forward_values": forward_latencies,
                "repeated_adapter_forward_mean": statistics.mean(forward_latencies),
                "repeated_adapter_forward_median": statistics.median(forward_latencies),
                "single_slice_harness_without_external_finite": harness_latency,
                "full_volume_adapter_forward_mean": statistics.mean(volume_forward_latencies),
                "full_volume_adapter_forward_max": max(volume_forward_latencies),
                "full_volume_retention_without_external_finite_mean": statistics.mean(
                    volume_harness_latencies
                ),
            },
            "memory_mib": {
                "peak_allocated_repeated_slice": repeat_allocated,
                "peak_reserved_repeated_slice": repeat_reserved,
                "process_gpu_memory_repeated_slice": repeat_process_memory,
                "peak_allocated_full_volume": volume_allocated,
                "peak_reserved_full_volume": volume_reserved,
                "process_gpu_memory_full_volume": volume_process_memory,
                "reserved_headroom_against_device_reported_total": report[
                    "environment"
                ]["gpu_total_memory_mib"]
                - volume_reserved,
            },
            "device_level_telemetry": {
                "evidence_label": (
                    "DEVICE_LEVEL_UNATTRIBUTED"
                    if args.telemetry_mode == "device-level-unattributed"
                    else "NOT_APPLICABLE_PROCESS_LEVEL_MODE"
                ),
                "preflight": (
                    preflight_gpu_idle
                    if args.telemetry_mode == "device-level-unattributed"
                    else None
                ),
                "maximum_slice": repeat_device_snapshot,
                "full_volume": volume_device_snapshot,
                "pid_attribution_status": (
                    "UNAVAILABLE"
                    if args.telemetry_mode == "device-level-unattributed"
                    else "REQUIRED"
                ),
            },
            "full_volume": {
                "processed": True,
                "shape_slices_height_width": list(retained.shape),
                "finite": True,
                "evaluator_faithful_gpu_retention": True,
                "post_volume_host_transfer_completed": True,
                "host_retained_shape": list(retained_host.shape),
            },
            "cuda_oom": False,
            "host_oom": False,
            "unsupported_operation": False,
            "nonfinite_output": False,
            "shape_contract_failure": False,
            "data_io_failure": False,
            "memory_evidence_failure": False,
            "gpu_busy_failure": False,
            "environment_evidence_failure": False,
            "required_process_gpu_memory_evidence_present": (
                args.telemetry_mode != "device-level-unattributed"
            ),
            "required_device_level_gpu_evidence_present": (
                args.telemetry_mode == "device-level-unattributed"
            ),
        }
    )
    del retained_host, retained, retained_outputs, item_output
    del volume, volume_mask, adapter, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    report["cleanup"] = {
        "gc_collected": True,
        "cuda_cache_emptied": True,
        "post_cleanup_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
        "post_cleanup_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
        "process_exit_required_for_zero_compute_rows": True,
    }
    return report


def main():
    args = parse_args()
    state = initial_probe_state(args)
    report = None
    try:
        report = run_probe(args, state)
    except Exception as error:
        status = classify_failure(error)
        report = {
            "schema": "promptmr_plus_vessl_feasibility_v1",
            "status": status,
            "acceleration": args.acceleration,
            "source": {
                "url": "https://github.com/hellopipu/PromptMR-plus",
                "commit": "934eeda6d4d18cd39e406fa1eee9e1f70603cb5e",
                "license": "Rutgers Non-commercial Research License",
                "license_status": "NONCOMMERCIAL_COMPETITION_USE_ALLOWED",
                "license_sha256": "c0c4c7d85180b493cd7a213d4509155d3734de26562e4490589963e1c356db21",
            },
            **state,
            "quality_evidence": False,
            "expected_raw_slice_input_shape": [
                1,
                *EXPECTED_SLICE_SHAPES[args.acceleration],
                2,
            ],
            "expected_retained_volume_shape": list(
                EXPECTED_VOLUME_SHAPES[args.acceleration]
            ),
            "memory_control": {
                "compute_sens_per_coil": args.compute_sens_per_coil,
                "use_checkpoint": args.use_checkpoint,
            },
            "cuda_oom": status == "FAIL_CUDA_OOM",
            "host_oom": status == "FAIL_HOST_OOM",
            "unsupported_operation": status == "FAIL_UNSUPPORTED_OPERATION",
            "nonfinite_output": status == "FAIL_NONFINITE",
            "shape_contract_failure": status == "FAIL_SHAPE_CONTRACT",
            "data_io_failure": status == "FAIL_DATA_IO",
            "memory_evidence_failure": status == "FAIL_MEMORY_EVIDENCE",
            "gpu_busy_failure": status == "FAIL_GPU_BUSY",
            "environment_evidence_failure": status == "FAIL_ENVIRONMENT_EVIDENCE",
            "error_type": type(error).__name__,
            "error_detail": "redacted; inspect local stderr only",
            "training_executed": False,
            "official_evaluation_executed": False,
            "leaderboard_submission_executed": False,
        }
        if torch.cuda.is_initialized():
            report["memory_mib"] = {
                "peak_allocated_before_failure": torch.cuda.max_memory_allocated(0)
                / 2**20,
                "peak_reserved_before_failure": torch.cuda.max_memory_reserved(0)
                / 2**20,
            }
    report["acceleration"] = args.acceleration
    report["process_peak_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    )
    report["host_available_gib_after"] = psutil.virtual_memory().available / 2**30
    report["completed_at_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    try:
        write_report_exclusive(args.output, report)
    except Exception as publication_error:
        print(
            json.dumps(
                {
                    "status": "FAIL_REPORT_PUBLICATION",
                    "error_type": type(publication_error).__name__,
                    "error_detail": "redacted",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "acceleration": args.acceleration,
                "result_published": True,
                "parameter_count": report.get("parameter_count"),
                "output": report.get("output"),
                "latency_ms_per_slice": report.get("latency_ms_per_slice"),
                "memory_mib": report.get("memory_mib"),
                "full_volume": report.get("full_volume"),
                "cuda_oom": report.get("cuda_oom"),
                "unsupported_operation": report.get("unsupported_operation"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
