#!/usr/bin/env python3
"""Print a reproducible training/inference environment fingerprint."""

import argparse
import hashlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_summary():
    try:
        import psutil

        memory = psutil.virtual_memory()
        return f"total={memory.total} available={memory.available}"
    except Exception:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            wanted = []
            for line in meminfo.read_text().splitlines():
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    wanted.append(line.strip())
            if wanted:
                return " ".join(wanted)
    return "unavailable"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-gpu-probe",
        action="store_true",
        help="Do not call torch.cuda or nvidia-smi; safe while another process owns the GPU",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional immutable checkpoint whose size and SHA-256 should be recorded",
    )
    parser.add_argument(
        "--include-packages",
        action="store_true",
        help="Append the exact `python -m pip freeze` output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    git_prefix = ["git", "-c", f"safe.directory={Path.cwd().as_posix()}"]
    print("=== Run context ===")
    print(f"hostname: {socket.gethostname()}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"processor: {platform.processor() or '<unknown>'}")
    print(f"logical_cpu_count: {os.cpu_count()}")
    print(f"memory: {memory_summary()}")
    print(f"python: {sys.version.replace(os.linesep, ' ')}")
    print(f"python_executable: {sys.executable}")
    print(f"cwd: {Path.cwd()}")
    print(f"git_branch: {run([*git_prefix, 'branch', '--show-current'])}")
    print(f"git_commit: {run([*git_prefix, 'rev-parse', 'HEAD'])}")
    print(
        f"git_status: "
        f"{run([*git_prefix, 'status', '--porcelain=v1']) or '<clean>'}"
    )

    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise SystemExit(f"checkpoint is not a regular file: {args.checkpoint}")
        print(f"checkpoint: {args.checkpoint.resolve()}")
        print(f"checkpoint_size_bytes: {args.checkpoint.stat().st_size}")
        print(f"checkpoint_sha256: {sha256_file(args.checkpoint)}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch_compiled_cuda: {torch.version.cuda}")
        print(f"cudnn: {torch.backends.cudnn.version()}")
        if args.no_gpu_probe:
            print("gpu_probe: disabled")
        else:
            cuda_available = torch.cuda.is_available()
            print(f"cuda_available: {cuda_available}")
            if cuda_available:
                print(f"cuda_device_count: {torch.cuda.device_count()}")
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    print(f"cuda_device_{index}: {properties.name}")
                    print(f"cuda_device_{index}_total_memory: {properties.total_memory}")
                    print(
                        f"cuda_device_{index}_capability: "
                        f"{properties.major}.{properties.minor}"
                    )
            print(
                "nvidia_smi: "
                + run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    ]
                )
            )
    except Exception as exc:
        print(f"torch: unavailable: {exc}")
        print("gpu_probe: disabled" if args.no_gpu_probe else "gpu_probe: unavailable")

    for key in [
        "FASTMRI_DATA_DIR",
        "FASTMRI_RUN_DIR",
        "FASTMRI_RESULT_DIR",
        "CUDA_VISIBLE_DEVICES",
    ]:
        print(f"{key}: {os.environ.get(key, '<unset>')}")

    if args.include_packages:
        print("=== pip freeze ===")
        print(run([sys.executable, "-m", "pip", "freeze"]))


if __name__ == "__main__":
    main()
