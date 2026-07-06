#!/usr/bin/env python3
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

def main():
    print("=== Run context ===")
    print(f"hostname: {socket.gethostname()}")
    print(f"platform: {platform.platform()}")
    print(f"python: {sys.version.split()[0]}")
    print(f"cwd: {Path.cwd()}")
    print(f"git_branch: {run(['git', 'branch', '--show-current'])}")
    print(f"git_commit: {run(['git', 'rev-parse', '--short', 'HEAD'])}")

    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch: unavailable: {exc}")

    for key in ["FASTMRI_DATA_DIR", "FASTMRI_RUN_DIR", "FASTMRI_RESULT_DIR", "CUDA_VISIBLE_DEVICES"]:
        print(f"{key}: {os.environ.get(key, '<unset>')}")

if __name__ == "__main__":
    main()
