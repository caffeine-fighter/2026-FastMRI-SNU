#!/usr/bin/env python3
"""Validate EXP035 provenance and print two matched continuation commands.

This script is deliberately dry-run only. It never imports torch, initializes CUDA,
or starts training.
"""

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path


SOURCE_GENERATION = "3e8af14268a64d67a308ebe30484ddf2"
SOURCE_SHA256 = "dc6e034f18df2a7872c416d4dccb4bb00e6e5b41fb89e438a86682db3097ffb7"
SOURCE_EPOCH = 30
TOTAL_EPOCHS = 35
CONTROL_NAME = "LOCAL_EXP035_E30_TO_E35_ADAM_LR1E3_SEED430"
CANDIDATE_NAME = "LOCAL_EXP035_E30_TO_E35_ADAM_LR3E4_SEED430"
REQUIRED_CUDA_DEVICE_NAME = "NVIDIA GeForce GTX 1080"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(checkpoint, expected_generation, expected_sha256):
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink():
        raise ValueError("Source checkpoint must be an immutable regular file, not a symlink")
    if not checkpoint.is_file():
        raise ValueError(f"Source checkpoint is not a regular file: {checkpoint}")
    if expected_generation not in checkpoint.name:
        raise ValueError(
            "Source checkpoint filename does not contain the expected immutable "
            f"generation {expected_generation}"
        )
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Source checkpoint SHA-256 mismatch: expected={expected_sha256} "
            f"actual={actual_sha256}"
        )
    return checkpoint.resolve(), actual_sha256, checkpoint.stat()


def _command(
    *,
    python_command,
    gpu_num,
    net_name,
    learning_rate,
    checkpoint,
    checkpoint_sha256,
    train_dir,
    val_dir,
):
    return [
        python_command,
        "train.py",
        "--GPU-NUM",
        str(gpu_num),
        "--require-cuda-device-name",
        REQUIRED_CUDA_DEVICE_NAME,
        "--batch-size",
        "1",
        "--num-epochs",
        str(TOTAL_EPOCHS),
        "--lr",
        str(learning_rate),
        "--net-name",
        net_name,
        "--data-path-train",
        str(train_dir),
        "--data-path-val",
        str(val_dir),
        "--cascade",
        "8",
        "--chans",
        "12",
        "--sens_chans",
        "8",
        "--seed",
        "430",
        "--resume-checkpoint",
        str(checkpoint),
        "--resume-checkpoint-sha256",
        checkpoint_sha256,
        "--resume-lr",
        str(learning_rate),
        "--retain-val-epochs",
    ]


def build_plan(
    checkpoint,
    *,
    candidate_checkpoint=None,
    result_root=Path("../result"),
    train_dir=Path("/root/Data/train"),
    val_dir=Path("/root/Data/val"),
    gpu_num=0,
    python_command="python",
    expected_generation=SOURCE_GENERATION,
    expected_sha256=SOURCE_SHA256,
    require_data=False,
    name_suffix="",
):
    if re.fullmatch(r"(?:_[A-Z0-9]+)*", name_suffix) is None:
        raise ValueError("Recovery name suffix must contain only _-prefixed uppercase alphanumerics")
    if name_suffix and candidate_checkpoint is None:
        raise ValueError("Recovery plans require a distinct candidate checkpoint")
    control_checkpoint, control_sha256, control_stat = _validate_source(
        checkpoint, expected_generation, expected_sha256
    )
    if candidate_checkpoint is None:
        candidate_checkpoint = control_checkpoint
        candidate_sha256 = control_sha256
        candidate_stat = control_stat
    else:
        candidate_checkpoint, candidate_sha256, candidate_stat = _validate_source(
            candidate_checkpoint, expected_generation, expected_sha256
        )
        if (control_stat.st_dev, control_stat.st_ino) == (
            candidate_stat.st_dev,
            candidate_stat.st_ino,
        ):
            raise ValueError(
                "Control and candidate checkpoints must be distinct private copies, not hardlinks"
            )

    result_root = Path(result_root)
    control_name = f"{CONTROL_NAME}{name_suffix}"
    candidate_name = f"{CANDIDATE_NAME}{name_suffix}"
    for name in (control_name, candidate_name):
        destination = result_root / name
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Continuation output already exists: {destination}")

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    if require_data:
        for description, path in (("training data", train_dir), ("validation data", val_dir)):
            if not path.is_dir():
                raise ValueError(f"Missing {description} directory: {path}")

    arms = []
    for role, name, learning_rate, source_checkpoint, source_sha256 in (
        (
            "fixed_lr_control",
            control_name,
            0.001,
            control_checkpoint,
            control_sha256,
        ),
        (
            "lower_lr_candidate",
            candidate_name,
            0.0003,
            candidate_checkpoint,
            candidate_sha256,
        ),
    ):
        command = _command(
            python_command=python_command,
            gpu_num=gpu_num,
            net_name=name,
            learning_rate=learning_rate,
            checkpoint=source_checkpoint,
            checkpoint_sha256=source_sha256,
            train_dir=train_dir,
            val_dir=val_dir,
        )
        arms.append(
            {
                "role": role,
                "net_name": name,
                "learning_rate": learning_rate,
                "command": command,
                "shell": shlex.join(command),
            }
        )

    return {
        "mode": "dry_run_only",
        "gpu_started": False,
        "source": {
            "experiment": "EXP035_varnet_c8_ch12_s8_e30",
            "epoch": SOURCE_EPOCH,
            "generation": expected_generation,
            "sha256": control_sha256,
            "checkpoint": str(control_checkpoint),
            "local_quality": 0.9199788092310326,
        },
        "candidate_source": {
            "experiment": "EXP035_varnet_c8_ch12_s8_e30",
            "epoch": SOURCE_EPOCH,
            "generation": expected_generation,
            "sha256": candidate_sha256,
            "checkpoint": str(candidate_checkpoint),
            "local_quality": 0.9199788092310326,
        },
        "fixed": {
            "architecture": "c8/ch12/s8",
            "optimizer": "Adam",
            "batch_size": 1,
            "seed": 430,
            "total_epochs": TOTAL_EPOCHS,
            "retained_epochs": [31, 32, 33, 34, 35],
            "objective": "standard_ssim",
        },
        "arms": arms,
        "promotion_gate": {
            "minimum_quality_gain_over_control": 0.0005,
            "protected_components": [
                "acc4_full",
                "acc4_bbox",
                "acc8_full",
                "acc8_bbox",
            ],
            "automatic_training": False,
            "automatic_official_evaluation": False,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
        help="Distinct private candidate source required for recovery suffixes",
    )
    parser.add_argument("--result-root", type=Path, default=Path("../result"))
    parser.add_argument("--train-dir", type=Path, default=Path("/root/Data/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("/root/Data/val"))
    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument("--python-command", default="python")
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Optional matched recovery suffix, for example _R1",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Also require the configured train/validation directories to exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        plan = build_plan(
            args.checkpoint,
            candidate_checkpoint=args.candidate_checkpoint,
            result_root=args.result_root,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            gpu_num=args.gpu_num,
            python_command=args.python_command,
            require_data=args.require_data,
            name_suffix=args.name_suffix,
        )
    except ValueError as exc:
        raise SystemExit(f"PRECHECK FAILED: {exc}") from exc
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
