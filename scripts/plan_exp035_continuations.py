#!/usr/bin/env python3
"""Validate EXP035 provenance and print two matched continuation commands.

This script is deliberately dry-run only. It never imports torch, initializes CUDA,
or starts training.
"""

import argparse
import hashlib
import json
import shlex
from pathlib import Path


SOURCE_GENERATION = "3e8af14268a64d67a308ebe30484ddf2"
SOURCE_SHA256 = "dc6e034f18df2a7872c416d4dccb4bb00e6e5b41fb89e438a86682db3097ffb7"
SOURCE_EPOCH = 30
TOTAL_EPOCHS = 35
CONTROL_NAME = "LOCAL_EXP035_E30_TO_E35_ADAM_LR1E3_SEED430"
CANDIDATE_NAME = "LOCAL_EXP035_E30_TO_E35_ADAM_LR3E4_SEED430"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(
    *,
    python_command,
    gpu_num,
    net_name,
    learning_rate,
    checkpoint,
    train_dir,
    val_dir,
):
    return [
        python_command,
        "train.py",
        "--GPU-NUM",
        str(gpu_num),
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
        "--resume-lr",
        str(learning_rate),
        "--retain-val-epochs",
    ]


def build_plan(
    checkpoint,
    *,
    result_root=Path("../result"),
    train_dir=Path("/root/Data/train"),
    val_dir=Path("/root/Data/val"),
    gpu_num=0,
    python_command="python",
    expected_generation=SOURCE_GENERATION,
    expected_sha256=SOURCE_SHA256,
    require_data=False,
):
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

    result_root = Path(result_root)
    for name in (CONTROL_NAME, CANDIDATE_NAME):
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
    for role, name, learning_rate in (
        ("fixed_lr_control", CONTROL_NAME, 0.001),
        ("lower_lr_candidate", CANDIDATE_NAME, 0.0003),
    ):
        command = _command(
            python_command=python_command,
            gpu_num=gpu_num,
            net_name=name,
            learning_rate=learning_rate,
            checkpoint=checkpoint.resolve(),
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
            "sha256": actual_sha256,
            "checkpoint": str(checkpoint.resolve()),
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
    parser.add_argument("--result-root", type=Path, default=Path("../result"))
    parser.add_argument("--train-dir", type=Path, default=Path("/root/Data/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("/root/Data/val"))
    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument("--python-command", default="python")
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
            result_root=args.result_root,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            gpu_num=args.gpu_num,
            python_command=args.python_command,
            require_data=args.require_data,
        )
    except ValueError as exc:
        raise SystemExit(f"PRECHECK FAILED: {exc}") from exc
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
