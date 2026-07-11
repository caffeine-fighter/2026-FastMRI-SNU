#!/usr/bin/env python3
"""Rank retained validation epochs without changing checkpoint aliases."""

import argparse
import csv
import errno
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_VAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_val.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODEL_ROOT = REPO_ROOT / "utils" / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from utils.learning.resume import (  # noqa: E402
    _load_checkpoint_from_handle,
    _open_checkpoint_directory,
    _open_regular_at,
    validate_training_checkpoint,
)
from utils.learning.train_part import (  # noqa: E402
    PublicationIndeterminateError,
    _cleanup_staged_directory,
    _create_staged_directory,
    _publish_staged_directory_no_replace,
    _seal_staged_directory,
    _staged_directory_descriptor_path,
)


_GENERATION_MODEL_RE = re.compile(
    r"\.checkpoint-generation-([0-9a-f]{32})-model\.pt"
)
_EPOCH_DIR_RE = re.compile(r"epoch_([0-9]{4,})")
_EXPECTED_SCOPES = [
    "overall",
    "acc4",
    "acc8",
    "leaderboard_equal_acc",
    "unknown",
]
_SCOPE_SCHEMA = {
    "scope",
    "ssim_full_mean",
    "ssim_bbox_mean",
    "ssim_full_count",
    "ssim_bbox_count",
    "volumes",
    "slices",
    "bbox_annotations",
}
_EQUAL_ACC_SCHEMA = _SCOPE_SCHEMA | {"quality_score", "aggregation"}


def positive_epoch(value):
    epoch = int(value)
    if epoch <= 0:
        raise argparse.ArgumentTypeError("must be a positive checkpoint-state epoch")
    return epoch


def nonnegative_count(value):
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return count


def positive_count(value):
    count = nonnegative_count(value)
    if count == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-epochs",
        type=positive_epoch,
        nargs="+",
        required=True,
        help="Complete set of 1-based checkpoint-state epochs to sweep",
    )
    parser.add_argument("--expected-volumes", type=positive_count, required=True)
    parser.add_argument("--expected-slices", type=positive_count, required=True)
    parser.add_argument("--expected-boxes", type=nonnegative_count, required=True)
    return parser.parse_args()


def epoch_directory(retained_root, epoch):
    return Path(retained_root) / f"epoch_{epoch:04d}"


def _regular_h5_names(directory, description):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{description} is not a real directory: {directory}")
    names = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.endswith(".h5") or not entry.is_file(
                follow_symlinks=False
            ):
                raise ValueError(
                    f"{description} entry is not a regular .h5 file: "
                    f"{directory / entry.name}"
                )
            names.add(entry.name)
    return names


def validate_retained_coverage(retained_root, target_dir, expected_epochs):
    """Require exact target-file coverage in every deterministic epoch directory."""
    retained_root = Path(retained_root)
    target_dir = Path(target_dir)
    target_files = _regular_h5_names(target_dir, "Target directory")
    if not target_files:
        raise ValueError(f"No target H5 files found in {target_dir}")
    if not retained_root.is_dir():
        raise ValueError(f"Retained root is not a directory: {retained_root}")
    expected_epoch_names = {
        f"epoch_{epoch:04d}" for epoch in expected_epochs
    }
    discovered_epoch_names = {
        entry.name
        for entry in os.scandir(retained_root)
        if entry.name.startswith("epoch_")
    }
    unexpected_epoch_names = sorted(
        discovered_epoch_names - expected_epoch_names
    )
    if unexpected_epoch_names:
        raise ValueError(
            "Retained root contains unexpected epoch directories: "
            f"{unexpected_epoch_names}"
        )

    epoch_dirs = {}
    for epoch in expected_epochs:
        directory = epoch_directory(retained_root, epoch)
        if not directory.is_dir():
            raise ValueError(f"Missing retained epoch directory: {directory}")
        reconstruction_files = _regular_h5_names(
            directory, f"Retained epoch {epoch} directory"
        )
        missing = sorted(target_files - reconstruction_files)
        unexpected = sorted(reconstruction_files - target_files)
        if missing or unexpected:
            raise ValueError(
                f"Epoch {epoch} coverage mismatch: missing={missing}; "
                f"unexpected={unexpected}"
            )
        epoch_dirs[epoch] = directory
    return epoch_dirs


def _finite_number(value, description):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{description} must be finite")
    return value


def _exact_nonnegative_integer(value, description):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return value


def _validate_strict_metrics(
    summary,
    target_dir,
    recon_dir,
    out_dir,
    expected_volumes,
    expected_slices,
    expected_boxes,
):
    expected_summary_schema = {"target_dir", "recon_dir", "out_dir", "rows", "skipped"}
    if not isinstance(summary, dict) or set(summary) != expected_summary_schema:
        raise ValueError("Evaluator metrics summary schema is not exact")
    expected_paths = {
        "target_dir": str(target_dir),
        "recon_dir": str(recon_dir),
        "out_dir": str(out_dir),
    }
    for key, expected in expected_paths.items():
        if summary[key] != expected:
            raise ValueError(f"Evaluator metrics {key} does not match invocation")
    if summary["skipped"] != []:
        raise ValueError(
            f"Epoch evaluation requires skips=0; evaluator reported {summary['skipped']!r}"
        )
    rows = summary["rows"]
    if not isinstance(rows, list) or [
        row.get("scope") if isinstance(row, dict) else None for row in rows
    ] != _EXPECTED_SCOPES:
        raise ValueError(
            f"Evaluator row scopes must be exactly {_EXPECTED_SCOPES!r}"
        )

    rows_by_scope = {}
    count_keys = (
        "ssim_full_count",
        "ssim_bbox_count",
        "volumes",
        "slices",
        "bbox_annotations",
    )
    for row in rows:
        scope = row["scope"]
        expected_schema = (
            _EQUAL_ACC_SCHEMA if scope == "leaderboard_equal_acc" else _SCOPE_SCHEMA
        )
        if set(row) != expected_schema:
            raise ValueError(f"Evaluator {scope} row schema is not exact")
        for key in count_keys:
            _exact_nonnegative_integer(row[key], f"Evaluator {scope} {key}")
        rows_by_scope[scope] = row

    overall = rows_by_scope["overall"]
    expected_overall_counts = {
        "ssim_full_count": expected_slices,
        "ssim_bbox_count": expected_boxes,
        "volumes": expected_volumes,
        "slices": expected_slices,
        "bbox_annotations": expected_boxes,
    }
    for key, expected in expected_overall_counts.items():
        if overall[key] != expected:
            raise ValueError(
                f"Evaluator overall {key} mismatch: expected={expected} actual={overall[key]}"
            )
    for key in ("ssim_full_mean", "ssim_bbox_mean"):
        _finite_number(overall[key], f"Evaluator overall {key}")

    for scope in ("acc4", "acc8"):
        row = rows_by_scope[scope]
        if row["volumes"] <= 0 or row["slices"] <= 0:
            raise ValueError(f"Evaluator {scope} row must cover volumes and slices")
        if row["ssim_full_count"] != row["slices"]:
            raise ValueError(f"Evaluator {scope} full count must equal slices")
        if row["ssim_bbox_count"] != row["bbox_annotations"]:
            raise ValueError(f"Evaluator {scope} bbox count must equal annotations")
        for key in ("ssim_full_mean", "ssim_bbox_mean"):
            _finite_number(row[key], f"Evaluator {scope} {key}")
    for key, expected in expected_overall_counts.items():
        actual = rows_by_scope["acc4"][key] + rows_by_scope["acc8"][key]
        if actual != expected:
            raise ValueError(
                f"Evaluator acceleration {key} total mismatch: "
                f"expected={expected} actual={actual}"
            )

    unknown = rows_by_scope["unknown"]
    if any(unknown[key] != 0 for key in count_keys) or any(
        unknown[key] is not None for key in ("ssim_full_mean", "ssim_bbox_mean")
    ):
        raise ValueError("Evaluator unknown row must be exactly empty")

    equal_acc = rows_by_scope["leaderboard_equal_acc"]
    for key, expected in expected_overall_counts.items():
        if equal_acc[key] != expected:
            raise ValueError(
                f"Evaluator equal-acc {key} mismatch: "
                f"expected={expected} actual={equal_acc[key]}"
            )
    if equal_acc["aggregation"] != "equal mean of acc4 and acc8":
        raise ValueError("Evaluator equal-acc aggregation is not exact")
    for key in ("ssim_full_mean", "ssim_bbox_mean", "quality_score"):
        _finite_number(equal_acc[key], f"Evaluator equal-acc {key}")
    for key in ("ssim_full_mean", "ssim_bbox_mean"):
        expected_mean = (
            rows_by_scope["acc4"][key] + rows_by_scope["acc8"][key]
        ) / 2
        if not math.isclose(
            equal_acc[key], expected_mean, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"Evaluator equal-acc {key} formula mismatch")
    expected_quality = (
        equal_acc["ssim_full_mean"] + equal_acc["ssim_bbox_mean"]
    ) / 2
    if not math.isclose(
        equal_acc["quality_score"], expected_quality, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("Evaluator equal-acc quality_score formula mismatch")
    return equal_acc


def evaluate_epoch(
    target_dir,
    recon_dir,
    out_dir,
    expected_volumes,
    expected_slices,
    expected_boxes,
    runner=subprocess.run,
):
    """Run evaluate_val.py strictly on CPU and return its exact equal-acc row."""
    target_dir = Path(target_dir).resolve()
    recon_dir = Path(recon_dir).resolve()
    out_dir = Path(out_dir).resolve(strict=False)
    if os.path.lexists(out_dir):
        raise ValueError(f"Evaluator output directory already exists: {out_dir}")
    staged = _create_staged_directory(out_dir, "Evaluator")
    staging_out_dir = _staged_directory_descriptor_path(staged)
    command = [
        sys.executable,
        str(EVALUATE_VAL_SCRIPT),
        "--target-dir",
        str(target_dir),
        "--recon-dir",
        str(recon_dir),
        "--out-dir",
        str(staging_out_dir),
        "--require-complete",
        "--expected-volumes",
        str(expected_volumes),
        "--expected-slices",
        str(expected_slices),
        "--expected-boxes",
        str(expected_boxes),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        runner(
            command,
            check=True,
            cwd=REPO_ROOT,
            env=environment,
            pass_fds=(staged.directory_fd,),
        )
        metrics_path = staging_out_dir / "metrics.json"
        try:
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Evaluator did not emit valid metrics: {metrics_path}"
            ) from exc
        equal_acc = _validate_strict_metrics(
            summary,
            target_dir,
            recon_dir,
            staging_out_dir,
            expected_volumes,
            expected_slices,
            expected_boxes,
        )
        _seal_staged_directory(staged)
        _publish_staged_directory_no_replace(staged, out_dir, "Evaluator")
        return equal_acc
    finally:
        _cleanup_staged_directory(staged)


def map_checkpoint_generations(checkpoint_dir, expected_epochs):
    """Return the unique immutable model generation for every expected state epoch."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    directory_fd = _open_checkpoint_directory(checkpoint_dir)
    by_epoch = {}
    try:
        names = sorted(path.name for path in checkpoint_dir.iterdir())
        for name in names:
            match = _GENERATION_MODEL_RE.fullmatch(name)
            if match is None:
                continue
            with _open_regular_at(
                directory_fd, name, "immutable checkpoint generation"
            ) as handle:
                state = _load_checkpoint_from_handle(handle)
            validate_training_checkpoint(state)
            epoch = state["epoch"]
            by_epoch.setdefault(epoch, []).append(
                {
                    "generation": match.group(1),
                    "artifact": str(checkpoint_dir / name),
                }
            )
    finally:
        os.close(directory_fd)

    mapped = {}
    for epoch in expected_epochs:
        matches = by_epoch.get(epoch, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one immutable checkpoint generation for epoch "
                f"{epoch}, found {len(matches)}"
            )
        mapped[epoch] = matches[0]
    return mapped


def _normalized_expected_epochs(expected_epochs):
    epochs = list(expected_epochs)
    if not epochs:
        raise ValueError("At least one expected epoch is required")
    if any(isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0 for epoch in epochs):
        raise ValueError("Expected epochs must be positive integers")
    if len(set(epochs)) != len(epochs):
        raise ValueError("Expected epochs must not contain duplicates")
    return sorted(epochs)


def _write_ranked_outputs(out_dir, rankings):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "selection_metric": "leaderboard_equal_acc.quality_score",
        "selection_rule": "descending quality score, then ascending epoch",
        "best_model_rewritten": False,
        "selected": rankings[0],
        "rankings": rankings,
    }
    (out_dir / "val_epoch_sweep.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fieldnames = [
        "rank",
        "epoch",
        "quality_score",
        "ssim_full_mean",
        "ssim_bbox_mean",
        "checkpoint_generation",
        "checkpoint_artifact",
        "reconstruction_dir",
        "metrics_json",
    ]
    with (out_dir / "val_epoch_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rankings)

    report_lines = [
        "# Validation epoch equal-acc sweep",
        "",
        "Selection metric: `leaderboard_equal_acc.quality_score` ",
        "(descending; epoch ascending breaks ties).",
        "",
        f"Selected epoch: {rankings[0]['epoch']}",
        f"Checkpoint generation: `{rankings[0]['checkpoint_generation']}`",
        "",
        "`best_model.pt` was not rewritten.",
        "",
        "| Rank | Epoch | Quality | Full SSIM | BBox SSIM | Generation |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rankings:
        report_lines.append(
            f"| {row['rank']} | {row['epoch']} | {row['quality_score']:.10g} | "
            f"{row['ssim_full_mean']:.10g} | {row['ssim_bbox_mean']:.10g} | "
            f"`{row['checkpoint_generation']}` |"
        )
    (out_dir / "val_epoch_sweep_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def run_sweep(
    retained_root,
    checkpoint_dir,
    target_dir,
    out_dir,
    expected_epochs,
    expected_volumes,
    expected_slices,
    expected_boxes,
    runner=subprocess.run,
):
    """Evaluate, map, rank, and report a fail-closed validation epoch sweep."""
    retained_root = Path(retained_root).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    target_dir = Path(target_dir).resolve()
    out_dir = Path(out_dir).resolve(strict=False)
    if os.path.lexists(out_dir):
        raise FileExistsError(
            errno.EEXIST,
            f"Sweep output already exists: {out_dir}",
            out_dir,
        )
    staged = _create_staged_directory(out_dir, "Sweep")
    staging_dir = _staged_directory_descriptor_path(staged)
    try:
        epochs = _normalized_expected_epochs(expected_epochs)
        epoch_dirs = validate_retained_coverage(
            retained_root, target_dir, epochs
        )
        checkpoints = map_checkpoint_generations(checkpoint_dir, epochs)

        rows = []
        for epoch in epochs:
            relative_metrics_dir = (
                Path("epoch_metrics") / f"epoch_{epoch:04d}"
            )
            metrics_dir = staging_dir / relative_metrics_dir
            equal_acc = evaluate_epoch(
                target_dir,
                epoch_dirs[epoch],
                metrics_dir,
                expected_volumes,
                expected_slices,
                expected_boxes,
                runner=runner,
            )
            for key in ("ssim_full_mean", "ssim_bbox_mean"):
                value = equal_acc.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"Evaluator equal-acc {key} must be finite"
                    )
            rows.append(
                {
                    "epoch": epoch,
                    "quality_score": equal_acc["quality_score"],
                    "ssim_full_mean": equal_acc["ssim_full_mean"],
                    "ssim_bbox_mean": equal_acc["ssim_bbox_mean"],
                    "checkpoint_generation": checkpoints[epoch]["generation"],
                    "checkpoint_artifact": checkpoints[epoch]["artifact"],
                    "reconstruction_dir": str(epoch_dirs[epoch]),
                    "metrics_json": str(
                        out_dir / relative_metrics_dir / "metrics.json"
                    ),
                }
            )

        rows.sort(key=lambda row: (-row["quality_score"], row["epoch"]))
        rankings = [
            dict(row, rank=index) for index, row in enumerate(rows, start=1)
        ]
        _write_ranked_outputs(staging_dir, rankings)
        _seal_staged_directory(staged)
        _publish_staged_directory_no_replace(staged, out_dir, "Sweep")
        return rankings
    finally:
        _cleanup_staged_directory(staged)


def main():
    args = parse_args()
    try:
        rankings = run_sweep(
            retained_root=args.retained_root,
            checkpoint_dir=args.checkpoint_dir,
            target_dir=args.target_dir,
            out_dir=args.out_dir,
            expected_epochs=args.expected_epochs,
            expected_volumes=args.expected_volumes,
            expected_slices=args.expected_slices,
            expected_boxes=args.expected_boxes,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"Selected epoch {rankings[0]['epoch']} generation "
        f"{rankings[0]['checkpoint_generation']}"
    )
    print(f"Saved ranked sweep: {Path(args.out_dir) / 'val_epoch_sweep.json'}")


if __name__ == "__main__":
    main()
