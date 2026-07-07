#!/usr/bin/env python3
import argparse
import csv
import socket
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = [
    "exp_id",
    "date",
    "machine",
    "branch",
    "commit",
    "config",
    "seed",
    "command",
    "status",
    "val_loss",
    "ssim_full",
    "ssim_bbox",
    "ssim_full_acc4",
    "ssim_bbox_acc4",
    "ssim_full_acc8",
    "ssim_bbox_acc8",
    "checkpoint_path",
    "notes",
]

METRIC_COLUMNS = {
    "ssim_full": ("overall", "ssim_full_mean"),
    "ssim_bbox": ("overall", "ssim_bbox_mean"),
    "ssim_full_acc4": ("acc4", "ssim_full_mean"),
    "ssim_bbox_acc4": ("acc4", "ssim_bbox_mean"),
    "ssim_full_acc8": ("acc8", "ssim_full_mean"),
    "ssim_bbox_acc8": ("acc8", "ssim_bbox_mean"),
}


def parse_args():
    examples = """
    Examples:
      python scripts/record_experiment.py --exp-id EXP001 --exp-name EXP001_baseline_varnet_c1_ch9_s4_e5 --status done --seed 430 --config none --command "python train.py -b 1 -e 5 -n EXP001_baseline_varnet_c1_ch9_s4_e5 --seed 430" --notes "5 epoch baseline"

      python scripts/record_experiment.py --exp-id EXP002 --exp-name EXP002_trial --status planned --allow-missing-metrics --notes "Queued on VESSL"

      python scripts/record_experiment.py --exp-id EXP003 --exp-name EXP003_trial --result-dir ../result/EXP003_trial --log-path experiments/experiment_log.csv
    """
    parser = argparse.ArgumentParser(
        description="Record or update one experiment row in experiments/experiment_log.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(examples).strip(),
    )
    parser.add_argument("--exp-id", required=True, help="Stable experiment id, for example EXP001.")
    parser.add_argument("--exp-name", required=True, help="Experiment result directory name under ../result.")
    parser.add_argument("--status", default="", help="Experiment status, for example planned, running, done, failed.")
    parser.add_argument("--seed", default="", help="Random seed used for the run.")
    parser.add_argument("--command", default="", help="Training or reconstruction command to record.")
    parser.add_argument("--config", default="", help="Config name, file, or compact parameter summary.")
    parser.add_argument("--notes", default="", help="Free-form notes for this experiment.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Experiment result directory. Defaults to ../result/<exp-name>.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("experiments/experiment_log.csv"),
        help="Experiment log CSV path. Defaults to experiments/experiment_log.csv.",
    )
    parser.add_argument(
        "--allow-missing-metrics",
        action="store_true",
        help="Record blank SSIM fields instead of failing when metrics/metrics.csv is missing.",
    )
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message):
    print(f"WARNING: {message}", file=sys.stderr)


def as_csv_value(value):
    if value is None:
        return ""
    return str(value)


def run_git(args):
    safe_root = str(REPO_ROOT).replace("\\", "/")
    cmd = ["git", "-c", f"safe.directory={safe_root}", "-C", str(REPO_ROOT), *args]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def current_branch():
    branch = run_git(["branch", "--show-current"])
    if branch:
        return branch
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return "" if branch == "HEAD" else branch


def current_commit():
    return run_git(["rev-parse", "--short", "HEAD"])


def read_metrics(metrics_path, allow_missing):
    values = {column: "" for column in METRIC_COLUMNS}

    if not metrics_path.exists():
        if allow_missing:
            warn(f"metrics.csv not found; recording blank SSIM fields: {metrics_path}")
            return values
        fail(
            "metrics.csv not found: "
            f"{metrics_path}\nRun scripts/evaluate_val.py first or pass --allow-missing-metrics."
        )

    with metrics_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            fail(f"metrics.csv is empty: {metrics_path}")
        rows = list(reader)

    if "scope" not in reader.fieldnames:
        fail(f"metrics.csv is missing required column 'scope': {metrics_path}")

    by_scope = {row.get("scope", ""): row for row in rows}
    for output_column, (scope, source_column) in METRIC_COLUMNS.items():
        if source_column not in reader.fieldnames:
            fail(f"metrics.csv is missing required column '{source_column}': {metrics_path}")
        if scope not in by_scope:
            warn(f"metrics.csv has no '{scope}' row; recording blank value for {output_column}")
            continue
        values[output_column] = as_csv_value(by_scope[scope].get(source_column, ""))

    return values


def read_best_val_loss(loss_path):
    if not loss_path.exists():
        warn(f"val_loss_log.npy not found; recording blank val_loss: {loss_path}")
        return ""

    try:
        import numpy as np
    except Exception as exc:
        fail(f"could not import numpy to read validation loss: {exc}")

    try:
        log = np.load(loss_path)
    except Exception as exc:
        fail(f"could not read validation loss log {loss_path}: {exc}")

    if log.ndim != 2 or log.shape[1] < 2:
        fail(f"expected val_loss_log.npy shape (N, 2), got {log.shape}: {loss_path}")
    if log.shape[0] == 0:
        warn(f"val_loss_log.npy is empty; recording blank val_loss: {loss_path}")
        return ""

    losses = log[:, 1]
    try:
        best_idx = int(np.nanargmin(losses))
    except ValueError:
        warn(f"val_loss_log.npy contains no finite losses; recording blank val_loss: {loss_path}")
        return ""
    return str(float(losses[best_idx]))


def read_existing_log(log_path):
    if not log_path.exists():
        return []

    with log_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [{field: row.get(field, "") for field in FIELDNAMES} for row in reader]


def write_log(log_path, rows):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def upsert_row(rows, new_row):
    found = False
    updated = []
    for row in rows:
        if row.get("exp_id") == new_row["exp_id"]:
            updated.append(new_row)
            found = True
        else:
            updated.append(row)
    if not found:
        updated.append(new_row)
    return updated, found


def main():
    args = parse_args()
    result_dir = args.result_dir if args.result_dir is not None else Path("../result") / args.exp_name

    metrics_path = result_dir / "metrics" / "metrics.csv"
    loss_path = result_dir / "val_loss_log.npy"
    checkpoint_path = result_dir / "checkpoints" / "best_model.pt"

    metric_values = read_metrics(metrics_path, args.allow_missing_metrics)
    val_loss = read_best_val_loss(loss_path)

    row = {
        "exp_id": args.exp_id,
        "date": date.today().isoformat(),
        "machine": socket.gethostname(),
        "branch": current_branch(),
        "commit": current_commit(),
        "config": args.config,
        "seed": args.seed,
        "command": args.command,
        "status": args.status,
        "val_loss": val_loss,
        "ssim_full": metric_values["ssim_full"],
        "ssim_bbox": metric_values["ssim_bbox"],
        "ssim_full_acc4": metric_values["ssim_full_acc4"],
        "ssim_bbox_acc4": metric_values["ssim_bbox_acc4"],
        "ssim_full_acc8": metric_values["ssim_full_acc8"],
        "ssim_bbox_acc8": metric_values["ssim_bbox_acc8"],
        "checkpoint_path": checkpoint_path.as_posix(),
        "notes": args.notes,
    }

    rows = read_existing_log(args.log_path)
    rows, updated_existing = upsert_row(rows, row)
    write_log(args.log_path, rows)

    action = "updated" if updated_existing else "appended"
    print(f"{action}: {args.exp_id}")
    print(f"log: {args.log_path}")


if __name__ == "__main__":
    main()
