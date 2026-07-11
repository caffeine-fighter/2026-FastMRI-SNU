#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate validation reconstructions with official 2026 FastMRI metric helpers.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name under ../result/<exp-name>.")
    parser.add_argument("--target-dir", type=Path, default=Path("/root/Data/val/image"))
    parser.add_argument("--recon-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--target-key", type=str, default="image_label")
    parser.add_argument("--recon-key", type=str, default="reconstruction")
    parser.add_argument("--max-key", type=str, default="max")
    return parser.parse_args()

def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)

def load_runtime_deps():
    try:
        import h5py
        import numpy as np
        import torch
        from utils.common.metrics import SSIM, foreground_mask, ssim_full, ssim_bbox
    except Exception as exc:
        fail(
            "Could not import required evaluation dependencies. "
            "Run this on VESSL after installing requirements. "
            f"Original error: {exc}"
        )
    return h5py, np, torch, SSIM, foreground_mask, ssim_full, ssim_bbox

def first_existing_dataset(hf, preferred_keys):
    for key in preferred_keys:
        if key in hf:
            return key
    keys = list(hf.keys())
    if len(keys) == 1:
        return keys[0]
    raise KeyError(f"none of preferred keys {preferred_keys} found; available keys={keys}")

def to_float(value, default_value):
    try:
        if hasattr(value, "shape"):
            value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return float(value)
    except Exception:
        return float(default_value)

def normalize_annotation_box(obj):
    if not isinstance(obj, dict):
        return None

    if all(k in obj for k in ["x", "y", "width", "height"]):
        try:
            return {
                "x": int(round(float(obj["x"]))),
                "y": int(round(float(obj["y"]))),
                "width": int(round(float(obj["width"]))),
                "height": int(round(float(obj["height"]))),
                "label": obj.get("label", obj.get("class", "")),
            }
        except Exception:
            return None

    for key in ["bbox", "box"]:
        if key in obj and isinstance(obj[key], (list, tuple)) and len(obj[key]) >= 4:
            try:
                x, y, w, h = obj[key][:4]
                return {
                    "x": int(round(float(x))),
                    "y": int(round(float(y))),
                    "width": int(round(float(w))),
                    "height": int(round(float(h))),
                    "label": obj.get("label", obj.get("class", "")),
                }
            except Exception:
                return None

    return None

def slice_index_from_obj(obj):
    if not isinstance(obj, dict):
        return None
    for key in ["slice", "slice_idx", "slice_index", "slice_num", "slice_id", "z", "index"]:
        if key in obj:
            try:
                return int(obj[key])
            except Exception:
                return None
    return None

def parse_annotations(raw, num_slices):
    boxes_by_slice = defaultdict(list)
    skipped_unassigned = 0

    if raw is None:
        return boxes_by_slice, 0, skipped_unassigned

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    if raw == "" or raw == "null":
        return boxes_by_slice, 0, skipped_unassigned

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return boxes_by_slice, 0, skipped_unassigned

    def visit(node, slice_hint=None, root_list=False):
        nonlocal skipped_unassigned

        box = normalize_annotation_box(node)
        if box is not None:
            explicit_slice = slice_index_from_obj(node)
            slice_idx = explicit_slice if explicit_slice is not None else slice_hint
            if slice_idx is None or slice_idx < 0 or slice_idx >= num_slices:
                skipped_unassigned += 1
                return
            boxes_by_slice[int(slice_idx)].append(box)
            return

        if isinstance(node, dict):
            explicit_slice = slice_index_from_obj(node)
            next_hint = explicit_slice if explicit_slice is not None else slice_hint

            for key, value in node.items():
                key_hint = next_hint
                if isinstance(key, str) and key.isdigit():
                    key_hint = int(key)
                visit(value, key_hint, root_list=False)

        elif isinstance(node, list):
            treat_index_as_slice = root_list and len(node) == num_slices
            for idx, value in enumerate(node):
                next_hint = idx if treat_index_as_slice else slice_hint
                visit(value, next_hint, root_list=False)

    visit(data, slice_hint=None, root_list=isinstance(data, list))

    total = sum(len(v) for v in boxes_by_slice.values())
    return boxes_by_slice, total, skipped_unassigned

def mean_or_none(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return float(sum(values) / len(values))

def scope_row(scope, full_values, bbox_values, volumes, slices, bbox_annotations):
    return {
        "scope": scope,
        "ssim_full_mean": mean_or_none(full_values),
        "ssim_bbox_mean": mean_or_none(bbox_values),
        "ssim_full_count": len(full_values),
        "ssim_bbox_count": len(bbox_values),
        "volumes": volumes,
        "slices": slices,
        "bbox_annotations": bbox_annotations,
    }

def leaderboard_equal_acc_row(rows):
    """Match recon_eval.py by averaging acc4 and acc8 before metrics."""
    by_scope = {row["scope"]: row for row in rows}
    if "acc4" not in by_scope or "acc8" not in by_scope:
        raise ValueError("Leaderboard aggregation requires acc4 and acc8 rows")
    selected = [by_scope["acc4"], by_scope["acc8"]]
    metrics = [
        row[key]
        for row in selected
        for key in ("ssim_full_mean", "ssim_bbox_mean")
    ]
    if any(value is None or not math.isfinite(value) for value in metrics):
        raise ValueError("Leaderboard aggregation requires finite acc4 and acc8 metrics")
    full = sum(row["ssim_full_mean"] for row in selected) / 2
    bbox = sum(row["ssim_bbox_mean"] for row in selected) / 2
    return {
        "scope": "leaderboard_equal_acc",
        "ssim_full_mean": full,
        "ssim_bbox_mean": bbox,
        "quality_score": (full + bbox) / 2,
        "aggregation": "equal mean of acc4 and acc8",
        "ssim_full_count": sum(row["ssim_full_count"] for row in selected),
        "ssim_bbox_count": sum(row["ssim_bbox_count"] for row in selected),
        "volumes": sum(row["volumes"] for row in selected),
        "slices": sum(row["slices"] for row in selected),
        "bbox_annotations": sum(row["bbox_annotations"] for row in selected),
    }


def infer_acc_name(path):
    name = path.name.lower()
    if "acc4" in name:
        return "acc4"
    if "acc8" in name:
        return "acc8"
    return "unknown"

def main():
    args = parse_args()

    if args.exp_name:
        if args.recon_dir is None:
            args.recon_dir = Path("../result") / args.exp_name / "reconstructions_val"
        if args.out_dir is None:
            args.out_dir = Path("../result") / args.exp_name / "metrics"

    if args.recon_dir is None:
        fail("Provide --recon-dir or --exp-name.")
    if args.out_dir is None:
        fail("Provide --out-dir or --exp-name.")
    if not args.target_dir.exists():
        fail(f"target-dir does not exist: {args.target_dir}")
    if not args.recon_dir.exists():
        fail(f"recon-dir does not exist: {args.recon_dir}")

    h5py, np, torch, SSIM, foreground_mask, ssim_full, ssim_bbox = load_runtime_deps()

    target_files = sorted(args.target_dir.glob("*.h5"))
    if not target_files:
        fail(f"no .h5 files found under target-dir: {args.target_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ssim = SSIM()
    values = {
        "overall": {"full": [], "bbox": [], "volumes": 0, "slices": 0, "bbox_annotations": 0},
        "acc4": {"full": [], "bbox": [], "volumes": 0, "slices": 0, "bbox_annotations": 0},
        "acc8": {"full": [], "bbox": [], "volumes": 0, "slices": 0, "bbox_annotations": 0},
        "unknown": {"full": [], "bbox": [], "volumes": 0, "slices": 0, "bbox_annotations": 0},
    }
    skipped = []

    for target_path in target_files:
        recon_path = args.recon_dir / target_path.name
        acc = infer_acc_name(target_path)

        if not recon_path.exists():
            skipped.append({"file": target_path.name, "reason": "missing reconstruction file"})
            continue

        try:
            with h5py.File(target_path, "r") as target_hf, h5py.File(recon_path, "r") as recon_hf:
                target_key = first_existing_dataset(target_hf, [args.target_key, "target", "image", "reconstruction"])
                recon_key = first_existing_dataset(recon_hf, [args.recon_key, "image", "target"])

                target = np.asarray(target_hf[target_key])
                recon = np.asarray(recon_hf[recon_key])

                if np.iscomplexobj(target):
                    target = np.abs(target)
                if np.iscomplexobj(recon):
                    recon = np.abs(recon)

                if target.ndim == 2:
                    target = target[None, ...]
                if recon.ndim == 2:
                    recon = recon[None, ...]

                if target.shape != recon.shape:
                    skipped.append({
                        "file": target_path.name,
                        "reason": f"shape mismatch target={target.shape} recon={recon.shape}",
                    })
                    continue

                data_range_default = float(np.max(target)) if target.size else 1.0
                data_range = to_float(target_hf.attrs.get(args.max_key, data_range_default), data_range_default)

                raw_annotations = target_hf.attrs.get("annotations", None)
                boxes_by_slice, box_count, skipped_unassigned = parse_annotations(raw_annotations, target.shape[0])
                if skipped_unassigned:
                    skipped.append({
                        "file": target_path.name,
                        "reason": f"{skipped_unassigned} annotations skipped because slice index could not be inferred",
                    })

                values["overall"]["volumes"] += 1
                values[acc]["volumes"] += 1
                values["overall"]["slices"] += int(target.shape[0])
                values[acc]["slices"] += int(target.shape[0])
                values["overall"]["bbox_annotations"] += int(box_count)
                values[acc]["bbox_annotations"] += int(box_count)

                for slice_idx in range(target.shape[0]):
                    target_slice = target[slice_idx].astype("float32")
                    recon_slice = recon[slice_idx].astype("float32")

                    target_t = torch.from_numpy(target_slice)
                    recon_t = torch.from_numpy(recon_slice)
                    mask_np = foreground_mask(target_slice)
                    mask_t = torch.from_numpy(mask_np.astype("float32"))

                    full_value = ssim_full(ssim, recon_t, target_t, mask_t, data_range)
                    if full_value is not None:
                        values["overall"]["full"].append(float(full_value))
                        values[acc]["full"].append(float(full_value))

                    for box in boxes_by_slice.get(slice_idx, []):
                        bbox_value = ssim_bbox(ssim, recon_t, target_t, box, data_range)
                        if bbox_value is not None:
                            values["overall"]["bbox"].append(float(bbox_value))
                            values[acc]["bbox"].append(float(bbox_value))

        except Exception as exc:
            skipped.append({"file": target_path.name, "reason": f"exception: {exc}"})
            continue

    rows = []
    for scope in ["overall", "acc4", "acc8", "unknown"]:
        v = values[scope]
        rows.append(scope_row(scope, v["full"], v["bbox"], v["volumes"], v["slices"], v["bbox_annotations"]))
    rows.insert(3, leaderboard_equal_acc_row(rows))

    summary = {
        "target_dir": str(args.target_dir),
        "recon_dir": str(args.recon_dir),
        "out_dir": str(args.out_dir),
        "rows": rows,
        "skipped": skipped,
    }

    json_path = args.out_dir / "metrics.json"
    csv_path = args.out_dir / "metrics.csv"
    skipped_path = args.out_dir / "skipped.json"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "ssim_full_mean",
                "ssim_bbox_mean",
                "quality_score",
                "aggregation",
                "ssim_full_count",
                "ssim_bbox_count",
                "volumes",
                "slices",
                "bbox_annotations",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {skipped_path}")

if __name__ == "__main__":
    main()
