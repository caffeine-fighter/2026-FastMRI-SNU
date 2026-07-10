#!/usr/bin/env python3
"""Build a fail-closed EXP031 versus LOCAL candidate decision report.

A final report is written only when:
- EXP031 exists in the official experiment log;
- a provenance-rich EXP031 validation handoff JSON exists and agrees with it;
- every required LOCAL metric CSV/JSON, skipped sidecar, loss array, and run
  context exists and passes exact configuration/count checks.

This tool never loads model weights and never invokes training, reconstruction,
or evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]


@dataclass(frozen=True)
class Thresholds:
    min_e5_quality_gain: float = 0.0
    max_component_regression: float = 0.001
    min_seed431_quality_gain: float = 0.0
    max_e10_vs_e5_quality_drop: float = 0.001


@dataclass(frozen=True)
class LocalSpec:
    exp_id: str
    cascade: int
    chans: int
    sens_chans: int
    epochs: int
    seed: int


EXPECTED_COUNTS = {
    "overall": {"volumes": 30, "slices": 791, "bbox_annotations": 161},
    "acc4": {"volumes": 15, "slices": 407, "bbox_annotations": 107},
    "acc8": {"volumes": 15, "slices": 384, "bbox_annotations": 54},
}
METRIC_KEYS = (
    "ssim_full",
    "ssim_bbox",
    "ssim_full_acc4",
    "ssim_bbox_acc4",
    "ssim_full_acc8",
    "ssim_bbox_acc8",
)
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOCAL_NAME_RE = re.compile(
    r"^LOCAL_EXP\d+_varnet_c(?P<cascade>\d+)_ch(?P<chans>\d+)_s(?P<sens>\d+)_e(?P<epochs>\d+)(?:_seed(?P<seed>\d+))?$"
)


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_json_constant,
    )


def rows_by_unique_key(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            raise ValueError(f"{label}: missing {key}")
        if value in indexed:
            raise ValueError(f"{label}: duplicate {key}={value!r}")
        indexed[value] = row
    return indexed


def quality(full: float, bbox: float) -> float:
    return 0.5 * full + 0.5 * bbox


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: non-finite value {value!r}")
    return number


def unit_metric(value: Any, label: str) -> float:
    number = finite_number(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label}: metric outside [0, 1]: {number}")
    return number


def float_or_none(value: str | None, label: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    return finite_number(value, label)


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(a - b) <= tolerance


def parse_official_log(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{path}: duplicate CSV header")
        rows = list(reader)
    parsed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=2):
        exp_id = row.get("exp_id", "").strip()
        if not exp_id:
            continue
        if exp_id in parsed:
            raise ValueError(f"{path}:{index}: duplicate exp_id={exp_id!r}")
        full = float_or_none(row.get("ssim_full"), f"{path}:{index}:ssim_full")
        bbox = float_or_none(row.get("ssim_bbox"), f"{path}:{index}:ssim_bbox")
        parsed[exp_id] = {
            "exp_id": exp_id,
            "exp_name": row.get("exp_name", "").strip(),
            "status": row.get("status", "").strip(),
            "cascade": int(row["cascade"]) if row.get("cascade") else None,
            "chans": int(row["chans"]) if row.get("chans") else None,
            "sens_chans": int(row["sens_chans"]) if row.get("sens_chans") else None,
            "epochs": int(row["epochs"]) if row.get("epochs") else None,
            "seed": int(row["seed"]) if row.get("seed") else None,
            "commit": row.get("commit", "").strip().lower(),
            "command": row.get("command", "").strip(),
            "checkpoint_path": row.get("checkpoint_path", "").strip(),
            "val_loss": float_or_none(row.get("val_loss"), f"{path}:{index}:val_loss"),
            "ssim_full": full,
            "ssim_bbox": bbox,
            "quality_score": quality(full, bbox) if full is not None and bbox is not None else None,
            "ssim_full_acc4": float_or_none(row.get("ssim_full_acc4"), f"{path}:{index}:ssim_full_acc4"),
            "ssim_bbox_acc4": float_or_none(row.get("ssim_bbox_acc4"), f"{path}:{index}:ssim_bbox_acc4"),
            "ssim_full_acc8": float_or_none(row.get("ssim_full_acc8"), f"{path}:{index}:ssim_full_acc8"),
            "ssim_bbox_acc8": float_or_none(row.get("ssim_bbox_acc8"), f"{path}:{index}:ssim_bbox_acc8"),
        }
    return parsed


def find_exp031(rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for exp_id, row in rows.items()
        if exp_id == "EXP031" or exp_id.startswith("EXP031_") or row["exp_name"].startswith("EXP031_")
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f"Expected one EXP031 official row, found {len(candidates)}")
    return candidates[0]


def validate_official_row(row: dict[str, Any], expected_epochs: int) -> list[str]:
    errors: list[str] = []
    prefix = row["exp_id"]
    if row["status"] != "done":
        errors.append(f"{prefix}: status is {row['status']!r}, expected 'done'")
    if row["epochs"] != expected_epochs:
        errors.append(f"{prefix}: epochs={row['epochs']}, expected {expected_epochs}")
    if (row["cascade"], row["chans"], row["sens_chans"]) != (4, 12, 8):
        errors.append(f"{prefix}: config must be (4, 12, 8)")
    if row["seed"] != 430:
        errors.append(f"{prefix}: seed={row['seed']}, expected 430")
    if not SHA_RE.fullmatch(row["commit"]):
        errors.append(f"{prefix}: missing or invalid commit provenance")
    if not row["checkpoint_path"]:
        errors.append(f"{prefix}: missing checkpoint_path")
    if not row["command"] or "train.py" not in row["command"]:
        errors.append(f"{prefix}: missing training command provenance")
    if row["val_loss"] is None or row["val_loss"] < 0:
        errors.append(f"{prefix}: missing or invalid val_loss")
    for key in METRIC_KEYS:
        value = row.get(key)
        if value is None:
            errors.append(f"{prefix}: missing {key}")
        else:
            try:
                unit_metric(value, f"{prefix}:{key}")
            except ValueError as exc:
                errors.append(str(exc))
    return errors


def parse_exp031_handoff(path: Path) -> dict[str, Any]:
    data = load_json_strict(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: handoff must be a JSON object")
    required = {
        "exp_id",
        "exp_name",
        "status",
        "branch",
        "commit",
        "command",
        "seed",
        "epochs",
        "best_epoch",
        "val_loss",
        "checkpoint_path",
        "checkpoint_sha256",
        "config",
        "metrics",
        "skipped",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if data["exp_id"] != "EXP031" or not str(data["exp_name"]).startswith("EXP031_"):
        raise ValueError(f"{path}: invalid EXP031 identity")
    if data["status"] != "done":
        raise ValueError(f"{path}: status must be 'done'")
    if not data["branch"]:
        raise ValueError(f"{path}: missing branch")
    commit = str(data["commit"]).lower()
    if not SHA_RE.fullmatch(commit):
        raise ValueError(f"{path}: invalid commit")
    if not data["command"] or "train.py" not in data["command"] or "EXP031" not in data["command"]:
        raise ValueError(f"{path}: command must identify train.py and EXP031")
    if int(data["seed"]) != 430 or int(data["epochs"]) != 30:
        raise ValueError(f"{path}: expected seed=430 and epochs=30")
    best_epoch = int(data["best_epoch"])
    if not 0 <= best_epoch < 30:
        raise ValueError(f"{path}: best_epoch outside [0, 29]")
    val_loss = finite_number(data["val_loss"], f"{path}:val_loss")
    if val_loss < 0:
        raise ValueError(f"{path}: negative val_loss")
    if "EXP031" not in str(data["checkpoint_path"]):
        raise ValueError(f"{path}: checkpoint_path does not identify EXP031")
    if not SHA256_RE.fullmatch(str(data["checkpoint_sha256"]).lower()):
        raise ValueError(f"{path}: invalid checkpoint_sha256")
    config = data["config"]
    if (int(config.get("cascade", -1)), int(config.get("chans", -1)), int(config.get("sens_chans", -1))) != (4, 12, 8):
        raise ValueError(f"{path}: config must be c4/ch12/s8")
    if data["skipped"] != []:
        raise ValueError(f"{path}: skipped must be []")

    parsed_metrics: dict[str, dict[str, Any]] = {}
    for scope, counts in EXPECTED_COUNTS.items():
        if scope not in data["metrics"]:
            raise ValueError(f"{path}: missing metrics scope {scope}")
        row = data["metrics"][scope]
        full = unit_metric(row.get("ssim_full"), f"{path}:{scope}:ssim_full")
        bbox = unit_metric(row.get("ssim_bbox"), f"{path}:{scope}:ssim_bbox")
        parsed = {"ssim_full": full, "ssim_bbox": bbox, "quality_score": quality(full, bbox)}
        for key, expected in counts.items():
            actual = int(row.get(key, -1))
            if actual != expected:
                raise ValueError(f"{path}:{scope}:{key}={actual}, expected {expected}")
            parsed[key] = actual
        parsed_metrics[scope] = parsed

    return {
        **data,
        "commit": commit,
        "checkpoint_sha256": str(data["checkpoint_sha256"]).lower(),
        "best_epoch": best_epoch,
        "val_loss": val_loss,
        "seed": int(data["seed"]),
        "epochs": int(data["epochs"]),
        "metrics": parsed_metrics,
    }


def cross_check_exp031(log_row: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    direct = ("exp_name", "status", "seed", "epochs", "checkpoint_path")
    for key in direct:
        if log_row[key] != handoff[key]:
            errors.append(f"EXP031 log/handoff mismatch for {key}: {log_row[key]!r} != {handoff[key]!r}")
    if not (log_row["commit"].startswith(handoff["commit"]) or handoff["commit"].startswith(log_row["commit"])):
        errors.append("EXP031 log/handoff commit mismatch")
    if log_row.get("val_loss") is None:
        errors.append("EXP031 log is missing val_loss")
    elif not close(float(log_row["val_loss"]), float(handoff["val_loss"])):
        errors.append("EXP031 log/handoff val_loss mismatch")
    mapping = {
        "ssim_full": ("overall", "ssim_full"),
        "ssim_bbox": ("overall", "ssim_bbox"),
        "ssim_full_acc4": ("acc4", "ssim_full"),
        "ssim_bbox_acc4": ("acc4", "ssim_bbox"),
        "ssim_full_acc8": ("acc8", "ssim_full"),
        "ssim_bbox_acc8": ("acc8", "ssim_bbox"),
    }
    for log_key, (scope, handoff_key) in mapping.items():
        if log_row.get(log_key) is None:
            errors.append(f"EXP031 log is missing {log_key}")
        elif not close(float(log_row[log_key]), float(handoff["metrics"][scope][handoff_key])):
            errors.append(f"EXP031 log/handoff metric mismatch for {log_key}")
    return errors


def validate_local_name(spec: LocalSpec) -> None:
    match = LOCAL_NAME_RE.fullmatch(spec.exp_id)
    if match is None:
        raise ValueError(f"{spec.exp_id}: name does not encode the expected configuration")
    encoded_seed = int(match.group("seed")) if match.group("seed") else 430
    encoded = (
        int(match.group("cascade")),
        int(match.group("chans")),
        int(match.group("sens")),
        int(match.group("epochs")),
        encoded_seed,
    )
    expected = (spec.cascade, spec.chans, spec.sens_chans, spec.epochs, spec.seed)
    if encoded != expected:
        raise ValueError(f"{spec.exp_id}: encoded config {encoded} != expected {expected}")


def parse_local_metrics(result_root: Path, spec: LocalSpec) -> dict[str, Any]:
    validate_local_name(spec)
    result_dir = result_root / spec.exp_id
    metrics_csv = result_dir / "metrics/metrics.csv"
    metrics_json = result_dir / "metrics/metrics.json"
    skipped_path = result_dir / "metrics/skipped.json"
    loss_path = result_dir / "val_loss_log.npy"
    context_path = result_dir / "logs/run_context.txt"
    missing = [
        str(path)
        for path in (metrics_csv, metrics_json, skipped_path, loss_path, context_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(";".join(missing))

    with metrics_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{spec.exp_id}: duplicate metrics.csv header")
        csv_rows = rows_by_unique_key(list(reader), "scope", f"{spec.exp_id}:metrics.csv")
    json_document = load_json_strict(metrics_json)
    if not isinstance(json_document, dict) or not isinstance(json_document.get("rows"), list):
        raise ValueError(f"{spec.exp_id}: metrics.json must contain an object with a rows list")
    if spec.exp_id not in str(json_document.get("recon_dir", "")) or spec.exp_id not in str(json_document.get("out_dir", "")):
        raise ValueError(f"{spec.exp_id}: metrics.json paths do not identify the expected experiment")
    json_rows = rows_by_unique_key(json_document["rows"], "scope", f"{spec.exp_id}:metrics.json")

    parsed_scopes: dict[str, dict[str, Any]] = {}
    for scope, counts in EXPECTED_COUNTS.items():
        if scope not in csv_rows or scope not in json_rows:
            raise ValueError(f"{spec.exp_id}: missing {scope} in CSV or JSON")
        csv_row = csv_rows[scope]
        json_row = json_rows[scope]
        full = unit_metric(csv_row["ssim_full_mean"], f"{spec.exp_id}:{scope}:ssim_full")
        bbox = unit_metric(csv_row["ssim_bbox_mean"], f"{spec.exp_id}:{scope}:ssim_bbox")
        if not close(full, unit_metric(json_row["ssim_full_mean"], f"{spec.exp_id}:{scope}:json_full")):
            raise ValueError(f"{spec.exp_id}:{scope}: CSV/JSON full mismatch")
        if not close(bbox, unit_metric(json_row["ssim_bbox_mean"], f"{spec.exp_id}:{scope}:json_bbox")):
            raise ValueError(f"{spec.exp_id}:{scope}: CSV/JSON bbox mismatch")
        parsed = {"ssim_full": full, "ssim_bbox": bbox, "quality_score": quality(full, bbox)}
        for key, expected in counts.items():
            csv_count = int(csv_row[key])
            json_count = int(json_row[key])
            if csv_count != json_count or csv_count != expected:
                raise ValueError(
                    f"{spec.exp_id}:{scope}:{key} CSV={csv_count} JSON={json_count} expected={expected}"
                )
            parsed[key] = csv_count
        parsed_scopes[scope] = parsed

    skipped = load_json_strict(skipped_path)
    if not isinstance(skipped, list):
        raise ValueError(f"{spec.exp_id}: skipped.json is not a list")
    if json_document.get("skipped") != skipped:
        raise ValueError(f"{spec.exp_id}: metrics.json/skipped.json mismatch")
    loss = np.load(loss_path, allow_pickle=False)
    if loss.ndim != 2 or loss.shape[1] < 2 or loss.shape[0] != spec.epochs:
        raise ValueError(
            f"{spec.exp_id}: val_loss_log shape {loss.shape}, expected ({spec.epochs}, >=2)"
        )
    if not np.isfinite(loss[:, :2]).all():
        raise ValueError(f"{spec.exp_id}: non-finite loss history")
    best_index = int(np.argmin(loss[:, 1]))
    context = context_path.read_text(encoding="utf-8", errors="strict")
    branch_match = re.search(r"^git_branch:\s*(\S+)$", context, re.MULTILINE)
    commit_match = re.search(r"^git_commit:\s*([0-9a-f]{7,40})$", context, re.MULTILINE)
    if branch_match is None or commit_match is None:
        raise ValueError(f"{spec.exp_id}: missing Git provenance in run_context.txt")

    overall = parsed_scopes["overall"]
    acc4 = parsed_scopes["acc4"]
    acc8 = parsed_scopes["acc8"]
    return {
        "exp_id": spec.exp_id,
        "expected": asdict(spec),
        "git_branch": branch_match.group(1),
        "git_commit": commit_match.group(1),
        "ssim_full": overall["ssim_full"],
        "ssim_bbox": overall["ssim_bbox"],
        "quality_score": overall["quality_score"],
        "ssim_full_acc4": acc4["ssim_full"],
        "ssim_bbox_acc4": acc4["ssim_bbox"],
        "quality_acc4": acc4["quality_score"],
        "ssim_full_acc8": acc8["ssim_full"],
        "ssim_bbox_acc8": acc8["ssim_bbox"],
        "quality_acc8": acc8["quality_score"],
        "volumes": overall["volumes"],
        "slices": overall["slices"],
        "bbox_annotations": overall["bbox_annotations"],
        "skipped_count": len(skipped),
        "val_loss_last": finite_number(loss[-1, 1], f"{spec.exp_id}:val_loss_last"),
        "val_loss_best": finite_number(loss[best_index, 1], f"{spec.exp_id}:val_loss_best"),
        "best_epoch": int(loss[best_index, 0]),
        "epochs_recorded": int(loss.shape[0]),
    }


def component_deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ("ssim_full", "ssim_bbox", "quality_score", "quality_acc4", "quality_acc8")
    return {key: float(candidate[key]) - float(baseline[key]) for key in keys}


def evaluate_gate(
    baseline_e5: dict[str, Any],
    candidate_e5: dict[str, Any],
    baseline_seed431: dict[str, Any],
    candidate_seed431: dict[str, Any],
    candidate_e10: dict[str, Any],
    thresholds: Thresholds,
) -> dict[str, Any]:
    e5_delta = component_deltas(candidate_e5, baseline_e5)
    seed431_delta = component_deltas(candidate_seed431, baseline_seed431)
    checks = {
        "e5_quality_gain": e5_delta["quality_score"] > thresholds.min_e5_quality_gain,
        "e5_full_component": e5_delta["ssim_full"] >= -thresholds.max_component_regression,
        "e5_bbox_component": e5_delta["ssim_bbox"] >= -thresholds.max_component_regression,
        "seed431_quality_gain": seed431_delta["quality_score"] > thresholds.min_seed431_quality_gain,
        "e10_trajectory": (
            float(candidate_e10["quality_score"]) - float(candidate_e5["quality_score"])
            >= -thresholds.max_e10_vs_e5_quality_drop
        ),
        "source_integrity": all(
            run["skipped_count"] == 0
            and run["volumes"] == 30
            and run["slices"] == 791
            and run["bbox_annotations"] == 161
            for run in (
                baseline_e5,
                candidate_e5,
                baseline_seed431,
                candidate_seed431,
                candidate_e10,
            )
        ),
    }
    return {
        "checks": checks,
        "proposal_eligible": all(checks.values()),
        "e5_candidate_minus_baseline": e5_delta,
        "seed431_candidate_minus_baseline": seed431_delta,
        "candidate_e10_minus_e5_quality": (
            float(candidate_e10["quality_score"]) - float(candidate_e5["quality_score"])
        ),
        "thresholds": asdict(thresholds),
    }


def build_payload(repo_root: Path, result_root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    official_path = repo_root / "experiments/experiment_log.csv"
    handoff_path = repo_root / "reports/phase2/EXP031_validation_handoff.json"
    plan_path = repo_root / "reports/local_comparisons/local_probe_adaptive_followup_plan_20260711.json"
    pending: list[str] = []
    errors: list[str] = []

    if not official_path.exists():
        pending.append(f"missing:{official_path}")
    if not handoff_path.exists():
        pending.append(f"official_handoff:{handoff_path}")
    if not plan_path.exists():
        pending.append(f"missing:{plan_path}")
    if not official_path.exists() or not plan_path.exists():
        return {}, pending, errors

    try:
        official = parse_official_log(official_path)
        exp013 = official.get("EXP013")
        exp030 = official.get("EXP030")
        exp031 = find_exp031(official)
    except (ValueError, OSError, csv.Error) as exc:
        errors.append(str(exc))
        return {"source_status": "pending_or_invalid", "pending": sorted(set(pending)), "errors": errors}, pending, errors
    for name, row, epochs in (("EXP013", exp013, 10), ("EXP030", exp030, 20), ("EXP031", exp031, 30)):
        if row is None:
            pending.append(f"official:{name}")
        else:
            errors.extend(validate_official_row(row, epochs))

    handoff: dict[str, Any] | None = None
    if handoff_path.exists():
        try:
            handoff = parse_exp031_handoff(handoff_path)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if exp031 is not None and handoff is not None:
        errors.extend(cross_check_exp031(exp031, handoff))

    try:
        plan = load_json_strict(plan_path)
        if not isinstance(plan, dict):
            raise ValueError(f"{plan_path}: plan must be a JSON object")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        return {"source_status": "pending_or_invalid", "pending": sorted(set(pending)), "errors": errors}, pending, errors
    candidate = plan.get("selected_candidate")
    expected_candidate_id = "LOCAL_EXP018_varnet_c4_ch16_s8_e1"
    if not isinstance(candidate, dict) or (
        candidate.get("cascade"), candidate.get("chans"), candidate.get("sens_chans")
    ) != (4, 16, 8) or candidate.get("exp_id") != expected_candidate_id:
        errors.append(f"Unexpected selected candidate: {candidate!r}")
        candidate = {
            "exp_id": expected_candidate_id,
            "cascade": 4,
            "chans": 16,
            "sens_chans": 8,
        }
    plan_thresholds = plan.get("decision_thresholds")
    if plan_thresholds != asdict(Thresholds()):
        errors.append("Adaptive plan decision_thresholds do not match report-builder thresholds")
    if not plan.get("required_evidence") or not plan.get("selection_population"):
        errors.append("Adaptive plan lacks required_evidence or selection_population")

    required_local = {
        "baseline_e1_seed430": LocalSpec("LOCAL_EXP013_varnet_c4_ch12_s8_e1", 4, 12, 8, 1, 430),
        "candidate_e1_seed430": LocalSpec(str(candidate["exp_id"]), 4, 16, 8, 1, 430),
        "baseline_e5_seed430": LocalSpec("LOCAL_EXP029_varnet_c4_ch12_s8_e5", 4, 12, 8, 5, 430),
        "candidate_e5_seed430": LocalSpec("LOCAL_EXP032_varnet_c4_ch16_s8_e5", 4, 16, 8, 5, 430),
        "baseline_e1_seed431": LocalSpec("LOCAL_EXP033_varnet_c4_ch12_s8_e1_seed431", 4, 12, 8, 1, 431),
        "candidate_e1_seed431": LocalSpec("LOCAL_EXP034_varnet_c4_ch16_s8_e1_seed431", 4, 16, 8, 1, 431),
        "candidate_e10_seed430": LocalSpec("LOCAL_EXP035_varnet_c4_ch16_s8_e10", 4, 16, 8, 10, 430),
    }
    local: dict[str, dict[str, Any]] = {}
    for role, spec in required_local.items():
        try:
            local[role] = parse_local_metrics(result_root, spec)
        except FileNotFoundError:
            pending.append(f"local:{spec.exp_id}")
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    if "candidate_e1_seed430" in local:
        try:
            planned_quality = unit_metric(candidate.get("quality_score"), "adaptive plan:selected_candidate:quality_score")
            source_quality = float(local["candidate_e1_seed430"]["quality_score"])
            if not close(planned_quality, source_quality):
                errors.append(
                    f"Adaptive plan candidate quality {planned_quality} does not match source {source_quality}"
                )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    payload: dict[str, Any] = {
        "scope": "Decision support only; LOCAL checkpoints are never official candidates",
        "source_status": "ready" if not pending and not errors else "pending_or_invalid",
        "pending": sorted(set(pending)),
        "errors": errors,
        "official": {
            key: value
            for key, value in {"EXP013": exp013, "EXP030": exp030, "EXP031": exp031}.items()
            if value is not None
        },
        "official_exp031_handoff": handoff,
        "selected_local_candidate": candidate,
        "local": local,
    }

    if not pending and not errors:
        assert exp013 is not None and exp030 is not None and exp031 is not None and handoff is not None
        gate = evaluate_gate(
            local["baseline_e5_seed430"],
            local["candidate_e5_seed430"],
            local["baseline_e1_seed431"],
            local["candidate_e1_seed431"],
            local["candidate_e10_seed430"],
            Thresholds(),
        )
        payload["gate"] = gate
        payload["official_learning_curve"] = {
            "EXP013_e10_quality": exp013["quality_score"],
            "EXP030_e20_quality": exp030["quality_score"],
            "EXP031_e30_quality": exp031["quality_score"],
            "EXP030_minus_EXP013": exp030["quality_score"] - exp013["quality_score"],
            "EXP031_minus_EXP030": exp031["quality_score"] - exp030["quality_score"],
        }
        payload["source_status"] = "ready"
    return payload, pending, errors


def markdown_report(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    official = payload["official_learning_curve"]
    local = payload["local"]
    lines = [
        "# EXP031 and LOCAL Candidate Decision Report",
        "",
        "**Scope:** Decision support only. LOCAL checkpoints are exploratory and cannot be submitted as official candidates.",
        "",
        "## Official c4/ch12/s8 learning curve",
        "",
        "| run | epochs | quality |",
        "|---|---:|---:|",
        f"| EXP013 | 10 | {official['EXP013_e10_quality']:.10f} |",
        f"| EXP030 | 20 | {official['EXP030_e20_quality']:.10f} |",
        f"| EXP031 | 30 | {official['EXP031_e30_quality']:.10f} |",
        "",
        f"EXP031 minus EXP030 quality: `{official['EXP031_minus_EXP030']:+.10f}`.",
        "",
        "## Matched LOCAL evidence",
        "",
        "| comparison | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
        (
            f"| five epochs, seed 430 | {local['baseline_e5_seed430']['quality_score']:.10f} | "
            f"{local['candidate_e5_seed430']['quality_score']:.10f} | "
            f"{gate['e5_candidate_minus_baseline']['quality_score']:+.10f} |"
        ),
        (
            f"| one epoch, seed 431 | {local['baseline_e1_seed431']['quality_score']:.10f} | "
            f"{local['candidate_e1_seed431']['quality_score']:.10f} | "
            f"{gate['seed431_candidate_minus_baseline']['quality_score']:+.10f} |"
        ),
        "",
        "## Decision gate",
        "",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            f"Proposal eligible: **{'YES' if gate['proposal_eligible'] else 'NO'}**",
            "",
            "A YES permits only a proposal for a separately approved VESSL experiment. It does not promote or submit a LOCAL checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def self_test() -> None:
    def run(value: float, full: float | None = None, bbox: float | None = None) -> dict[str, Any]:
        full_value = value if full is None else full
        bbox_value = value if bbox is None else bbox
        return {
            "quality_score": value,
            "ssim_full": full_value,
            "ssim_bbox": bbox_value,
            "quality_acc4": value,
            "quality_acc8": value,
            "skipped_count": 0,
            "volumes": 30,
            "slices": 791,
            "bbox_annotations": 161,
        }

    passing = evaluate_gate(run(0.90), run(0.91), run(0.89), run(0.90), run(0.92), Thresholds())
    assert passing["proposal_eligible"]
    reversing = evaluate_gate(run(0.90), run(0.91), run(0.90), run(0.89), run(0.92), Thresholds())
    assert not reversing["proposal_eligible"]
    component_drop = evaluate_gate(
        run(0.90, 0.90, 0.90), run(0.91, 0.898, 0.922), run(0.89), run(0.90), run(0.92), Thresholds()
    )
    assert not component_drop["proposal_eligible"]
    try:
        unit_metric(float("nan"), "nan-test")
    except ValueError:
        pass
    else:
        raise AssertionError("NaN metric must be rejected")
    try:
        reject_duplicate_pairs([("scope", 1), ("scope", 2)])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate JSON keys must be rejected")
    try:
        rows_by_unique_key([{"scope": "overall"}, {"scope": "overall"}], "scope", "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate row keys must be rejected")
    print("self_test=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(os.environ.get("FASTMRI_RESULT_DIR", str(Path.home() / "result"))),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    repo_root = args.repo_root.resolve()
    payload, pending, errors = build_payload(repo_root, args.result_root.resolve())
    report_dir = repo_root / "reports/local_comparisons"
    outputs = [
        report_dir / "exp031_candidate_decision.json",
        report_dir / "exp031_candidate_decision.md",
    ]
    stale_outputs = [str(path) for path in outputs if path.exists()] if (pending or errors) else []
    print(
        json.dumps(
            {
                "status": payload.get("source_status", "unavailable"),
                "pending": sorted(set(pending)),
                "errors": errors,
                "stale_outputs": stale_outputs,
            },
            indent=2,
        )
    )
    if errors or (args.write and stale_outputs):
        return 2
    if pending:
        return 3
    if not args.write:
        return 0

    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(outputs[0], json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_write(outputs[1], markdown_report(payload))
    print(f"wrote:{outputs[0]}")
    print(f"wrote:{outputs[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
