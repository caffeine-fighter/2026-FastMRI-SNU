#!/usr/bin/env python3
"""Create and seal the one official-evaluation receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
MANIFEST = ROOT / "package-manifest.json"
MODEL = ROOT / "best_model.pt"
START = EVIDENCE / "official-evaluation-start.json"
RECEIPT = EVIDENCE / "official-evaluation-receipt.json"
LOG = EVIDENCE / "official-evaluation.log"
KST = ZoneInfo("Asia/Seoul")
HARD_DEADLINE_UNIX = 1_787_237_940


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(
            (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def package_files() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if (
            path.is_file()
            and path != MANIFEST
            and path.relative_to(ROOT).parts[0] != "result"
        )
    }


def start() -> int:
    if START.exists() or RECEIPT.exists() or LOG.exists():
        raise RuntimeError("official evaluation was already started")
    manifest = load_json(MANIFEST)
    files = manifest.get("files")
    if not isinstance(files, dict) or package_files() != set(files):
        raise RuntimeError("pre-evaluation package is not exactly manifest-sealed")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if now >= HARD_DEADLINE_UNIX:
        raise RuntimeError("official evaluation start is after the hard deadline")
    atomic_json(
        START,
        {
            "schema": "fastmri-r23-official-evaluation-start-v1",
            "state": "STARTED",
            "attempt": 1,
            "command": "bash run_official_evaluation_once.sh",
            "best_model_sha256": sha256(MODEL),
            "pre_evaluation_manifest_sha256": sha256(MANIFEST),
            "started_unix": now,
            "started_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "started_kst": datetime.fromtimestamp(now, KST).isoformat(),
        },
    )
    print("OFFICIAL_EVALUATION_ATTEMPT_1_STARTED")
    return 0


def metric(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match is None:
        raise RuntimeError(f"official output is missing {label}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise RuntimeError(f"official output contains non-finite {label}")
    return value


def finish(return_code: int) -> int:
    if RECEIPT.exists():
        raise RuntimeError("official evaluation receipt already exists")
    start_value = load_json(START)
    if (
        start_value.get("schema") != "fastmri-r23-official-evaluation-start-v1"
        or start_value.get("state") != "STARTED"
        or start_value.get("attempt") != 1
        or start_value.get("best_model_sha256") != sha256(MODEL)
    ):
        raise RuntimeError("official evaluation start record is invalid")
    if not LOG.is_file() or LOG.stat().st_size == 0:
        raise RuntimeError("official evaluation log is absent")
    text = LOG.read_text(encoding="utf-8", errors="replace")
    scores = None
    state = "FAIL"
    if return_code == 0:
        scores = {
            "ssim_full": metric(text, "Leaderboard SSIM_full"),
            "ssim_bbox": metric(text, "Leaderboard SSIM_bbox"),
            "recon_time_seconds": metric(text, "Leaderboard Recon Time"),
            "ssim_full_acc4": metric(text, "SSIM_full (acc4)"),
            "ssim_full_acc8": metric(text, "SSIM_full (acc8)"),
            "ssim_bbox_acc4": metric(text, "SSIM_bbox (acc4)"),
            "ssim_bbox_acc8": metric(text, "SSIM_bbox (acc8)"),
        }
        if (
            any(not 0.0 <= value <= 1.0 for key, value in scores.items() if key.startswith("ssim"))
            or scores["recon_time_seconds"] <= 0.0
        ):
            raise RuntimeError("official evaluation metrics are outside valid bounds")
        state = "PASS"
    now = time.time()
    if now > HARD_DEADLINE_UNIX:
        return_code = return_code or 124
        state = "FAIL"
    receipt = {
        "schema": "fastmri-r23-official-evaluation-receipt-v1",
        "state": state,
        "attempt": 1,
        "return_code": return_code,
        "command": start_value["command"],
        "best_model_sha256": start_value["best_model_sha256"],
        "started_unix": start_value["started_unix"],
        "completed_unix": now,
        "completed_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "completed_kst": datetime.fromtimestamp(now, KST).isoformat(),
        "log": LOG.relative_to(ROOT).as_posix(),
        "log_sha256": sha256(LOG),
        "scores": scores,
        "leaderboard_data_used_for_training_or_selection": False,
        "official_evaluation_attempt_count": 1,
    }
    atomic_json(RECEIPT, receipt)

    manifest = load_json(MANIFEST)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("package manifest file map is absent")
    generated = {
        START.relative_to(ROOT).as_posix(),
        LOG.relative_to(ROOT).as_posix(),
        RECEIPT.relative_to(ROOT).as_posix(),
    }
    if package_files() != set(files) | generated:
        raise RuntimeError("unexpected package mutation during official evaluation")
    for path in (START, LOG, RECEIPT):
        files[path.relative_to(ROOT).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    atomic_json(MANIFEST, manifest)
    print(f"OFFICIAL_EVALUATION_ATTEMPT_1_{state}")
    return 0 if state == "PASS" else return_code or 1


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="action", required=True)
subparsers.add_parser("start")
finish_parser = subparsers.add_parser("finish")
finish_parser.add_argument("--return-code", type=int, required=True)
args = parser.parse_args()

try:
    raise SystemExit(start() if args.action == "start" else finish(args.return_code))
except Exception as error:
    print(f"OFFICIAL_EVALUATION_RECEIPT_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
    raise SystemExit(2)
