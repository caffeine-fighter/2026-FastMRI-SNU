#!/usr/bin/env python3
"""Atomically seal the fully populated pre-evaluation R30 package."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "package-manifest.json"
MODEL = ROOT / "best_model.pt"
LEARNED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
KST = ZoneInfo("Asia/Seoul")
HARD_DEADLINE_UNIX = 1_787_237_940


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"FINAL_PACKAGE_SEAL_REFUSED: {message}")


if MANIFEST.exists():
    fail("package-manifest.json already exists")
if time.time() >= HARD_DEADLINE_UNIX:
    fail("package sealing is after the hard deadline")
if not MODEL.is_file() or MODEL.is_symlink():
    fail("one regular best_model.pt is required")
if (ROOT / "result").exists():
    fail("generated result directory must not exist before sealing")
if any((ROOT / "evidence").glob("official-evaluation-*")):
    fail("official evaluation evidence exists before package sealing")

unsafe_links = sorted(
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_symlink()
)
if unsafe_links:
    fail(f"symlinks are forbidden: {unsafe_links}")

learned = sorted(
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() in LEARNED_SUFFIXES
)
if learned != ["best_model.pt"]:
    fail(f"exactly one learned-state file is required, observed {learned}")

files = {}
for path in sorted(ROOT.rglob("*")):
    if path.is_file():
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
if not files:
    fail("package has no files")

now = time.time()
manifest = {
    "schema": "fastmri-r30-single-final-package-v1",
    "state": "SEALED",
    "candidate_count": 1,
    "final_package_count": 1,
    "fallback_registered": False,
    "official_evaluation_max_runs": 1,
    "generalist_handoff_epoch": 49,
    "generalist_handoff_optimizer_step": 228928,
    "scheduler_horizon_epoch": 51,
    "external_learned_state_imported": False,
    "leaderboard_data_used_for_training_or_selection": False,
    "hard_deadline_unix": HARD_DEADLINE_UNIX,
    "hard_deadline_kst": "2026-08-20T23:59:00+09:00",
    "sealed_unix": now,
    "sealed_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
    "sealed_kst": datetime.fromtimestamp(now, KST).isoformat(),
    "files": files,
}
temporary = MANIFEST.with_name(f".{MANIFEST.name}.{uuid.uuid4().hex}.tmp")
with temporary.open("xb") as handle:
    handle.write(
        (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
    )
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, MANIFEST)

completed = subprocess.run(
    [sys.executable, str(ROOT / "verify_package.py")],
    cwd=ROOT,
    text=True,
    capture_output=True,
    check=False,
)
if completed.returncode != 0:
    MANIFEST.unlink(missing_ok=True)
    sys.stderr.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    fail("post-seal verification failed; generated manifest was removed")

print(completed.stdout.strip())
print(
    json.dumps(
        {
            "state": "FINAL_PACKAGE_SEALED_PRE_EVALUATION",
            "candidate_count": 1,
            "best_model_sha256": sha256(MODEL),
            "manifest_sha256": sha256(MANIFEST),
            "file_count": len(files),
        },
        sort_keys=True,
    )
)
