#!/usr/bin/env python3
"""Fail-closed verifier for the one R23 final package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
STRUCTURE = (
    "README.md",
    "requirements.txt",
    "recon_eval.sh",
    "reproduce_final.sh",
    "verify_package.py",
)
FINAL_STRUCTURE = (
    "best_model.pt",
    "project/recon_eval.py",
    "project/reconstruct.py",
    "project/utils/learning/test_part.py",
    "project/utils/learning/promptmr_production.py",
    "project/utils/learning/promptmr_legal_mask.py",
    "project/utils/learning/promptmr_mask_conditioning.py",
    "project/utils/learning/promptmr_mask_router.py",
    "project/utils/learning/promptmr_router.py",
    "project/utils/learning/promptmr_post_refiner.py",
    "project/utils/model/promptmr_plus_adapter.py",
    "project/third_party/promptmr_plus/SOURCE_MANIFEST.json",
    "reproduction/FINAL_C10_SINGLE_LINEAGE_R23_E49.json",
    "reproduction/final-tactics-c10-r23-e49.json",
    "reproduction/organizer-data-provenance.json",
    "reproduction/source-sha256sums.txt",
    "evidence/generalist-e49-receipt.json",
    "evidence/acc4-specialist-receipt.json",
    "evidence/acc8-specialist-receipt.json",
    "evidence/naf-s-training-receipt.json",
    "evidence/policy-receipt.json",
    "evidence/inference-admission-receipt.json",
)
LEARNED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"FINAL_PACKAGE_INVALID: {message}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--structure-only",
    action="store_true",
    help="verify the pre-weight package skeleton",
)
parser.add_argument(
    "--submission-ready",
    action="store_true",
    help="also require the official evaluation or submission receipt",
)
args = parser.parse_args()

for relative in STRUCTURE:
    if not (ROOT / relative).is_file():
        fail(f"missing {relative}")

if args.structure_only:
    print("FINAL_PACKAGE_STRUCTURE_OK")
    raise SystemExit(0)

manifest_path = ROOT / "package-manifest.json"
model_path = ROOT / "best_model.pt"
if not manifest_path.is_file() or not model_path.is_file():
    fail("sealed manifest or best_model.pt is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

for relative in FINAL_STRUCTURE:
    if not (ROOT / relative).is_file():
        fail(f"missing final artifact: {relative}")
if args.submission_ready and not (
    ROOT / "evidence/official-evaluation-receipt.json"
).is_file():
    fail("official evaluation receipt is missing")

required = {
    "schema": "fastmri-r23-single-final-package-v1",
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
}
for key, expected in required.items():
    if manifest.get(key) != expected:
        fail(f"manifest {key}={manifest.get(key)!r}, expected {expected!r}")

files = manifest.get("files")
if not isinstance(files, dict) or not files:
    fail("manifest file map is absent")
actual_files = {
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if (
        path.is_file()
        and path != manifest_path
        and path.relative_to(ROOT).parts[0] != "result"
    )
}
unsafe_links = sorted(
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_symlink() and path.relative_to(ROOT).parts[0] != "result"
)
if unsafe_links:
    fail(f"symlinks are forbidden: {unsafe_links}")
sealed_files = set(files)
if actual_files != sealed_files:
    fail(
        "manifest coverage mismatch: "
        f"unsealed={sorted(actual_files - sealed_files)}, "
        f"missing={sorted(sealed_files - actual_files)}"
    )
for relative, contract in files.items():
    if not isinstance(relative, str) or relative.startswith(("/", "..")):
        fail(f"unsafe manifest path: {relative!r}")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        fail(f"manifest path escapes package: {relative}")
    if not path.is_file() or path.is_symlink():
        fail(f"missing or symlinked file: {relative}")
    if not isinstance(contract, dict):
        fail(f"invalid file contract: {relative}")
    if path.stat().st_size != int(contract.get("bytes", -1)):
        fail(f"size mismatch: {relative}")
    if sha256(path) != contract.get("sha256"):
        fail(f"SHA-256 mismatch: {relative}")
for relative in FINAL_STRUCTURE:
    if relative not in files:
        fail(f"required artifact is not sealed by manifest: {relative}")
if args.submission_ready and "evidence/official-evaluation-receipt.json" not in files:
    fail("official evaluation receipt is not sealed by manifest")

learned = sorted(
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if (
        path.is_file()
        and path.suffix.lower() in LEARNED_SUFFIXES
        and path.relative_to(ROOT).parts[0] != "result"
    )
)
if learned != ["best_model.pt"]:
    fail(f"expected exactly one learned-state file, observed {learned}")
if "best_model.pt" not in files:
    fail("best_model.pt is not sealed by the manifest")

print(
    json.dumps(
        {
            "state": "FINAL_PACKAGE_OK",
            "candidate_count": 1,
            "best_model_sha256": sha256(model_path),
            "file_count": len(files),
        },
        sort_keys=True,
    )
)
