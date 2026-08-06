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
    "record_official_evaluation.py",
    "reproduce_final.sh",
    "run_official_evaluation_once.sh",
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


def verify_routed_model(path: Path) -> dict[str, str]:
    """Verify the learned payload, not only its outer file hash."""
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        fail(f"best_model.pt cannot be loaded safely: {type(error).__name__}: {error}")
    if not isinstance(value, dict):
        fail("best_model.pt is not a dictionary checkpoint")
    architecture = value.get("architecture")
    routing = value.get("routing_contract")
    components = value.get("components")
    post = value.get("post_refiner")
    if (
        value.get("schema") != "vessl-acceleration-routed-promptmr-v2"
        or value.get("all_components_trained_on_vessl") is not True
        or architecture
        != {
            "rung": "R2",
            "num_cascades": 10,
            "n_history": 0,
            "parameter_count": 54090459,
            "trainable_parameter_count": 45381339,
        }
        or value.get("routing_feature")
        != "exact_legal_mask_family_with_generalist_fail_safe"
        or not isinstance(routing, dict)
        or routing.get("schema") != "promptmr-legal-mask-routing-contract-v2"
        or routing.get("supported_accelerations") != [4, 8]
        or routing.get("generalist_component") != "generalist"
        or routing.get("unknown_or_mismatch") != "generalist"
        or routing.get("public_frequency_weighting") is not False
        or routing.get("generator")
        != {
            "center_fraction": 0.08,
            "acs_width": "round(native_width*0.08)",
            "acs_start": "(native_width-acs_width+1)//2",
            "outer_lines": "column%acceleration==residue",
        }
        or value.get("tta_views") != ["identity"]
        or value.get("tta_views_by_acceleration")
        != {
            "acc4": ["identity"],
            "acc8": ["identity", "flip_lr"],
        }
        or not isinstance(components, dict)
        or set(components) != {"generalist", "acc4", "acc8"}
        or not isinstance(post, dict)
    ):
        fail("best_model.pt outer routed contract is not exact R23")

    expected = {
        "generalist": (None, "generalist", "all", 228928, 49),
        "acc4": (4, "acc4", "acc4", 2336, 0),
        "acc8": (8, "acc8", "acc8", 1158, 0),
    }
    source_hashes: dict[str, str] = {}
    for name, (acceleration, role, route, step, epoch) in expected.items():
        component = components[name]
        source_hash = component.get("source_checkpoint_sha256") if isinstance(component, dict) else None
        if (
            not isinstance(component, dict)
            or component.get("acceleration") != acceleration
            or component.get("role") != role
            or component.get("rung") != "R2"
            or component.get("num_cascades") != 10
            or component.get("n_history") != 0
            or component.get("parameter_count") != 54090459
            or component.get("trainable_parameter_count") != 45381339
            or component.get("train_acceleration") != route
            or component.get("scratch") is not True
            or component.get("external_learned_state") is not False
            or component.get("trained_on_vessl") is not True
            or component.get("source_step") != step
            or (epoch is not None and component.get("source_epoch") != epoch)
            or not isinstance(component.get("model"), dict)
            or not isinstance(source_hash, str)
            or len(source_hash) != 64
        ):
            fail(f"best_model.pt {name} component is not exact R23")
        source_hashes[name] = source_hash
    if len(set(source_hashes.values())) != 3:
        fail("generalist and specialist source hashes must be distinct")

    if (
        post.get("enabled") is not True
        or post.get("role") != "main_output_post_refiner"
        or post.get("variant") != "NAF_S"
        or post.get("views") != ["identity", "flip_lr"]
        or post.get("views_batched") is not True
        or post.get("mask_conditioned") is not False
        or post.get("epoch") != 21
        or post.get("parent_epoch") != 49
        or post.get("late_branch_epochs") != [50, 70]
        or post.get("optimizer_steps") != 93567
        or post.get("steps_per_epoch") != 4672
        or post.get("partial_terminal_epoch") is not True
        or post.get("training_data") != "organizer_train_plus_val_final"
        or post.get("loss_family")
        != "winner_foreground_ssim_l1_sqrt_area_plus_official384_bbox05_v2"
        or post.get("bbox_loss_coefficient") != 0.5
        or post.get("organizer_annotations_used_for_training") is not True
        or post.get("inference_annotation_access") is not False
        or post.get("validation_used_for_checkpoint_selection") is not False
        or post.get("trainable_parameter_scope") != "naf_s_only"
        or post.get("frozen_parameter_scope") != "main_c10_e49_all_parameters"
        or post.get("main_parameters_updated") is not False
        or post.get("parameter_count") != 72625
        or post.get("trained_on_vessl") is not True
        or post.get("external_learned_state_imported") is not False
        or post.get("base_checkpoint_sha256") != source_hashes["generalist"]
        or post.get("routed_branch_sha256")
        != {"acc4": source_hashes["acc4"], "acc8": source_hashes["acc8"]}
        or post.get("sampler_policy") != "equal_acc_real_acc8_real80_virtual20_v1"
        or not isinstance(post.get("post_refiner_state"), dict)
    ):
        fail("best_model.pt NAF_S component is not exact frozen-C10 R23")
    forbidden = {"optimizer", "scheduler", "rng_state", "ema", "swa", "scaler"}
    if forbidden.intersection(value):
        fail("best_model.pt contains forbidden training state")
    return source_hashes


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

source_hashes = verify_routed_model(model_path)

if args.submission_ready:
    receipt_path = ROOT / "evidence/official-evaluation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    start_path = ROOT / "evidence/official-evaluation-start.json"
    start = (
        json.loads(start_path.read_text(encoding="utf-8"))
        if start_path.is_file()
        else {}
    )
    scores = receipt.get("scores")
    log_path = ROOT / str(receipt.get("log", ""))
    if (
        receipt.get("schema") != "fastmri-r23-official-evaluation-receipt-v1"
        or receipt.get("state") != "PASS"
        or receipt.get("attempt") != 1
        or receipt.get("official_evaluation_attempt_count") != 1
        or receipt.get("return_code") != 0
        or receipt.get("command") != "bash run_official_evaluation_once.sh"
        or receipt.get("best_model_sha256") != sha256(model_path)
        or receipt.get("leaderboard_data_used_for_training_or_selection") is not False
        or not isinstance(receipt.get("started_unix"), (int, float))
        or not isinstance(receipt.get("completed_unix"), (int, float))
        or float(receipt["started_unix"]) >= 1787237940
        or float(receipt["completed_unix"]) > 1787237940
        or float(receipt["completed_unix"]) < float(receipt["started_unix"])
        or start.get("schema") != "fastmri-r23-official-evaluation-start-v1"
        or start.get("state") != "STARTED"
        or start.get("attempt") != 1
        or start.get("command") != receipt.get("command")
        or start.get("best_model_sha256") != receipt.get("best_model_sha256")
        or start.get("started_unix") != receipt.get("started_unix")
        or not isinstance(scores, dict)
        or set(scores)
        != {
            "ssim_full",
            "ssim_bbox",
            "recon_time_seconds",
            "ssim_full_acc4",
            "ssim_full_acc8",
            "ssim_bbox_acc4",
            "ssim_bbox_acc8",
        }
        or any(
            not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
            for key, value in scores.items()
            if key.startswith("ssim")
        )
        or not isinstance(scores.get("recon_time_seconds"), (int, float))
        or float(scores["recon_time_seconds"]) <= 0.0
        or not log_path.is_file()
        or log_path.is_symlink()
        or receipt.get("log_sha256") != sha256(log_path)
    ):
        fail("official evaluation receipt is invalid")

print(
    json.dumps(
        {
            "state": "FINAL_PACKAGE_OK",
            "candidate_count": 1,
            "best_model_sha256": sha256(model_path),
            "component_source_sha256": source_hashes,
            "file_count": len(files),
        },
        sort_keys=True,
    )
)
