#!/usr/bin/env python3
"""Normalize and fail-closed validate the authoritative R30 VESSL receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


ROOT = Path(__file__).resolve().parent
REPRO = ROOT / "reproduction"
EVIDENCE = ROOT / "evidence"
MODEL = ROOT / "best_model.pt"
TACTICS = REPRO / "final-tactics-c10-r30-neighbor-zf.json"
AMENDMENT = REPRO / "FINAL_C10_SINGLE_LINEAGE_R30_NEIGHBOR_ZF.json"
RUNTIME = REPRO / "FINAL_C10_SINGLE_LINEAGE_R30_INFERENCE.json"
PREFLIGHT = REPRO / "R30_CPU_PREFLIGHT.json"
INPUT_MODE = "recon_zero_filled_residual_neighbor_zf"
ZF_DEFINITION = "rss(fftshift(ifft2(ifftshift(masked_kspace),norm=ortho)))"


def fail(message: str) -> None:
    raise SystemExit(f"R30_EVIDENCE_REFUSED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"invalid JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(
            (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def copy_exact(source: Path, destination: Path) -> None:
    if destination.exists():
        fail(f"normalized evidence already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        fail(f"copy hash mismatch: {source}")


parser = argparse.ArgumentParser()
parser.add_argument("--controller-receipt", type=Path, required=True)
parser.add_argument("--r30-deployment-receipt", type=Path, required=True)
parser.add_argument("--generalist-receipt", type=Path, required=True)
parser.add_argument("--acc4-terminal", type=Path, required=True)
parser.add_argument("--acc8-terminal", type=Path, required=True)
parser.add_argument("--naf-receipt", type=Path, required=True)
args = parser.parse_args()

for required in (MODEL, TACTICS, AMENDMENT, RUNTIME, PREFLIGHT):
    if not required.is_file() or required.is_symlink():
        fail(f"required sealed input is absent: {required}")

try:
    import torch

    model = torch.load(MODEL, map_location="cpu", weights_only=True)
except Exception as error:
    fail(f"cannot safely load best_model.pt: {type(error).__name__}: {error}")
if not isinstance(model, dict):
    fail("best_model.pt is not a dictionary checkpoint")
components = model.get("components")
post = model.get("post_refiner")
if not isinstance(components, dict) or not isinstance(post, dict):
    fail("routed checkpoint components are absent")
try:
    source_hashes = {
        "generalist": components["generalist"]["source_checkpoint_sha256"],
        "acc4": components["acc4"]["source_checkpoint_sha256"],
        "acc8": components["acc8"]["source_checkpoint_sha256"],
        "post_refiner": post["source_checkpoint_sha256"],
    }
except (KeyError, TypeError) as error:
    fail(f"routed checkpoint source binding is absent: {error}")
if any(not isinstance(value, str) or len(value) != 64 for value in source_hashes.values()):
    fail("routed checkpoint contains an invalid source SHA-256")

tactics = load(TACTICS)
amendment = load(AMENDMENT)
runtime = load(RUNTIME)
preflight = load(PREFLIGHT)
policy = amendment.get("policy")
change = amendment.get("amendment")
if (
    tactics.get("schema") != "vessl-final-tactics-scalar-handoff-v1"
    or tactics.get("state") != "SEALED"
    or amendment.get("schema") != "final-c10-single-lineage-r30-neighbor-zf-amendment-v1"
    or amendment.get("state") != "SEALED"
    or runtime.get("schema") != "final-c10-single-lineage-r30-inference-amendment-v1"
    or runtime.get("state") != "SEALED"
    or preflight.get("schema") != "vessl-r30-neighbor-zf-cpu-preflight-v1"
    or preflight.get("state") != "PASS"
    or preflight.get("cpu_only") is not True
    or preflight.get("cuda_initialized") is not False
    or preflight.get("actual_shipped_router_validation") is not True
    or preflight.get("timed_neighbor_recon_slice_executed") is not True
    or preflight.get("zero_initialized_output_identity") is not True
    or preflight.get("post_e49_optimizer_steps") != 97_061
    or preflight.get("naf_s_parameter_count") != 73_489
    or preflight.get("unknown_mask_outer_tta") != ["identity"]
    or not isinstance(policy, dict)
    or policy.get("candidate_count") != 1
    or policy.get("fallback_registered") is not False
    or policy.get("official_evaluation_max_runs") != 1
    or not isinstance(change, dict)
    or change.get("input_mode") != INPUT_MODE
    or change.get("zero_filled_definition") != ZF_DEFINITION
    or change.get("post_e49_total_optimizer_steps") != 97_061
    or change.get("optimizer_step_budget_changed") is not False
):
    fail("sealed R30 contract or CPU preflight is invalid")

source_paths = {
    "controller.py": REPRO / "controller.py",
    "train.py": REPRO / "specialist/train.py",
    "promptmr_production.py": REPRO / "specialist/promptmr_production.py",
    "vessl_train_post_refiner.py": REPRO / "vessl_train_post_refiner.py",
    "vessl_build_routed_promptmr_checkpoint.py": REPRO / "vessl_build_routed_promptmr_checkpoint.py",
    "promptmr_post_refiner.py": REPRO / "promptmr_post_refiner.py",
    "promptmr_router.py": REPRO / "promptmr_router.py",
    "promptmr_mask_router.py": REPRO / "promptmr_mask_router.py",
    "promptmr_legal_mask.py": REPRO / "promptmr_legal_mask.py",
    "test_part.py": REPRO / "test_part.py",
    "preflight_r30.py": REPRO / "preflight_r30.py",
}
sealed_sources = amendment.get("source_hashes")
if not isinstance(sealed_sources, dict):
    fail("R30 source hash map is absent")
for name, path in source_paths.items():
    if not path.is_file() or sha256(path) != sealed_sources.get(name):
        fail(f"R30 reproduction source drifted: {name}")

deployment = load(args.r30_deployment_receipt)
deployment_sources = deployment.get("source_hashes")
if (
    deployment.get("schema") != "vessl-c10-single-lineage-r30-neighbor-zf-deployment-v1"
    or deployment.get("state") != "ACTIVE_CURRENT_C10_PRESERVED_R30_NEIGHBOR_ZF_ARMED"
    or deployment.get("trainer_signal_sent") is not False
    or deployment.get("active_generalist_process_touched") is not False
    or deployment.get("active_generalist_recipe_changed") is not False
    or deployment.get("future_specialist_recipe_changed") is not True
    or deployment.get("post_e49_optimizer_step_budget_changed") is not False
    or deployment.get("post_e49_optimizer_steps") != 97_061
    or deployment.get("candidate_count") != 1
    or deployment.get("final_package_count") != 1
    or deployment.get("fallback_registered") is not False
    or deployment.get("official_evaluation_max_runs") != 1
    or deployment.get("r30_amendment_sha256") != sha256(AMENDMENT)
    or deployment.get("r30_tactics_sha256") != sha256(TACTICS)
    or deployment.get("r30_runtime_sha256") != sha256(RUNTIME)
    or not isinstance(deployment_sources, dict)
    or any(deployment_sources.get(name) != digest for name, digest in sealed_sources.items())
):
    fail("R30 deployment did not preserve the active single lineage")

controller = load(args.controller_receipt)
model_sha = sha256(MODEL)
admission = controller.get("final_admission")
if (
    controller.get("schema") != "vessl-g10-architecture-dispatcher-receipt-v1"
    or controller.get("state") != "PASS"
    or controller.get("winner") != "R2_C10"
    or controller.get("final_checkpoint_sha256") != model_sha
    or controller.get("final_admission_mode") != "PRIMARY_REQUESTED"
    or controller.get("acc4_specialist_checkpoint_sha256") != source_hashes["acc4"]
    or controller.get("acc8_specialist_checkpoint_sha256") != source_hashes["acc8"]
    or controller.get("post_refiner_checkpoint_sha256") != source_hashes["post_refiner"]
    or controller.get("fallback_checkpoint") is not None
    or controller.get("fallback_checkpoint_sha256") is not None
    or controller.get("leaderboard_data_read") is not False
    or controller.get("external_learned_state_imported") is not False
    or controller.get("all_final_learned_state_vessl_only") is not True
    or not isinstance(admission, dict)
    or admission.get("state") != "PASS"
    or admission.get("final_checkpoint_sha256") != model_sha
    or admission.get("gpu") != "NVIDIA GeForce GTX 1080"
    or admission.get("leaderboard_data_used") is not False
    or admission.get("official_reconstruction_path") is not True
):
    fail("controller final receipt is not the exact R30 primary package")

generalist = load(args.generalist_receipt)
if (
    generalist.get("schema") != "vessl-g10-generalist-terminal-checkpoint-v1"
    or generalist.get("state") != "SEALED"
    or generalist.get("checkpoint_sha256") != source_hashes["generalist"]
    or generalist.get("epoch") != 49
    or generalist.get("optimizer_step") != 228_928
    or generalist.get("scheduler_horizon_epochs") != 51
    or generalist.get("scheduler_total_steps") != 238_272
):
    fail("generalist receipt is not the exact E49 handoff")

specialist_receipts: dict[str, dict] = {}
for route, terminal_path, steps, horizon, epoch in (
    ("acc4", args.acc4_terminal, 7_008, 35_040, 2),
    ("acc8", args.acc8_terminal, 1_158, 2_315, 0),
):
    terminal = load(terminal_path)
    checkpoint = controller.get(f"{route}_specialist_checkpoint")
    if (
        terminal.get("status") != "COMPLETED"
        or terminal.get("epoch") != epoch
        or terminal.get("step") != steps
        or terminal.get("exact_optimizer_step_budget") is not True
        or terminal.get("checkpoint") != checkpoint
    ):
        fail(f"{route} terminal receipt is invalid")
    specialist_receipts[route] = {
        "schema": "fastmri-r30-specialist-receipt-v1",
        "state": "PASS",
        "route": route,
        "checkpoint": checkpoint,
        "checkpoint_sha256": source_hashes[route],
        "parent_checkpoint_sha256": source_hashes["generalist"],
        "parent_epoch": 49,
        "parent_optimizer_step": 228_928,
        "optimizer_steps": steps,
        "lr_horizon_optimizer_steps": horizon,
        "trained_on_vessl": True,
        "external_learned_state_imported": False,
        "leaderboard_data_used": False,
        "source_terminal_sha256": sha256(terminal_path),
    }

naf = load(args.naf_receipt)
if (
    naf.get("schema") != "vessl-post-refiner-training-receipt-v1"
    or naf.get("state") != "PASS"
    or naf.get("checkpoint_sha256") != source_hashes["post_refiner"]
    or naf.get("base_checkpoint_sha256") != source_hashes["generalist"]
    or naf.get("routed_branch_sha256") != {"acc4": source_hashes["acc4"], "acc8": source_hashes["acc8"]}
    or naf.get("variant") != "NAF_S"
    or naf.get("epochs") != 21
    or naf.get("parent_epoch") != 49
    or naf.get("optimizer_steps") != 88_895
    or naf.get("lr_horizon_optimizer_steps") != 93_567
    or naf.get("input_mode") != INPUT_MODE
    or naf.get("zero_filled_definition") != ZF_DEFINITION
    or naf.get("normalization") != "shared_detached_reconstruction_amax"
    or naf.get("spatial_match") != "center_crop_then_zero_pad"
    or naf.get("adjacent_slice_context")
    != {
        "count": 3,
        "positions": ["previous", "current", "next"],
        "boundary_policy": "replicate_nearest_slice",
        "source": "same_volume_masked_kspace_only",
    }
    or naf.get("main_parameters_updated") is not False
    or naf.get("external_learned_state_imported") is not False
):
    fail("NAF_S receipt is not the exact R30 neighbor-ZF contract")

policy_receipt = {
    "schema": "fastmri-r30-policy-receipt-v1",
    "state": "PASS",
    "candidate_count": 1,
    "final_package_count": 1,
    "fallback_registered": False,
    "official_evaluation_max_runs": 1,
    "routing_input": "input_kspace_mask_only_inside_recon_slice",
    "unknown_or_mismatch_route": "generalist_identity",
    "post_refiner_input_mode": INPUT_MODE,
    "adjacent_slice_context": {
        "count": 3,
        "boundary_policy": "replicate_nearest_slice",
        "source": "same_volume_masked_kspace_only",
    },
    "learned_state_source": "VESSL_ONLY",
    "external_learned_state_imported": False,
    "leaderboard_data_used_for_training_or_selection": False,
    "tactics_sha256": sha256(TACTICS),
    "amendment_sha256": sha256(AMENDMENT),
    "runtime_sha256": sha256(RUNTIME),
    "deployment_receipt_sha256": sha256(args.r30_deployment_receipt),
    "controller_receipt_sha256": sha256(args.controller_receipt),
}

outputs = {
    EVIDENCE / "generalist-e49-receipt.json": generalist,
    EVIDENCE / "acc4-specialist-receipt.json": specialist_receipts["acc4"],
    EVIDENCE / "acc8-specialist-receipt.json": specialist_receipts["acc8"],
    EVIDENCE / "naf-s-training-receipt.json": naf,
    EVIDENCE / "policy-receipt.json": policy_receipt,
    EVIDENCE / "inference-admission-receipt.json": admission,
}
for path, payload in outputs.items():
    if path.exists():
        fail(f"normalized evidence already exists: {path}")
    atomic_json(path, payload)
copy_exact(args.r30_deployment_receipt, EVIDENCE / "r30-amendment-deployment-receipt.json")

atomic_json(
    EVIDENCE / "evidence-materialization-receipt.json",
    {
        "schema": "fastmri-r30-evidence-materialization-v1",
        "state": "PASS",
        "created_unix": time.time(),
        "best_model_sha256": model_sha,
        "component_source_sha256": source_hashes,
        "candidate_count": 1,
        "fallback_registered": False,
        "outputs": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (*outputs.keys(), EVIDENCE / "r30-amendment-deployment-receipt.json")
        },
    },
)
print("R30_EVIDENCE_MATERIALIZED")
