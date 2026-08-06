#!/usr/bin/env python3
"""Materialize normalized R23/R24 evidence from authoritative VESSL receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import torch


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
TACTICS = ROOT / "reproduction/final-tactics-c10-r23-e49.json"
SCHEDULER_AMENDMENT = (
    ROOT / "reproduction/FINAL_C10_SINGLE_LINEAGE_R24_SCHEDULER_BOUNDARY.json"
)
POST_E49_PREFLIGHT = (
    ROOT / "reproduction/R24_POST_E49_COMMAND_PARSER_PREFLIGHT.json"
)
MODEL = ROOT / "best_model.pt"
R24_PRODUCTION_SHA256 = (
    "ea4695f5fada7c417323d9efad495544d0743ad1d35b3023c1a645a421d8688b"
)
R24_CONTROLLER_SHA256 = (
    "929e53558cbb976f011d9ab925980c12a8b39be49b283a2de71621f45c18ce31"
)
R24_TRAIN_SHA256 = (
    "95465b1b09189af87359a39518559f1759fdbeb881a4abb35b1f7b7faa832e47"
)
R23_POST_TRAINER_SHA256 = (
    "23ea62bf1e95772e5d1b00392718dff0604d554f638eb354793bb0613f1b6428"
)
R23_BUILDER_SHA256 = (
    "3d5263fc93e631c2b769201b9f547a6bfa8ca9b03270fc92ebd6287ffe6530a7"
)


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
        raise RuntimeError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


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


parser = argparse.ArgumentParser()
parser.add_argument("--controller-receipt", type=Path, required=True)
parser.add_argument(
    "--scheduler-deployment-receipt",
    type=Path,
    default=Path(
        "/root/result/"
        "VESSL_G10_G11_TERMINAL_SUCCESSOR_AMENDMENT_R24_SCHEDULER_FIX_R1/"
        "r24-deployment-receipt.json"
    ),
)
parser.add_argument("--generalist-receipt", type=Path)
parser.add_argument("--acc4-terminal", type=Path)
parser.add_argument("--acc8-terminal", type=Path)
parser.add_argument("--naf-receipt", type=Path)
args = parser.parse_args()

if not all(
    path.is_file()
    for path in (MODEL, TACTICS, SCHEDULER_AMENDMENT, POST_E49_PREFLIGHT)
):
    raise SystemExit(
        "R23_EVIDENCE_REFUSED: model, sealed tactics, R24 amendment, or "
        "post-E49 preflight is absent"
    )
outputs = {
    "generalist": EVIDENCE / "generalist-e49-receipt.json",
    "acc4": EVIDENCE / "acc4-specialist-receipt.json",
    "acc8": EVIDENCE / "acc8-specialist-receipt.json",
    "naf": EVIDENCE / "naf-s-training-receipt.json",
    "policy": EVIDENCE / "policy-receipt.json",
    "admission": EVIDENCE / "inference-admission-receipt.json",
    "scheduler": EVIDENCE / "scheduler-amendment-deployment-receipt.json",
}
if any(path.exists() for path in outputs.values()):
    raise SystemExit("R23_EVIDENCE_REFUSED: normalized evidence already exists")

try:
    model = torch.load(MODEL, map_location="cpu", weights_only=True)
    components = model["components"]
    source_hashes = {
        name: components[name]["source_checkpoint_sha256"]
        for name in ("generalist", "acc4", "acc8")
    }
    source_hashes["post_refiner"] = model["post_refiner"][
        "source_checkpoint_sha256"
    ]
except Exception as error:
    raise SystemExit(
        f"R23_EVIDENCE_REFUSED: invalid routed model: {type(error).__name__}: {error}"
    ) from error

tactics_envelope = load(TACTICS)
tactics = tactics_envelope.get("selected_recipe")
controller = load(args.controller_receipt)
scheduler_amendment = load(SCHEDULER_AMENDMENT)
scheduler_deployment = load(args.scheduler_deployment_receipt)
post_e49_preflight = load(POST_E49_PREFLIGHT)


def controller_artifact(
    explicit: Path | None,
    run_key: str,
    filename: str,
) -> Path:
    if explicit is not None:
        return explicit
    run_value = controller.get(run_key)
    if not isinstance(run_value, str) or not run_value:
        raise SystemExit(
            f"R23_EVIDENCE_REFUSED: controller is missing {run_key}"
        )
    return Path(run_value) / filename


generalist_receipt_path = controller_artifact(
    args.generalist_receipt,
    "selected_run",
    "controller-terminal-checkpoint.json",
)
terminal_paths = {
    "acc4": controller_artifact(
        args.acc4_terminal, "acc4_specialist_run", "terminal.json"
    ),
    "acc8": controller_artifact(
        args.acc8_terminal, "acc8_specialist_run", "terminal.json"
    ),
}
naf_receipt_path = controller_artifact(
    args.naf_receipt,
    "post_refiner_run",
    "receipt.json",
)
generalist = load(generalist_receipt_path)
terminals = {route: load(path) for route, path in terminal_paths.items()}
naf = load(naf_receipt_path)
admission = controller.get("final_admission")
model_sha = sha256(MODEL)

amendment = scheduler_amendment.get("amendment")
deployment_contract = scheduler_amendment.get("deployment_contract")
deployment_sources = scheduler_deployment.get("source_hashes")
preflight_sources = post_e49_preflight.get("source_hashes")
preflight_handoff = post_e49_preflight.get("handoff")
preflight_specialists = post_e49_preflight.get("specialists")
preflight_post = post_e49_preflight.get("post_refiner")
preflight_builder = post_e49_preflight.get("final_builder")
if (
    scheduler_amendment.get("schema")
    != "final-c10-single-lineage-r24-scheduler-boundary-amendment-v1"
    or scheduler_amendment.get("state") != "SEALED"
    or scheduler_amendment.get("parent", {}).get("r23_tactics_sha256")
    != sha256(TACTICS)
    or not isinstance(amendment, dict)
    or amendment.get("equal_boundary_semantics")
    != "ONE_EPOCH_LINEAR_WARMUP_WITH_ZERO_COSINE_TAIL"
    or amendment.get("r23_recipe_changed") is not False
    or amendment.get("optimizer_step_budget_changed") is not False
    or amendment.get("learning_rate_changed") is not False
    or scheduler_amendment.get("source_hashes", {}).get(
        "promptmr_production.py"
    )
    != R24_PRODUCTION_SHA256
    or not isinstance(deployment_contract, dict)
    or deployment_contract.get("active_generalist_gpu_process_must_be_preserved")
    is not True
    or deployment_contract.get("candidate_count") != 1
    or deployment_contract.get("fallback_registered") is not False
    or scheduler_deployment.get("schema")
    != "vessl-c10-single-lineage-r24-scheduler-fix-deployment-v1"
    or scheduler_deployment.get("state")
    != "ACTIVE_CURRENT_C10_PRESERVED_R24_SCHEDULER_FIX_ARMED"
    or scheduler_deployment.get("r24_amendment_sha256")
    != sha256(SCHEDULER_AMENDMENT)
    or scheduler_deployment.get("r23_tactics_sha256") != sha256(TACTICS)
    or scheduler_deployment.get("scheduler_boundary_fixed") is not True
    or scheduler_deployment.get("scheduler_cpu_preflight") != "PASS"
    or scheduler_deployment.get("trainer_signal_sent") is not False
    or scheduler_deployment.get("active_generalist_process_touched") is not False
    or scheduler_deployment.get("active_generalist_recipe_changed") is not False
    or scheduler_deployment.get("specialist_recipe_changed") is not False
    or scheduler_deployment.get("optimizer_step_budget_changed") is not False
    or scheduler_deployment.get("learning_rate_changed") is not False
    or scheduler_deployment.get("candidate_count") != 1
    or scheduler_deployment.get("fallback_registered") is not False
    or not isinstance(deployment_sources, dict)
    or deployment_sources.get("promptmr_production.py")
    != R24_PRODUCTION_SHA256
):
    raise SystemExit("R23_EVIDENCE_REFUSED: R24 scheduler evidence is invalid")

expected_specialist_flags = {
    "acc4": {
        "--promptmr-train-acceleration": "acc4",
        "--promptmr-stop-after-optimizer-steps": "2336",
        "--promptmr-specialist-lr-horizon-optimizer-steps": "2336",
        "--promptmr-specialist-loss-family": "exact_upstream_ssim",
        "--promptmr-mraugment": "conservative_immediate",
        "--lr": "1e-05",
    },
    "acc8": {
        "--promptmr-train-acceleration": "acc8",
        "--promptmr-stop-after-optimizer-steps": "1158",
        "--promptmr-specialist-lr-horizon-optimizer-steps": "2315",
        "--promptmr-specialist-loss-family": (
            "r10_image_masked_ssim_valid_windows_mean"
        ),
        "--promptmr-mraugment": "off",
        "--lr": "5e-05",
    },
}
specialist_parser_valid = isinstance(preflight_specialists, dict)
if specialist_parser_valid:
    for route, expected_flags in expected_specialist_flags.items():
        for mode in ("fresh", "resume"):
            entry = preflight_specialists.get(f"{route}_{mode}")
            flags = entry.get("flags") if isinstance(entry, dict) else None
            specialist_parser_valid = specialist_parser_valid and (
                isinstance(entry, dict)
                and entry.get("parser") == "PASS"
                and isinstance(flags, dict)
                and all(flags.get(key) == value for key, value in expected_flags.items())
                and flags.get("--promptmr-num-cascades") == "10"
                and flags.get("--promptmr-n-history") == "0"
                and flags.get("--precision") == "fp32"
                and flags.get("--batch-size") == "1"
                and flags.get("--seed") == "430"
                and flags.get("--promptmr-skip-validation") is True
            )

post_flags = preflight_post.get("flags") if isinstance(preflight_post, dict) else None
builder_flags = (
    preflight_builder.get("flags") if isinstance(preflight_builder, dict) else None
)
if (
    post_e49_preflight.get("schema")
    != "vessl-r24-post-e49-command-parser-preflight-v1"
    or post_e49_preflight.get("state") != "PASS"
    or post_e49_preflight.get("cpu_only") is not True
    or post_e49_preflight.get("cuda_initialized") is not False
    or post_e49_preflight.get("remote_process_read") is not False
    or post_e49_preflight.get("remote_process_changed") is not False
    or post_e49_preflight.get("active_generalist_process_touched") is not False
    or post_e49_preflight.get("recipe_changed") is not False
    or post_e49_preflight.get("candidate_count") != 1
    or post_e49_preflight.get("fallback_registered") is not False
    or post_e49_preflight.get("external_learned_state_imported") is not False
    or post_e49_preflight.get("leaderboard_data_used") is not False
    or post_e49_preflight.get("official_evaluation_executed") is not False
    or post_e49_preflight.get("chain")
    != [
        "E49_HASH_SEALED_GENERALIST",
        "ACC4_MODEL_ONLY_SPECIALIST",
        "ACC8_MODEL_ONLY_R10_SPECIALIST",
        "FROZEN_ROUTED_BASE_PLUS_NAF_S",
        "SINGLE_ROUTED_BEST_MODEL_PT",
    ]
    or preflight_handoff
    != {
        "epoch": 49,
        "optimizer_step": 228928,
        "scheduler_horizon_epoch": 51,
        "scheduler_horizon_optimizer_step": 238272,
    }
    or not isinstance(preflight_sources, dict)
    or preflight_sources.get("final-tactics-c10-r23-e49.json") != sha256(TACTICS)
    or preflight_sources.get(
        "FINAL_C10_SINGLE_LINEAGE_R24_SCHEDULER_BOUNDARY.json"
    )
    != sha256(SCHEDULER_AMENDMENT)
    or preflight_sources.get("r24-deployment-receipt.json")
    != sha256(args.scheduler_deployment_receipt)
    or preflight_sources.get("controller.py") != R24_CONTROLLER_SHA256
    or preflight_sources.get("train.py") != R24_TRAIN_SHA256
    or preflight_sources.get("promptmr_production.py") != R24_PRODUCTION_SHA256
    or preflight_sources.get("vessl_train_post_refiner.py")
    != R23_POST_TRAINER_SHA256
    or preflight_sources.get("vessl_build_routed_promptmr_checkpoint.py")
    != R23_BUILDER_SHA256
    or not specialist_parser_valid
    or not isinstance(preflight_post, dict)
    or preflight_post.get("parser") != "PASS"
    or preflight_post.get("main_c10_frozen") is not True
    or preflight_post.get("optimizer_scope") != "naf_s_only"
    or preflight_post.get("bbox_loss_coefficient") != 0.5
    or not isinstance(post_flags, dict)
    or post_flags.get("--variant") != "NAF_S"
    or post_flags.get("--views") != ["identity", "flip_lr"]
    or post_flags.get("--epochs") != "21"
    or post_flags.get("--optimizer-steps") != "93567"
    or post_flags.get("--peak-lr") != "0.0001"
    or post_flags.get("--seed") != "430"
    or not isinstance(preflight_builder, dict)
    or preflight_builder.get("parser") != "PASS"
    or preflight_builder.get("candidate_count") != 1
    or not isinstance(builder_flags, dict)
    or builder_flags.get("--tta-views") != "acc8_identity_flip_lr"
    or builder_flags.get("--output") != "/root/result/final/best_model.pt"
):
    raise SystemExit("R23_EVIDENCE_REFUSED: post-E49 command preflight is invalid")

if (
    tactics_envelope.get("state") != "SEALED"
    or not isinstance(tactics, dict)
    or controller.get("schema") != "vessl-g10-architecture-dispatcher-receipt-v1"
    or controller.get("state") != "PASS"
    or controller.get("winner") != "R2_C10"
    or controller.get("final_checkpoint_sha256") != model_sha
    or controller.get("final_admission_mode") != "PRIMARY_REQUESTED"
    or controller.get("tta_views") != "acc8_identity_flip_lr"
    or controller.get("acc4_specialist_status") != "TRAINED_ON_VESSL_AND_ROUTED"
    or controller.get("acc8_specialist_status") != "TRAINED_ON_VESSL_AND_ROUTED"
    or controller.get("post_refiner_status") != "TRAINED_ON_VESSL_AND_PACKAGED"
    or controller.get("acc4_specialist_checkpoint_sha256") != source_hashes["acc4"]
    or controller.get("acc8_specialist_checkpoint_sha256") != source_hashes["acc8"]
    or controller.get("post_refiner_checkpoint_sha256")
    != source_hashes["post_refiner"]
    or controller.get("fallback_checkpoint") is not None
    or controller.get("fallback_checkpoint_sha256") is not None
    or controller.get("leaderboard_data_read") is not False
    or controller.get("external_learned_state_imported") is not False
    or controller.get("all_final_learned_state_vessl_only") is not True
    or not isinstance(admission, dict)
):
    raise SystemExit("R23_EVIDENCE_REFUSED: controller final receipt is not exact R23")

if (
    admission.get("schema") != "vessl-final-lazy-router-admission-v2"
    or admission.get("state") != "PASS"
    or admission.get("final_checkpoint_sha256") != model_sha
    or admission.get("gpu") != "NVIDIA GeForce GTX 1080"
    or admission.get("leaderboard_data_used") is not False
    or admission.get("official_reconstruction_path") is not True
):
    raise SystemExit("R23_EVIDENCE_REFUSED: final admission is not a VESSL PASS")

if (
    generalist.get("schema") != "vessl-g10-generalist-terminal-checkpoint-v1"
    or generalist.get("state") != "SEALED"
    or generalist.get("checkpoint_sha256") != source_hashes["generalist"]
    or generalist.get("epoch") != 49
    or generalist.get("optimizer_step") != 228928
    or generalist.get("scheduler_horizon_epochs") != 51
    or generalist.get("scheduler_total_steps") != 238272
):
    raise SystemExit("R23_EVIDENCE_REFUSED: generalist receipt is not exact E49")

specialist_receipts = {}
for route in ("acc4", "acc8"):
    recipe = tactics.get(f"{route}_specialist")
    terminal = terminals[route]
    expected_steps = 2336 if route == "acc4" else 1158
    expected_horizon = 2336 if route == "acc4" else 2315
    checkpoint_path = controller.get(f"{route}_specialist_checkpoint")
    if (
        not isinstance(recipe, dict)
        or recipe.get("enabled") is not True
        or recipe.get("late_branch_parent_epoch") != 49
        or recipe.get("late_branch_parent_optimizer_step") != 228928
        or recipe.get("optimizer_steps") != expected_steps
        or recipe.get("lr_horizon_optimizer_steps") != expected_horizon
        or terminal.get("status") != "COMPLETED"
        or terminal.get("epoch") != 0
        or terminal.get("step") != expected_steps
        or terminal.get("exact_optimizer_step_budget") is not True
        or terminal.get("checkpoint") != checkpoint_path
    ):
        raise SystemExit(f"R23_EVIDENCE_REFUSED: {route} terminal is invalid")
    specialist_receipts[route] = {
        "schema": "fastmri-r23-specialist-receipt-v1",
        "state": "PASS",
        "route": route,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": source_hashes[route],
        "parent_checkpoint_sha256": source_hashes["generalist"],
        "parent_epoch": 49,
        "parent_optimizer_step": 228928,
        "source_epoch": 0,
        "optimizer_steps": expected_steps,
        "lr_horizon_optimizer_steps": expected_horizon,
        "peak_lr": recipe["peak_lr"],
        "loss_family": recipe["loss_family"],
        "mraugment": recipe["mraugment"],
        "training_pool": (
            "organizer_train_acc4"
            if route == "acc4"
            else "organizer_real_acc8_only"
        ),
        "validation_forward_count": 0,
        "trained_on_vessl": True,
        "external_learned_state_imported": False,
        "leaderboard_data_used": False,
        "source_terminal_sha256": sha256(
            terminal_paths[route]
        ),
        "controller_receipt_sha256": sha256(args.controller_receipt),
    }

if (
    naf.get("schema") != "vessl-post-refiner-training-receipt-v1"
    or naf.get("state") != "PASS"
    or naf.get("checkpoint_sha256") != source_hashes["post_refiner"]
    or naf.get("base_checkpoint_sha256") != source_hashes["generalist"]
    or naf.get("routed_branch_sha256")
    != {"acc4": source_hashes["acc4"], "acc8": source_hashes["acc8"]}
    or naf.get("variant") != "NAF_S"
    or naf.get("epochs") != 21
    or naf.get("parent_epoch") != 49
    or naf.get("optimizer_steps") != 93567
):
    raise SystemExit("R23_EVIDENCE_REFUSED: NAF_S receipt is invalid")

policy = tactics.get("candidate_policy")
augmentation = tactics.get("training_mask_augmentation")
inference = tactics.get("inference")
if not all(isinstance(value, dict) for value in (policy, augmentation, inference)):
    raise SystemExit("R23_EVIDENCE_REFUSED: tactics policy is absent")
policy_receipt = {
    "schema": "fastmri-r23-policy-receipt-v1",
    "state": "PASS",
    "candidate_count": 1,
    "final_package_count": policy.get("final_package_count"),
    "fallback_registered": policy.get("fallback_registered"),
    "official_evaluation_max_runs": policy.get("official_evaluation_max_runs"),
    "routing_input": "input_kspace_mask_only",
    "routing_features": ["mask_density", "acs_width", "period_residue", "offset"],
    "unknown_or_mismatch_route": "generalist",
    "routing_forbidden_inputs": [
        "filename", "image", "bbox", "target", "leaderboard_result"
    ],
    "augmentation_schema": augmentation.get("schema"),
    "augmentation_inference_enabled": augmentation.get("inference_enabled"),
    "official_mask_unchanged": augmentation.get("official_mask_unchanged"),
    "validation_used_for_checkpoint_selection": False,
    "learned_state_source": "VESSL_ONLY",
    "external_learned_state_imported": False,
    "leaderboard_data_used_for_training_or_selection": False,
    "all_reconstruction_inside_recon_slice": inference.get(
        "all_reconstruction_inside_recon_slice"
    ),
    "tactics_sha256": sha256(TACTICS),
    "controller_receipt_sha256": sha256(args.controller_receipt),
    "scheduler_deployment_receipt_sha256": sha256(
        args.scheduler_deployment_receipt
    ),
    "post_e49_command_parser_preflight_sha256": sha256(POST_E49_PREFLIGHT),
}

if (
    policy_receipt["final_package_count"] != 1
    or policy_receipt["fallback_registered"] is not False
    or policy_receipt["official_evaluation_max_runs"] != 1
    or policy_receipt["augmentation_schema"]
    != "acc4-to-acc8-pair-mask-augmentation-v1"
    or policy_receipt["augmentation_inference_enabled"] is not False
    or policy_receipt["official_mask_unchanged"] is not True
    or policy_receipt["all_reconstruction_inside_recon_slice"] is not True
):
    raise SystemExit("R23_EVIDENCE_REFUSED: normalized policy is invalid")

EVIDENCE.mkdir(parents=True, exist_ok=True)
payloads = {
    outputs["generalist"]: generalist,
    outputs["acc4"]: specialist_receipts["acc4"],
    outputs["acc8"]: specialist_receipts["acc8"],
    outputs["naf"]: naf,
    outputs["policy"]: policy_receipt,
    outputs["admission"]: admission,
    outputs["scheduler"]: scheduler_deployment,
}
created = []
try:
    for path, value in payloads.items():
        atomic_json(path, value)
        created.append(path)
except Exception:
    for path in created:
        path.unlink(missing_ok=True)
    raise

print(
    json.dumps(
        {
            "state": "R23_EVIDENCE_MATERIALIZED",
            "best_model_sha256": model_sha,
            "files": {
                path.relative_to(ROOT).as_posix(): sha256(path)
                for path in payloads
            },
        },
        sort_keys=True,
    )
)
