#!/usr/bin/env python3
"""Fail-closed verifier for the one R25 single-lineage package."""

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
    "assemble_final_package.py",
    "materialize_r23_evidence.py",
    "seal_package.py",
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
    "reproduction/FINAL_C10_SINGLE_LINEAGE_R24_SCHEDULER_BOUNDARY.json",
    "reproduction/FINAL_C10_SINGLE_LINEAGE_R25_ACC4_E2_NAF_TAIL.json",
    "reproduction/R25_POST_E49_COMMAND_PARSER_PREFLIGHT.json",
    "reproduction/R25_CPU_PREFLIGHT.json",
    "reproduction/final-tactics-c10-r23-e49.json",
    "reproduction/final-tactics-c10-r25-acc4-e2-naf-tail.json",
    "reproduction/controller.py",
    "reproduction/generalist/train.py",
    "reproduction/generalist/promptmr_production.py",
    "reproduction/specialist/train.py",
    "reproduction/specialist/promptmr_production.py",
    "reproduction/vessl_train_post_refiner.py",
    "reproduction/vessl_build_routed_promptmr_checkpoint.py",
    "reproduction/organizer-data-provenance.json",
    "reproduction/inference-source-snapshot-manifest.json",
    "reproduction/source-sha256sums.txt",
    "evidence/assembly-receipt.json",
    "evidence/raw/controller-final-receipt.json",
    "evidence/raw/acc4-terminal.json",
    "evidence/raw/acc8-terminal.json",
    "evidence/generalist-e49-receipt.json",
    "evidence/acc4-specialist-receipt.json",
    "evidence/acc8-specialist-receipt.json",
    "evidence/naf-s-training-receipt.json",
    "evidence/policy-receipt.json",
    "evidence/inference-admission-receipt.json",
    "evidence/scheduler-amendment-deployment-receipt.json",
    "evidence/r25-amendment-deployment-receipt.json",
)
LEARNED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
R24_PRODUCTION_SHA256 = (
    "ea4695f5fada7c417323d9efad495544d0743ad1d35b3023c1a645a421d8688b"
)
R25_PRODUCTION_SHA256 = (
    "ec86306162dae4da087f0033beec924d2368e242a2df4a4d3e4607f399e0de2b"
)
R25_CONTROLLER_SHA256 = (
    "78718ea26e1049546627710c515a7e5d7dd78c11c6ccd8b06b0075de28048b55"
)
R25_TRAIN_SHA256 = (
    "ae559aea9d13a35af00c3fc9c37d70471b273f740577d6056fa17728d3ab950b"
)
R25_POST_TRAINER_SHA256 = (
    "399410e2e4b9155a95d5cf0bef344d896bab7e4e9c140c0e8abb7475cd405e17"
)
R25_BUILDER_SHA256 = (
    "cf9aafb064d9e4aa2b04f35f7faf293970ea2f055059780950e91d213c42c8e7"
)


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
        fail("best_model.pt outer routed contract is not exact R25")

    expected = {
        "generalist": (None, "generalist", "all", 228928, 49),
        "acc4": (4, "acc4", "acc4", 4672, 1),
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
            fail(f"best_model.pt {name} component is not exact R25")
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
        or post.get("optimizer_steps") != 91231
        or post.get("lr_horizon_optimizer_steps") != 93567
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
        or not isinstance(post.get("source_checkpoint_sha256"), str)
        or len(post["source_checkpoint_sha256"]) != 64
        or not isinstance(post.get("post_refiner_state"), dict)
    ):
        fail("best_model.pt NAF_S component is not exact frozen-C10 R25")
    source_hashes["post_refiner"] = post["source_checkpoint_sha256"]
    forbidden = {"optimizer", "scheduler", "rng_state", "ema", "swa", "scaler"}
    if forbidden.intersection(value):
        fail("best_model.pt contains forbidden training state")
    return source_hashes


def json_object(relative: str) -> dict:
    try:
        value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"invalid JSON evidence {relative}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON evidence is not an object: {relative}")
    return value


def verify_evidence_receipts(
    source_hashes: dict[str, str], model_sha256: str
) -> None:
    parent_tactics_path = ROOT / "reproduction/final-tactics-c10-r23-e49.json"
    tactics_path = (
        ROOT / "reproduction/final-tactics-c10-r25-acc4-e2-naf-tail.json"
    )
    amendment_path = (
        ROOT / "reproduction/FINAL_C10_SINGLE_LINEAGE_R24_SCHEDULER_BOUNDARY.json"
    )
    r25_amendment_path = (
        ROOT / "reproduction/FINAL_C10_SINGLE_LINEAGE_R25_ACC4_E2_NAF_TAIL.json"
    )
    amendment_envelope = json_object(
        "reproduction/FINAL_C10_SINGLE_LINEAGE_R24_SCHEDULER_BOUNDARY.json"
    )
    amendment = amendment_envelope.get("amendment")
    deployment_contract = amendment_envelope.get("deployment_contract")
    scheduler = json_object(
        "evidence/scheduler-amendment-deployment-receipt.json"
    )
    r25_amendment = json_object(
        "reproduction/FINAL_C10_SINGLE_LINEAGE_R25_ACC4_E2_NAF_TAIL.json"
    )
    r25_deployment_path = (
        ROOT / "evidence/r25-amendment-deployment-receipt.json"
    )
    r25_deployment = json_object(
        "evidence/r25-amendment-deployment-receipt.json"
    )
    r25_cpu_preflight_path = ROOT / "reproduction/R25_CPU_PREFLIGHT.json"
    r25_cpu_preflight = json_object("reproduction/R25_CPU_PREFLIGHT.json")
    post_e49_preflight_path = (
        ROOT / "reproduction/R25_POST_E49_COMMAND_PARSER_PREFLIGHT.json"
    )
    post_e49_preflight = json_object(
        "reproduction/R25_POST_E49_COMMAND_PARSER_PREFLIGHT.json"
    )
    scheduler_sources = scheduler.get("source_hashes")
    if (
        amendment_envelope.get("schema")
        != "final-c10-single-lineage-r24-scheduler-boundary-amendment-v1"
        or amendment_envelope.get("state") != "SEALED"
        or amendment_envelope.get("parent", {}).get("r23_tactics_sha256")
        != sha256(parent_tactics_path)
        or not isinstance(amendment, dict)
        or amendment.get("equal_boundary_semantics")
        != "ONE_EPOCH_LINEAR_WARMUP_WITH_ZERO_COSINE_TAIL"
        or amendment.get("r23_recipe_changed") is not False
        or amendment.get("optimizer_step_budget_changed") is not False
        or amendment.get("learning_rate_changed") is not False
        or amendment_envelope.get("source_hashes", {}).get(
            "promptmr_production.py"
        )
        != R24_PRODUCTION_SHA256
        or not isinstance(deployment_contract, dict)
        or deployment_contract.get(
            "active_generalist_gpu_process_must_be_preserved"
        )
        is not True
        or deployment_contract.get("candidate_count") != 1
        or deployment_contract.get("fallback_registered") is not False
        or scheduler.get("schema")
        != "vessl-c10-single-lineage-r24-scheduler-fix-deployment-v1"
        or scheduler.get("state")
        != "ACTIVE_CURRENT_C10_PRESERVED_R24_SCHEDULER_FIX_ARMED"
        or scheduler.get("r24_amendment_sha256") != sha256(amendment_path)
        or scheduler.get("r23_tactics_sha256") != sha256(parent_tactics_path)
        or scheduler.get("scheduler_boundary_fixed") is not True
        or scheduler.get("scheduler_cpu_preflight") != "PASS"
        or scheduler.get("trainer_signal_sent") is not False
        or scheduler.get("active_generalist_process_touched") is not False
        or scheduler.get("active_generalist_recipe_changed") is not False
        or scheduler.get("specialist_recipe_changed") is not False
        or scheduler.get("optimizer_step_budget_changed") is not False
        or scheduler.get("learning_rate_changed") is not False
        or scheduler.get("candidate_count") != 1
        or scheduler.get("final_package_count") != 1
        or scheduler.get("fallback_registered") is not False
        or scheduler.get("official_evaluation_max_runs") != 1
        or not isinstance(scheduler_sources, dict)
        or scheduler_sources.get("promptmr_production.py")
        != R24_PRODUCTION_SHA256
    ):
        fail("R24 parent scheduler amendment/deployment evidence is invalid")

    r25_sources = r25_deployment.get("source_hashes")
    r25_preflight_hashes = r25_deployment.get("preflight_hashes")
    if (
        r25_amendment.get("schema")
        != "final-c10-single-lineage-r25-acc4-e2-naf-tail-amendment-v1"
        or r25_amendment.get("state") != "SEALED"
        or r25_amendment.get("parent", {}).get("r23_tactics_sha256")
        != sha256(parent_tactics_path)
        or r25_amendment.get("parent", {}).get(
            "r24_scheduler_amendment_sha256"
        )
        != sha256(amendment_path)
        or r25_amendment.get("amendment", {}).get(
            "post_e49_total_optimizer_steps"
        )
        != 97_061
        or r25_deployment.get("schema")
        != "vessl-c10-single-lineage-r25-acc4-e2-naf-tail-deployment-v1"
        or r25_deployment.get("state")
        != "ACTIVE_CURRENT_C10_PRESERVED_R25_ACC4_E2_NAF_TAIL_ARMED"
        or r25_deployment.get("parent_r24_deployment_receipt_sha256")
        != sha256(ROOT / "evidence/scheduler-amendment-deployment-receipt.json")
        or r25_deployment.get("r24_amendment_sha256") != sha256(amendment_path)
        or r25_deployment.get("r25_amendment_sha256")
        != sha256(r25_amendment_path)
        or r25_deployment.get("r25_tactics_sha256") != sha256(tactics_path)
        or r25_deployment.get("trainer_signal_sent") is not False
        or r25_deployment.get("active_generalist_process_touched") is not False
        or r25_deployment.get("active_generalist_recipe_changed") is not False
        or r25_deployment.get("future_specialist_recipe_changed") is not True
        or r25_deployment.get("post_e49_optimizer_step_budget_changed") is not False
        or r25_deployment.get("post_e49_optimizer_steps") != 97_061
        or r25_deployment.get("candidate_count") != 1
        or r25_deployment.get("final_package_count") != 1
        or r25_deployment.get("fallback_registered") is not False
        or r25_deployment.get("official_evaluation_max_runs") != 1
        or r25_deployment.get("cpu_preflight") != "PASS"
        or not isinstance(r25_sources, dict)
        or r25_sources.get("controller.py") != R25_CONTROLLER_SHA256
        or r25_sources.get("train.py") != R25_TRAIN_SHA256
        or r25_sources.get("promptmr_production.py") != R25_PRODUCTION_SHA256
        or r25_sources.get("vessl_train_post_refiner.py")
        != R25_POST_TRAINER_SHA256
        or r25_sources.get("vessl_build_routed_promptmr_checkpoint.py")
        != R25_BUILDER_SHA256
        or sha256(ROOT / "reproduction/controller.py")
        != R25_CONTROLLER_SHA256
        or sha256(ROOT / "reproduction/specialist/train.py")
        != R25_TRAIN_SHA256
        or sha256(ROOT / "reproduction/specialist/promptmr_production.py")
        != R25_PRODUCTION_SHA256
        or sha256(ROOT / "reproduction/vessl_train_post_refiner.py")
        != R25_POST_TRAINER_SHA256
        or sha256(
            ROOT / "reproduction/vessl_build_routed_promptmr_checkpoint.py"
        )
        != R25_BUILDER_SHA256
        or not isinstance(r25_preflight_hashes, dict)
        or r25_preflight_hashes.get("command_parser")
        != sha256(post_e49_preflight_path)
        or r25_preflight_hashes.get("cpu_contract")
        != sha256(r25_cpu_preflight_path)
        or r25_cpu_preflight.get("state") != "PASS"
        or r25_cpu_preflight.get("post_e49_optimizer_steps") != 97_061
        or r25_cpu_preflight.get("tactics_sha256") != sha256(tactics_path)
        or r25_cpu_preflight.get("amendment_sha256")
        != sha256(r25_amendment_path)
        or r25_cpu_preflight.get("command_preflight_sha256")
        != sha256(post_e49_preflight_path)
    ):
        fail("R25 contract/deployment/CPU-preflight evidence is invalid")

    assembly = json_object("evidence/assembly-receipt.json")
    raw_controller_path = ROOT / "evidence/raw/controller-final-receipt.json"
    raw_controller = json_object("evidence/raw/controller-final-receipt.json")
    snapshot_manifest_path = (
        ROOT / "reproduction/inference-source-snapshot-manifest.json"
    )
    snapshot_manifest = json_object(
        "reproduction/inference-source-snapshot-manifest.json"
    )
    snapshot_files = snapshot_manifest.get("files")
    snapshot_destinations = assembly.get("snapshot_destinations")
    if (
        assembly.get("schema")
        != "vessl-r25-single-final-package-assembly-v1"
        or assembly.get("state") != "PASS"
        or assembly.get("candidate_count") != 1
        or assembly.get("fallback_registered") is not False
        or assembly.get("controller_receipt_sha256")
        != sha256(raw_controller_path)
        or assembly.get("r24_deployment_receipt_sha256")
        != sha256(ROOT / "evidence/scheduler-amendment-deployment-receipt.json")
        or assembly.get("r25_deployment_receipt_sha256")
        != sha256(r25_deployment_path)
        or assembly.get("final_checkpoint_sha256") != model_sha256
        or assembly.get("inference_source_snapshot_manifest_sha256")
        != sha256(snapshot_manifest_path)
        or raw_controller.get("schema")
        != "vessl-g10-architecture-dispatcher-receipt-v1"
        or raw_controller.get("state") != "PASS"
        or raw_controller.get("winner") != "R2_C10"
        or raw_controller.get("final_checkpoint_sha256") != model_sha256
        or raw_controller.get("final_admission_mode") != "PRIMARY_REQUESTED"
        or raw_controller.get("fallback_checkpoint") is not None
        or raw_controller.get("fallback_checkpoint_sha256") is not None
        or snapshot_manifest.get("schema")
        != "vessl-final-inference-source-snapshot-v1"
        or snapshot_manifest.get("state") != "SEALED"
        or snapshot_manifest.get("leaderboard_data_read") is not False
        or not isinstance(snapshot_files, list)
        or not snapshot_files
        or not isinstance(snapshot_destinations, dict)
    ):
        fail("single-final assembly/source-snapshot evidence is invalid")
    for item in snapshot_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            fail("source snapshot file record is invalid")
        source_relative = item["path"]
        destination_relative = snapshot_destinations.get(source_relative)
        if not isinstance(destination_relative, str):
            fail(f"source snapshot destination is absent: {source_relative}")
        destination = ROOT / "project" / destination_relative
        if (
            not destination.is_file()
            or sha256(destination) != item.get("sha256")
            or destination.stat().st_size != int(item.get("size_bytes", -1))
        ):
            fail(f"source snapshot destination mismatch: {source_relative}")

    preflight_sources = post_e49_preflight.get("source_hashes")
    preflight_handoff = post_e49_preflight.get("handoff")
    preflight_specialists = post_e49_preflight.get("specialists")
    preflight_post = post_e49_preflight.get("post_refiner")
    preflight_builder = post_e49_preflight.get("final_builder")
    expected_specialist_flags = {
        "acc4": {
            "--promptmr-train-acceleration": "acc4",
            "--promptmr-stop-after-optimizer-steps": "4672",
            "--promptmr-specialist-lr-horizon-optimizer-steps": "35040",
            "--promptmr-specialist-loss-family": "exact_upstream_ssim",
            "--promptmr-mraugment": "conservative_immediate",
            "--lr": "2.5e-05",
            "--num-epochs": "2",
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
                    and all(
                        flags.get(key) == value
                        for key, value in expected_flags.items()
                    )
                    and flags.get("--promptmr-num-cascades") == "10"
                    and flags.get("--promptmr-n-history") == "0"
                    and flags.get("--precision") == "fp32"
                    and flags.get("--batch-size") == "1"
                    and flags.get("--seed") == "430"
                    and flags.get("--promptmr-skip-validation") is True
                )

    post_flags = (
        preflight_post.get("flags") if isinstance(preflight_post, dict) else None
    )
    builder_flags = (
        preflight_builder.get("flags")
        if isinstance(preflight_builder, dict)
        else None
    )
    if (
        post_e49_preflight.get("schema")
        != "vessl-r25-post-e49-command-parser-preflight-v1"
        or post_e49_preflight.get("state") != "PASS"
        or post_e49_preflight.get("cpu_only") is not True
        or post_e49_preflight.get("cuda_initialized") is not False
        or post_e49_preflight.get("remote_process_read") is not False
        or post_e49_preflight.get("remote_process_changed") is not False
        or post_e49_preflight.get("active_generalist_process_touched") is not False
        or post_e49_preflight.get("recipe_changed") is not True
        or post_e49_preflight.get("active_generalist_recipe_changed") is not False
        or post_e49_preflight.get("post_e49_optimizer_steps") != 97_061
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
        or preflight_sources.get("final-tactics-c10-r25-acc4-e2-naf-tail.json")
        != sha256(tactics_path)
        or preflight_sources.get(
            "FINAL_C10_SINGLE_LINEAGE_R25_ACC4_E2_NAF_TAIL.json"
        )
        != sha256(r25_amendment_path)
        or preflight_sources.get("r24-deployment-receipt.json")
        != sha256(ROOT / "evidence/scheduler-amendment-deployment-receipt.json")
        or preflight_sources.get("controller.py") != R25_CONTROLLER_SHA256
        or preflight_sources.get("train.py") != R25_TRAIN_SHA256
        or preflight_sources.get("promptmr_production.py")
        != R25_PRODUCTION_SHA256
        or preflight_sources.get("vessl_train_post_refiner.py")
        != R25_POST_TRAINER_SHA256
        or preflight_sources.get("vessl_build_routed_promptmr_checkpoint.py")
        != R25_BUILDER_SHA256
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
        or post_flags.get("--optimizer-steps") != "91231"
        or post_flags.get("--lr-horizon-optimizer-steps") != "93567"
        or post_flags.get("--peak-lr") != "0.0001"
        or post_flags.get("--seed") != "430"
        or not isinstance(preflight_builder, dict)
        or preflight_builder.get("parser") != "PASS"
        or preflight_builder.get("candidate_count") != 1
        or not isinstance(builder_flags, dict)
        or builder_flags.get("--tta-views") != "acc8_identity_flip_lr"
        or builder_flags.get("--output") != "/root/result/final/best_model.pt"
    ):
        diagnostic = {
            "schema": post_e49_preflight.get("schema"),
            "state": post_e49_preflight.get("state"),
            "cpu_only": post_e49_preflight.get("cpu_only"),
            "cuda_initialized": post_e49_preflight.get("cuda_initialized"),
            "remote_process_read": post_e49_preflight.get("remote_process_read"),
            "remote_process_changed": post_e49_preflight.get("remote_process_changed"),
            "active_generalist_process_touched": post_e49_preflight.get(
                "active_generalist_process_touched"
            ),
            "recipe_changed": post_e49_preflight.get("recipe_changed"),
            "candidate_count": post_e49_preflight.get("candidate_count"),
            "fallback_registered": post_e49_preflight.get(
                "fallback_registered"
            ),
            "external_learned_state_imported": post_e49_preflight.get(
                "external_learned_state_imported"
            ),
            "leaderboard_data_used": post_e49_preflight.get(
                "leaderboard_data_used"
            ),
            "official_evaluation_executed": post_e49_preflight.get(
                "official_evaluation_executed"
            ),
            "chain": post_e49_preflight.get("chain"),
            "handoff": preflight_handoff,
            "source_hashes_observed": preflight_sources,
            "source_hashes_expected": {
                "final-tactics-c10-r25-acc4-e2-naf-tail.json": sha256(
                    tactics_path
                ),
                "FINAL_C10_SINGLE_LINEAGE_R25_ACC4_E2_NAF_TAIL.json": sha256(
                    r25_amendment_path
                ),
                "r24-deployment-receipt.json": sha256(
                    ROOT / "evidence/scheduler-amendment-deployment-receipt.json"
                ),
                "controller.py": R25_CONTROLLER_SHA256,
                "train.py": R25_TRAIN_SHA256,
                "promptmr_production.py": R25_PRODUCTION_SHA256,
                "vessl_train_post_refiner.py": R25_POST_TRAINER_SHA256,
                "vessl_build_routed_promptmr_checkpoint.py": R25_BUILDER_SHA256,
            },
            "specialist_parser_valid": specialist_parser_valid,
            "post_refiner": preflight_post,
            "final_builder": preflight_builder,
        }
        fail(
            "post-E49 command/parser preflight receipt is invalid: "
            + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
        )

    generalist = json_object("evidence/generalist-e49-receipt.json")
    if (
        generalist.get("schema") != "vessl-g10-generalist-terminal-checkpoint-v1"
        or generalist.get("state") != "SEALED"
        or generalist.get("checkpoint_sha256") != source_hashes["generalist"]
        or generalist.get("epoch") != 49
        or generalist.get("optimizer_step") != 228928
        or generalist.get("scheduler_horizon_epochs") != 51
        or generalist.get("scheduler_total_steps") != 238272
        or generalist.get("leaderboard_data_used") is not False
        or generalist.get("external_learned_state_imported") is not False
    ):
        fail("E49 generalist evidence receipt is invalid")

    specialist_contracts = {
        "acc4": {
            "optimizer_steps": 4672,
            "lr_horizon_optimizer_steps": 35040,
            "peak_lr": 0.000025,
            "source_epoch": 1,
            "loss_family": "exact_upstream_ssim",
            "mraugment": "conservative_immediate",
            "training_pool": "organizer_train_acc4",
        },
        "acc8": {
            "optimizer_steps": 1158,
            "lr_horizon_optimizer_steps": 2315,
            "peak_lr": 0.00005,
            "source_epoch": 0,
            "loss_family": "r10_image_masked_ssim_valid_windows_mean",
            "mraugment": "off",
            "training_pool": "organizer_real_acc8_only",
        },
    }
    for route, expected in specialist_contracts.items():
        receipt = json_object(f"evidence/{route}-specialist-receipt.json")
        raw_terminal_path = ROOT / f"evidence/raw/{route}-terminal.json"
        if (
            receipt.get("schema") != "fastmri-r25-specialist-receipt-v1"
            or receipt.get("state") != "PASS"
            or receipt.get("route") != route
            or receipt.get("checkpoint_sha256") != source_hashes[route]
            or receipt.get("parent_checkpoint_sha256")
            != source_hashes["generalist"]
            or receipt.get("parent_epoch") != 49
            or receipt.get("parent_optimizer_step") != 228928
            or receipt.get("source_epoch") != expected["source_epoch"]
            or receipt.get("optimizer_steps") != expected["optimizer_steps"]
            or receipt.get("lr_horizon_optimizer_steps")
            != expected["lr_horizon_optimizer_steps"]
            or receipt.get("peak_lr") != expected["peak_lr"]
            or receipt.get("loss_family") != expected["loss_family"]
            or receipt.get("mraugment") != expected["mraugment"]
            or receipt.get("training_pool") != expected["training_pool"]
            or receipt.get("validation_forward_count") != 0
            or receipt.get("trained_on_vessl") is not True
            or receipt.get("external_learned_state_imported") is not False
            or receipt.get("leaderboard_data_used") is not False
            or receipt.get("source_terminal_sha256")
            != sha256(raw_terminal_path)
        ):
            fail(f"{route.upper()} specialist evidence receipt is invalid")

    naf = json_object("evidence/naf-s-training-receipt.json")
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
        or naf.get("optimizer_steps") != 91231
        or naf.get("lr_horizon_optimizer_steps") != 93567
        or naf.get("loss_family")
        != "winner_foreground_ssim_l1_sqrt_area_plus_official384_bbox05_v2"
        or naf.get("bbox_loss_coefficient") != 0.5
        or naf.get("training_data") != "organizer_train_plus_val_final"
        or naf.get("validation_used_for_checkpoint_selection") is not False
        or naf.get("leaderboard_data_read") is not False
        or naf.get("external_learned_state_imported") is not False
    ):
        fail("NAF_S training evidence receipt is invalid")

    policy = json_object("evidence/policy-receipt.json")
    if (
        policy.get("schema") != "fastmri-r25-policy-receipt-v1"
        or policy.get("state") != "PASS"
        or policy.get("candidate_count") != 1
        or policy.get("final_package_count") != 1
        or policy.get("fallback_registered") is not False
        or policy.get("official_evaluation_max_runs") != 1
        or policy.get("routing_input") != "input_kspace_mask_only"
        or policy.get("routing_features")
        != ["mask_density", "acs_width", "period_residue", "offset"]
        or policy.get("unknown_or_mismatch_route") != "generalist"
        or policy.get("routing_forbidden_inputs")
        != ["filename", "image", "bbox", "target", "leaderboard_result"]
        or policy.get("augmentation_schema")
        != "acc4-to-acc8-pair-mask-augmentation-v1"
        or policy.get("augmentation_inference_enabled") is not False
        or policy.get("official_mask_unchanged") is not True
        or policy.get("validation_used_for_checkpoint_selection") is not False
        or policy.get("learned_state_source") != "VESSL_ONLY"
        or policy.get("external_learned_state_imported") is not False
        or policy.get("leaderboard_data_used_for_training_or_selection") is not False
        or policy.get("controller_receipt_sha256")
        != sha256(raw_controller_path)
        or policy.get("scheduler_deployment_receipt_sha256")
        != sha256(
            ROOT / "evidence/scheduler-amendment-deployment-receipt.json"
        )
        or policy.get("r25_deployment_receipt_sha256")
        != sha256(r25_deployment_path)
        or policy.get("r25_amendment_sha256") != sha256(r25_amendment_path)
        or policy.get("post_e49_command_parser_preflight_sha256")
        != sha256(post_e49_preflight_path)
    ):
        fail("augmentation/dispatch/single-candidate policy receipt is invalid")

    admission = json_object("evidence/inference-admission-receipt.json")
    runtime = admission.get("runtime_contract")
    records = admission.get("records")
    if (
        admission.get("schema") != "vessl-final-lazy-router-admission-v2"
        or admission.get("state") != "PASS"
        or admission.get("final_checkpoint_sha256") != model_sha256
        or admission.get("gpu") != "NVIDIA GeForce GTX 1080"
        or admission.get("memory_limit_mib") != 8192
        or not isinstance(admission.get("nvml_exclusive_device_peak_mib"), (int, float))
        or float(admission["nvml_exclusive_device_peak_mib"]) > 8192
        or admission.get("leaderboard_data_used") is not False
        or admission.get("official_reconstruction_path") is not True
        or not isinstance(runtime, dict)
        or runtime.get("training_frame_height") != 384
        or runtime.get("sensitivity_coil_batch_size") != 8
        or runtime.get("alignment_inside_timed_recon_slice") is not True
        or not isinstance(records, dict)
        or set(records) != {"acc4", "acc8"}
    ):
        fail("GTX1080 inference admission receipt is invalid")
    for route in ("acc4", "acc8"):
        canaries = records[route].get("stress_canaries")
        if (
            not isinstance(canaries, list)
            or not canaries
            or any(item.get("finite") is not True for item in canaries)
        ):
            fail(f"{route.upper()} max-shape inference canary is invalid")


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
parser.add_argument(
    "--evaluation-in-progress",
    action="store_true",
    help="allow only the sealed attempt-1 start marker and its live log",
)
args = parser.parse_args()

if args.submission_ready and args.evaluation_in_progress:
    fail("submission-ready and evaluation-in-progress are mutually exclusive")

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
    "schema": "fastmri-r25-single-final-package-v1",
    "state": "SEALED",
    "candidate_count": 1,
    "final_package_count": 1,
    "fallback_registered": False,
    "official_evaluation_max_runs": 1,
    "generalist_handoff_epoch": 49,
    "generalist_handoff_optimizer_step": 228928,
    "scheduler_horizon_epoch": 51,
    "hard_deadline_unix": 1787237940,
    "hard_deadline_kst": "2026-08-20T23:59:00+09:00",
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
evaluation_generated: set[str] = set()
if args.evaluation_in_progress:
    start_relative = "evidence/official-evaluation-start.json"
    log_relative = "evidence/official-evaluation.log"
    receipt_relative = "evidence/official-evaluation-receipt.json"
    start_path = ROOT / start_relative
    receipt_path = ROOT / receipt_relative
    if not start_path.is_file() or receipt_path.exists():
        fail("evaluation-in-progress requires one start marker and no receipt")
    try:
        start = json.loads(start_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"invalid official evaluation start marker: {error}")
    if (
        not isinstance(start, dict)
        or start.get("schema") != "fastmri-r25-official-evaluation-start-v1"
        or start.get("state") != "STARTED"
        or start.get("attempt") != 1
        or start.get("command") != "bash run_official_evaluation_once.sh"
        or start.get("best_model_sha256") != sha256(model_path)
        or start.get("pre_evaluation_manifest_sha256") != sha256(manifest_path)
        or not isinstance(start.get("started_unix"), (int, float))
        or float(start["started_unix"]) >= 1787237940
    ):
        fail("official evaluation start marker is not exact attempt 1")
    evaluation_generated = {start_relative}
    if (ROOT / log_relative).is_file():
        evaluation_generated.add(log_relative)

if actual_files != sealed_files | evaluation_generated:
    fail(
        "manifest coverage mismatch: "
        f"unsealed={sorted(actual_files - sealed_files - evaluation_generated)}, "
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
verify_evidence_receipts(source_hashes, sha256(model_path))

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
        receipt.get("schema") != "fastmri-r25-official-evaluation-receipt-v1"
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
        or start.get("schema") != "fastmri-r25-official-evaluation-start-v1"
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
