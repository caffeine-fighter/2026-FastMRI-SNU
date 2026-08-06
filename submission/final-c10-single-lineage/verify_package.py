#!/usr/bin/env python3
"""Fail-closed verifier for the one R29 ZF-context single-lineage package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
INPUT_MODE = "recon_zero_filled_residual"
ZF_DEFINITION = "rss(fftshift(ifft2(ifftshift(masked_kspace),norm=ortho)))"
DEADLINE = 1_787_237_940
STRUCTURE = (
    "README.md",
    "requirements.txt",
    "recon_eval.sh",
    "record_official_evaluation.py",
    "reproduce_final.sh",
    "run_official_evaluation_once.sh",
    "assemble_final_package.py",
    "materialize_r29_evidence.py",
    "seal_package.py",
    "verify_package.py",
    "reproduction/FINAL_C10_SINGLE_LINEAGE_R29_ZF_CONTEXT.json",
    "reproduction/final-tactics-c10-r29-zf-context.json",
    "reproduction/FINAL_C10_SINGLE_LINEAGE_R29_INFERENCE.json",
    "reproduction/R29_CPU_PREFLIGHT.json",
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
    "reproduction/controller.py",
    "reproduction/generalist/train.py",
    "reproduction/generalist/promptmr_production.py",
    "reproduction/specialist/train.py",
    "reproduction/specialist/promptmr_production.py",
    "reproduction/vessl_train_post_refiner.py",
    "reproduction/vessl_build_routed_promptmr_checkpoint.py",
    "reproduction/promptmr_post_refiner.py",
    "reproduction/promptmr_router.py",
    "reproduction/promptmr_mask_router.py",
    "reproduction/promptmr_legal_mask.py",
    "reproduction/test_part.py",
    "reproduction/preflight_r29.py",
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
    "evidence/r29-amendment-deployment-receipt.json",
    "evidence/evidence-materialization-receipt.json",
)
LEARNED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def fail(message: str) -> None:
    raise SystemExit(f"FINAL_PACKAGE_INVALID: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(relative: str) -> dict:
    path = ROOT / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"invalid JSON {relative}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON object required: {relative}")
    return value


def verify_routed_model(path: Path) -> dict[str, str]:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        fail(f"best_model.pt cannot be loaded safely: {type(error).__name__}: {error}")
    if not isinstance(checkpoint, dict):
        fail("best_model.pt is not a dictionary checkpoint")

    project = ROOT / "project"
    sys.path.insert(0, str(project))
    try:
        from utils.learning.promptmr_router import RoutedCheckpointContract

        contract = RoutedCheckpointContract.validate(checkpoint)
    except Exception as error:
        fail(f"shipped promptmr_router rejected best_model.pt: {type(error).__name__}: {error}")
    finally:
        if sys.path and sys.path[0] == str(project):
            sys.path.pop(0)

    architecture = checkpoint.get("architecture")
    routing = checkpoint.get("routing_contract")
    components = checkpoint.get("components")
    post = checkpoint.get("post_refiner")
    if (
        checkpoint.get("schema") != "vessl-acceleration-routed-promptmr-v2"
        or checkpoint.get("all_components_trained_on_vessl") is not True
        or architecture
        != {
            "rung": "R2",
            "num_cascades": 10,
            "n_history": 0,
            "parameter_count": 54_090_459,
            "trainable_parameter_count": 45_381_339,
        }
        or checkpoint.get("routing_feature")
        != "exact_legal_mask_family_with_generalist_fail_safe"
        or not isinstance(routing, dict)
        or routing.get("schema") != "promptmr-legal-mask-routing-contract-v2"
        or routing.get("supported_accelerations") != [4, 8]
        or routing.get("generalist_component") != "generalist"
        or routing.get("unknown_or_mismatch") != "generalist"
        or routing.get("public_frequency_weighting") is not False
        or checkpoint.get("tta_views") != ["identity"]
        or checkpoint.get("tta_views_by_acceleration")
        != {"acc4": ["identity"], "acc8": ["identity", "flip_lr"]}
        or not isinstance(components, dict)
        or set(components) != {"generalist", "acc4", "acc8"}
        or not isinstance(post, dict)
        or contract.generalist_component != "generalist"
        or contract.tta_views_by_acceleration
        != {4: ("identity",), 8: ("identity", "flip_lr")}
    ):
        fail("best_model.pt outer routed contract is not exact R29")

    expected = {
        "generalist": (None, "all", 228_928, 49),
        "acc4": (4, "acc4", 4_672, 1),
        "acc8": (8, "acc8", 1_158, 0),
    }
    source_hashes: dict[str, str] = {}
    for name, (acceleration, route, step, epoch) in expected.items():
        component = components[name]
        source_hash = component.get("source_checkpoint_sha256") if isinstance(component, dict) else None
        if (
            not isinstance(component, dict)
            or component.get("acceleration") != acceleration
            or component.get("role") != name
            or component.get("rung") != "R2"
            or component.get("num_cascades") != 10
            or component.get("n_history") != 0
            or component.get("parameter_count") != 54_090_459
            or component.get("trainable_parameter_count") != 45_381_339
            or component.get("train_acceleration") != route
            or component.get("scratch") is not True
            or component.get("external_learned_state") is not False
            or component.get("trained_on_vessl") is not True
            or component.get("source_step") != step
            or component.get("source_epoch") != epoch
            or not isinstance(component.get("model"), dict)
            or not isinstance(source_hash, str)
            or len(source_hash) != 64
        ):
            fail(f"best_model.pt {name} component is invalid")
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
        or post.get("input_mode") != INPUT_MODE
        or post.get("zero_filled_definition") != ZF_DEFINITION
        or post.get("normalization") != "shared_detached_reconstruction_amax"
        or post.get("spatial_match") != "center_crop_then_zero_pad"
        or post.get("epoch") != 21
        or post.get("parent_epoch") != 49
        or post.get("late_branch_epochs") != [50, 70]
        or post.get("optimizer_steps") != 91_231
        or post.get("lr_horizon_optimizer_steps") != 93_567
        or post.get("steps_per_epoch") != 4_672
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
        or post.get("parameter_count") != 72_625
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
        fail("best_model.pt NAF_S component is not exact R29 ZF-context")
    source_hashes["post_refiner"] = post["source_checkpoint_sha256"]
    if {"optimizer", "scheduler", "rng_state", "ema", "swa", "scaler"}.intersection(checkpoint):
        fail("best_model.pt contains forbidden training state")
    return source_hashes


def verify_static_contracts() -> dict:
    amendment = json_object("reproduction/FINAL_C10_SINGLE_LINEAGE_R29_ZF_CONTEXT.json")
    tactics = json_object("reproduction/final-tactics-c10-r29-zf-context.json")
    runtime = json_object("reproduction/FINAL_C10_SINGLE_LINEAGE_R29_INFERENCE.json")
    preflight = json_object("reproduction/R29_CPU_PREFLIGHT.json")
    change = amendment.get("amendment")
    policy = amendment.get("policy")
    if (
        amendment.get("schema") != "final-c10-single-lineage-r29-zf-context-amendment-v1"
        or amendment.get("state") != "SEALED"
        or tactics.get("schema") != "vessl-final-tactics-scalar-handoff-v1"
        or tactics.get("state") != "SEALED"
        or runtime.get("schema") != "final-c10-single-lineage-r29-inference-amendment-v1"
        or runtime.get("state") != "SEALED"
        or preflight.get("schema") != "vessl-r29-zf-context-cpu-preflight-v1"
        or preflight.get("state") != "PASS"
        or preflight.get("cpu_only") is not True
        or preflight.get("cuda_initialized") is not False
        or preflight.get("actual_shipped_router_validation") is not True
        or preflight.get("contract_mutations_rejected")
        != ["optimizer_steps", "lr_horizon_optimizer_steps", "input_mode"]
        or preflight.get("unknown_mask_outer_tta") != ["identity"]
        or preflight.get("post_e49_optimizer_steps") != 97_061
        or not isinstance(change, dict)
        or change.get("input_mode") != INPUT_MODE
        or change.get("zero_filled_definition") != ZF_DEFINITION
        or change.get("post_e49_total_optimizer_steps") != 97_061
        or change.get("optimizer_step_budget_changed") is not False
        or not isinstance(policy, dict)
        or policy.get("candidate_count") != 1
        or policy.get("fallback_registered") is not False
        or policy.get("official_evaluation_max_runs") != 1
    ):
        fail("R29 contracts or CPU preflight are invalid")

    source_paths = {
        "controller.py": ROOT / "reproduction/controller.py",
        "train.py": ROOT / "reproduction/specialist/train.py",
        "promptmr_production.py": ROOT / "reproduction/specialist/promptmr_production.py",
        "vessl_train_post_refiner.py": ROOT / "reproduction/vessl_train_post_refiner.py",
        "vessl_build_routed_promptmr_checkpoint.py": ROOT / "reproduction/vessl_build_routed_promptmr_checkpoint.py",
        "promptmr_post_refiner.py": ROOT / "reproduction/promptmr_post_refiner.py",
        "promptmr_router.py": ROOT / "reproduction/promptmr_router.py",
        "promptmr_mask_router.py": ROOT / "reproduction/promptmr_mask_router.py",
        "promptmr_legal_mask.py": ROOT / "reproduction/promptmr_legal_mask.py",
        "test_part.py": ROOT / "reproduction/test_part.py",
        "preflight_r29.py": ROOT / "reproduction/preflight_r29.py",
    }
    hashes = amendment.get("source_hashes")
    if not isinstance(hashes, dict):
        fail("R29 source hash map is absent")
    for name, path in source_paths.items():
        if not path.is_file() or sha256(path) != hashes.get(name):
            fail(f"sealed R29 source mismatch: {name}")
    for name, relative in {
        "promptmr_post_refiner.py": "project/utils/learning/promptmr_post_refiner.py",
        "promptmr_router.py": "project/utils/learning/promptmr_router.py",
        "promptmr_mask_router.py": "project/utils/learning/promptmr_mask_router.py",
        "promptmr_legal_mask.py": "project/utils/learning/promptmr_legal_mask.py",
        "test_part.py": "project/utils/learning/test_part.py",
    }.items():
        if sha256(ROOT / relative) != hashes[name]:
            fail(f"shipped inference source is not sealed R29: {relative}")
    return amendment


def verify_evidence(source_hashes: dict[str, str], model_sha: str) -> None:
    deployment = json_object("evidence/r29-amendment-deployment-receipt.json")
    assembly = json_object("evidence/assembly-receipt.json")
    generalist = json_object("evidence/generalist-e49-receipt.json")
    acc4 = json_object("evidence/acc4-specialist-receipt.json")
    acc8 = json_object("evidence/acc8-specialist-receipt.json")
    naf = json_object("evidence/naf-s-training-receipt.json")
    policy = json_object("evidence/policy-receipt.json")
    admission = json_object("evidence/inference-admission-receipt.json")
    materialized = json_object("evidence/evidence-materialization-receipt.json")
    controller = json_object("evidence/raw/controller-final-receipt.json")
    if (
        deployment.get("schema") != "vessl-c10-single-lineage-r29-zf-context-deployment-v1"
        or deployment.get("state") != "ACTIVE_CURRENT_C10_PRESERVED_R29_ZF_CONTEXT_ARMED"
        or deployment.get("trainer_signal_sent") is not False
        or deployment.get("active_generalist_process_touched") is not False
        or deployment.get("active_generalist_recipe_changed") is not False
        or deployment.get("post_e49_optimizer_step_budget_changed") is not False
        or deployment.get("post_e49_optimizer_steps") != 97_061
        or deployment.get("candidate_count") != 1
        or deployment.get("fallback_registered") is not False
        or assembly.get("schema") != "vessl-r29-single-final-package-assembly-v1"
        or assembly.get("state") != "PASS"
        or assembly.get("candidate_count") != 1
        or assembly.get("fallback_registered") is not False
        or assembly.get("final_checkpoint_sha256") != model_sha
        or materialized.get("schema") != "fastmri-r29-evidence-materialization-v1"
        or materialized.get("state") != "PASS"
        or materialized.get("best_model_sha256") != model_sha
        or materialized.get("component_source_sha256") != source_hashes
    ):
        fail("R29 deployment, assembly, or materialization evidence is invalid")
    if (
        generalist.get("epoch") != 49
        or generalist.get("optimizer_step") != 228_928
        or generalist.get("checkpoint_sha256") != source_hashes["generalist"]
        or acc4.get("schema") != "fastmri-r29-specialist-receipt-v1"
        or acc4.get("optimizer_steps") != 4_672
        or acc4.get("lr_horizon_optimizer_steps") != 35_040
        or acc4.get("checkpoint_sha256") != source_hashes["acc4"]
        or acc8.get("schema") != "fastmri-r29-specialist-receipt-v1"
        or acc8.get("optimizer_steps") != 1_158
        or acc8.get("lr_horizon_optimizer_steps") != 2_315
        or acc8.get("checkpoint_sha256") != source_hashes["acc8"]
        or naf.get("checkpoint_sha256") != source_hashes["post_refiner"]
        or naf.get("optimizer_steps") != 91_231
        or naf.get("lr_horizon_optimizer_steps") != 93_567
        or naf.get("input_mode") != INPUT_MODE
        or naf.get("zero_filled_definition") != ZF_DEFINITION
        or naf.get("main_parameters_updated") is not False
        or policy.get("schema") != "fastmri-r29-policy-receipt-v1"
        or policy.get("candidate_count") != 1
        or policy.get("fallback_registered") is not False
        or policy.get("unknown_or_mismatch_route") != "generalist_identity"
        or controller.get("final_checkpoint_sha256") != model_sha
        or controller.get("fallback_checkpoint") is not None
        or admission.get("state") != "PASS"
        or admission.get("final_checkpoint_sha256") != model_sha
    ):
        fail("R29 lineage/component/policy evidence is invalid")


parser = argparse.ArgumentParser()
parser.add_argument("--structure-only", action="store_true")
parser.add_argument("--submission-ready", action="store_true")
parser.add_argument("--evaluation-in-progress", action="store_true")
args = parser.parse_args()
if args.submission_ready and args.evaluation_in_progress:
    fail("submission-ready and evaluation-in-progress are mutually exclusive")
for relative in STRUCTURE:
    if not (ROOT / relative).is_file():
        fail(f"missing {relative}")
if args.structure_only:
    print("FINAL_R29_PACKAGE_STRUCTURE_OK")
    raise SystemExit(0)

manifest_path = ROOT / "package-manifest.json"
model_path = ROOT / "best_model.pt"
if not manifest_path.is_file() or not model_path.is_file():
    fail("sealed manifest or best_model.pt is missing")
for relative in FINAL_STRUCTURE:
    if not (ROOT / relative).is_file():
        fail(f"missing final artifact: {relative}")
manifest = json_object("package-manifest.json")
required_manifest = {
    "schema": "fastmri-r29-single-final-package-v1",
    "state": "SEALED",
    "candidate_count": 1,
    "final_package_count": 1,
    "fallback_registered": False,
    "official_evaluation_max_runs": 1,
    "generalist_handoff_epoch": 49,
    "generalist_handoff_optimizer_step": 228_928,
    "scheduler_horizon_epoch": 51,
    "hard_deadline_unix": DEADLINE,
    "hard_deadline_kst": "2026-08-20T23:59:00+09:00",
    "external_learned_state_imported": False,
    "leaderboard_data_used_for_training_or_selection": False,
}
for key, expected in required_manifest.items():
    if manifest.get(key) != expected:
        fail(f"manifest {key}={manifest.get(key)!r}, expected {expected!r}")

files = manifest.get("files")
if not isinstance(files, dict) or not files:
    fail("manifest file map is absent")
actual_files = {
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file() and path != manifest_path and path.relative_to(ROOT).parts[0] != "result"
}
if any(path.is_symlink() for path in ROOT.rglob("*")):
    fail("symlinks are forbidden")
generated: set[str] = set()
if args.evaluation_in_progress:
    start_relative = "evidence/official-evaluation-start.json"
    receipt_relative = "evidence/official-evaluation-receipt.json"
    start = json_object(start_relative)
    if (ROOT / receipt_relative).exists():
        fail("evaluation-in-progress cannot already have a receipt")
    if (
        start.get("schema") != "fastmri-r29-official-evaluation-start-v1"
        or start.get("state") != "STARTED"
        or start.get("attempt") != 1
        or start.get("best_model_sha256") != sha256(model_path)
        or float(start.get("started_unix", DEADLINE)) >= DEADLINE
    ):
        fail("official evaluation start marker is invalid")
    generated.add(start_relative)
    if (ROOT / "evidence/official-evaluation.log").is_file():
        generated.add("evidence/official-evaluation.log")
if actual_files != set(files) | generated:
    fail("manifest coverage mismatch")
for relative, record in files.items():
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        fail(f"manifest path escapes package: {relative}")
    if not path.is_file() or path.is_symlink() or not isinstance(record, dict):
        fail(f"invalid sealed file: {relative}")
    if path.stat().st_size != int(record.get("bytes", -1)) or sha256(path) != record.get("sha256"):
        fail(f"sealed file mismatch: {relative}")
for relative in FINAL_STRUCTURE:
    if relative not in files:
        fail(f"required artifact is not sealed: {relative}")

learned = sorted(
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() in LEARNED_SUFFIXES and path.relative_to(ROOT).parts[0] != "result"
)
if learned != ["best_model.pt"]:
    fail(f"expected exactly one learned-state file, observed {learned}")

source_hashes = verify_routed_model(model_path)
verify_static_contracts()
verify_evidence(source_hashes, sha256(model_path))

if args.submission_ready:
    receipt = json_object("evidence/official-evaluation-receipt.json")
    start = json_object("evidence/official-evaluation-start.json")
    scores = receipt.get("scores")
    log_path = ROOT / str(receipt.get("log", ""))
    if (
        receipt.get("schema") != "fastmri-r29-official-evaluation-receipt-v1"
        or receipt.get("state") != "PASS"
        or receipt.get("attempt") != 1
        or receipt.get("official_evaluation_attempt_count") != 1
        or receipt.get("return_code") != 0
        or receipt.get("best_model_sha256") != sha256(model_path)
        or receipt.get("leaderboard_data_used_for_training_or_selection") is not False
        or float(receipt.get("completed_unix", DEADLINE + 1)) > DEADLINE
        or start.get("schema") != "fastmri-r29-official-evaluation-start-v1"
        or start.get("started_unix") != receipt.get("started_unix")
        or not isinstance(scores, dict)
        or set(scores)
        != {
            "ssim_full", "ssim_bbox", "recon_time_seconds", "ssim_full_acc4",
            "ssim_full_acc8", "ssim_bbox_acc4", "ssim_bbox_acc8",
        }
        or not log_path.is_file()
        or receipt.get("log_sha256") != sha256(log_path)
    ):
        fail("official evaluation receipt is invalid")

print(
    json.dumps(
        {
            "state": "FINAL_R29_PACKAGE_OK",
            "candidate_count": 1,
            "best_model_sha256": sha256(model_path),
            "component_source_sha256": source_hashes,
            "file_count": len(files),
        },
        sort_keys=True,
    )
)
