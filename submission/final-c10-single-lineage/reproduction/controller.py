#!/usr/bin/env python3
"""Promote a non-R1 G10 architecture into a fresh VESSL final lineage.

The controller is deliberately dormant while the already-running R1
generalist builds the sealed E35 fallback.  If G10 retains R1, the existing
post-E35 dispatcher remains authoritative.  If G10 selects R2 or R1/H11, this
controller waits for the R1 E35 fallback, runs an idle-GPU GTX1080 admission,
then trains the selected architecture from fresh initialization on VESSL.
No learned state, score, metric, or organizer leaderboard payload enters via
the scalar handoffs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import signal
import subprocess
import time
from typing import Any


ROOT = Path("/root/2026-FastMRI-SNU-promptmr-training")
RESULT = Path("/root/result")
CONTROL = Path(
    os.environ.get(
        "FASTMRI_G10_CONTROL",
        str(RESULT / "VESSL_G10_ARCHITECTURE_DISPATCHER_R1"),
    )
)
STATUS = CONTROL / "status.json"
RECEIPT = CONTROL / "receipt.json"
LOCK = CONTROL / "owner.lock"
FINAL_GPU_LOCK = RESULT / "VESSL_FINAL_GPU_OWNER_R1/owner.lock"
AUTHORITY_CLAIM = CONTROL / "authority-claim.json"
LOGS = CONTROL / "logs"
STAGED = Path(
    os.environ.get(
        "FASTMRI_G10_STAGED",
        "/root/codex_ops/vessl_g10_architecture_dispatcher_r1",
    )
)

ARCH_HANDOFF = Path(
    os.environ.get(
        "FASTMRI_ARCH_HANDOFF",
        str(
            RESULT
            / "VESSL_SCORE_FREE_HANDOFFS/architecture-g10-r1.json"
        ),
    )
)
TACTICS_HANDOFF = Path(
    os.environ.get(
        "FASTMRI_TACTICS_HANDOFF",
        str(
            RESULT
            / "VESSL_SCORE_FREE_HANDOFFS/final-tactics-g11-r1.json"
        ),
    )
)
TERMINAL_TACTICS_HANDOFF = Path(
    os.environ.get(
        "FASTMRI_TERMINAL_TACTICS_HANDOFF",
        str(
            RESULT
            / "VESSL_SCORE_FREE_HANDOFFS/final-tactics-g11-terminal-naf-r2.json"
        ),
    )
)
TERMINAL_TACTICS_WAIT_SECONDS = float(
    # The RunPod research parent and the VESSL-scratch generalist can reach
    # E40 within the same day.  ACC8 and ACC4 late-MoE research then runs
    # serially on the single A6000, so a two-hour wait could silently select
    # the pre-research fallback just before the score-free receipts arrive.
    # Four days is still bounded and leaves the August 20 finalization window.
    os.environ.get("FASTMRI_TERMINAL_TACTICS_WAIT_SECONDS", "345600")
)
_ARCHITECTURE_ADMISSION_SOURCE = os.environ.get(
    "FASTMRI_G10_ARCHITECTURE_ADMISSION_SOURCE"
)
ARCHITECTURE_ADMISSION_SOURCE = (
    Path(_ARCHITECTURE_ADMISSION_SOURCE)
    if _ARCHITECTURE_ADMISSION_SOURCE
    else None
)
REQUIRE_TERMINAL_TACTICS_HANDOFF = (
    os.environ.get("FASTMRI_REQUIRE_TERMINAL_TACTICS_HANDOFF") == "1"
)
SINGLE_FINAL_REQUIRED = (
    os.environ.get("FASTMRI_SINGLE_FINAL_REQUIRED") == "1"
)
R1_E35_CONTRACT = RESULT / "VESSL_GENERALIST_E35_MODEL_ONLY_R1/contract.json"
R1_HARDSTART_CONTRACT = (
    RESULT / "VESSL_GENERALIST_HARDSTART_FALLBACK_R1/contract.json"
)
LEGACY_LOOP_FRAGMENT = "vessl_post_e35_final_loop_r2.sh"
LEGACY_DISPATCHER_FRAGMENT = "vessl_post_e35_final_dispatcher_r2.py"

TRAIN_SOURCE = ROOT / "train.py"
PRODUCTION_SOURCE = ROOT / "utils/learning/promptmr_production.py"
LEGAL_MASK_SOURCE = ROOT / "utils/learning/promptmr_legal_mask.py"
MASK_ROUTER_SOURCE = ROOT / "utils/learning/promptmr_mask_router.py"
ROUTER_DESTINATION = ROOT / "utils/learning/promptmr_router.py"
POST_REFINER_DESTINATION = (
    ROOT / "utils/learning/promptmr_post_refiner.py"
)
MASK_CONDITIONING_DESTINATION = (
    ROOT / "utils/learning/promptmr_mask_conditioning.py"
)
MASK_DC_DESTINATION = (
    ROOT / "utils/learning/promptmr_mask_conditioned_dc.py"
)
MASK_PROMPT_DESTINATION = (
    ROOT / "utils/learning/promptmr_mask_conditioned_prompt.py"
)
TEST_PART_DESTINATION = ROOT / "test_part.py"

STAGED_TRAIN = STAGED / "train.py"
STAGED_PRODUCTION = STAGED / "promptmr_production.py"
STAGED_LEGAL_MASK = STAGED / "promptmr_legal_mask.py"
STAGED_MASK_ROUTER = STAGED / "promptmr_mask_router.py"
STAGED_ADMISSION = STAGED / "vessl_architecture_training_admission.py"
STAGED_BUILDER = STAGED / "vessl_build_routed_promptmr_checkpoint.py"
STAGED_ROUTER = STAGED / "promptmr_router.py"
STAGED_TEST_PART = STAGED / "test_part.py"
STAGED_FINAL_ADMISSION = STAGED / "vessl_final_lazy_router_admission.py"
STAGED_POST_REFINER = STAGED / "promptmr_post_refiner.py"
STAGED_POST_REFINER_TRAIN = STAGED / "vessl_train_post_refiner.py"
STAGED_RUNTIME_AMENDMENT = (
    STAGED / "FINAL_C10_SINGLE_LINEAGE_R29_INFERENCE.json"
)
SPECIALIST_STAGED = Path(
    os.environ.get(
        "FASTMRI_R19_SPECIALIST_STAGED",
        "/root/codex_ops/terminal_legal_specialist_r12_scheduler_fix",
    )
)
STAGED_SPECIALIST_TRAIN = SPECIALIST_STAGED / "train.py"
STAGED_SPECIALIST_PRODUCTION = (
    SPECIALIST_STAGED / "promptmr_production.py"
)
STAGED_SPECIALIST_COMPOSER = (
    SPECIALIST_STAGED / "compose_specialist_checkpoint.py"
)
STAGED_SPECIALIST_SOUP_COMPOSER = (
    SPECIALIST_STAGED / "compose_specialist_soup.py"
)
STAGED_MASK_CONDITIONING = STAGED / "promptmr_mask_conditioning.py"
STAGED_MASK_DC = STAGED / "promptmr_mask_conditioned_dc.py"
STAGED_MASK_PROMPT = STAGED / "promptmr_mask_conditioned_prompt.py"
STAGED_MASK_SPECIALIST_TRAIN = (
    STAGED / "vessl_train_mask_conditioned_specialist.py"
)

EXPECTED_STAGED_SHA256 = {
    STAGED_TRAIN: os.environ.get(
        "FASTMRI_G10_STAGED_TRAIN_SHA256",
        "d234cfffb3f15c3953da41087411c61c7570cd33ec0ac0988924881ec7f365d1",
    ),
    STAGED_PRODUCTION: os.environ.get(
        "FASTMRI_G10_STAGED_PRODUCTION_SHA256",
        "dc5da80e513191bb3695641f541c3041c82d523af613ada107c4eb79317e000c",
    ),
    STAGED_LEGAL_MASK: "906916b8834580cd0ce7c456890d2ffb4e01766065238b2f9f0a72b5b3e9a239",
    STAGED_MASK_ROUTER: "d0647a16cd83572386fc9b3aef5dc69c66a9812a1a55e071ef05411e55234601",
    STAGED_ADMISSION: os.environ.get(
        "FASTMRI_G10_STAGED_ADMISSION_SHA256",
        "51df92e9bc0a16a6c188579cf780ea407fc8a10c938a86421b3afb77df2b31d8",
    ),
    STAGED_BUILDER: os.environ.get(
        "FASTMRI_G10_STAGED_BUILDER_SHA256",
        "4da709fa46f99456b88a2011275f3ddf4e135e02917c6bc85032969ce245c916",
    ),
    STAGED_ROUTER: os.environ.get(
        "FASTMRI_G10_STAGED_ROUTER_SHA256",
        "b87d38e84da34820258a067f9ebd61d12403d483a124fd367be9f45908fa1825",
    ),
    STAGED_TEST_PART: os.environ.get(
        "FASTMRI_G10_STAGED_TEST_PART_SHA256",
        "8111150c1e4b2056fccb83a5f301367fbaf6f3d2eecf43730c894966d982b3f6",
    ),
    STAGED_FINAL_ADMISSION: os.environ.get(
        "FASTMRI_G10_STAGED_FINAL_ADMISSION_SHA256",
        "60ff7cf7cc6a04168e842d6053c0d23e2f1dd39f1a790718cf5eea9be7949b72",
    ),
    STAGED_POST_REFINER: os.environ.get(
        "FASTMRI_G10_STAGED_POST_REFINER_SHA256",
        "2b06495d2f192134155fa771b1f482942ee5afb590f2a945bf42d982af990537",
    ),
    STAGED_POST_REFINER_TRAIN: os.environ.get(
        "FASTMRI_G10_STAGED_POST_REFINER_TRAIN_SHA256",
        "cfaf578c5ccec3d3db7fe02a63735d72b13613528ca93993987779b655dc88af",
    ),
    STAGED_RUNTIME_AMENDMENT: os.environ.get(
        "FASTMRI_R29_INFERENCE_AMENDMENT_SHA256",
        "R29_RUNTIME_SHA256_MUST_BE_EXPLICITLY_BOUND",
    ),
    STAGED_SPECIALIST_TRAIN: os.environ.get(
        "FASTMRI_R19_SPECIALIST_TRAIN_SHA256",
        "95465b1b09189af87359a39518559f1759fdbeb881a4abb35b1f7b7faa832e47",
    ),
    STAGED_SPECIALIST_PRODUCTION: os.environ.get(
        "FASTMRI_R19_SPECIALIST_PRODUCTION_SHA256",
        "ea4695f5fada7c417323d9efad495544d0743ad1d35b3023c1a645a421d8688b",
    ),
    STAGED_SPECIALIST_COMPOSER: os.environ.get(
        "FASTMRI_R19_SPECIALIST_COMPOSER_SHA256",
        "8e855eb6dba1e297c7550c28b68853133d2266dd20252f610e69dc1931ea3feb",
    ),
    STAGED_SPECIALIST_SOUP_COMPOSER: os.environ.get(
        "FASTMRI_R19_SPECIALIST_SOUP_COMPOSER_SHA256",
        "8afc17dbb945462c1c52edc24855bc69d82dd348276e5bc1a52732647d2665ef",
    ),
    STAGED_MASK_CONDITIONING: os.environ.get(
        "FASTMRI_G10_STAGED_MASK_CONDITIONING_SHA256",
        "2d83f9ba844dd6361fef00d28d08605426751224620c036cb7361224fb87dbbb",
    ),
    STAGED_MASK_DC: os.environ.get(
        "FASTMRI_G10_STAGED_MASK_DC_SHA256",
        "f7e5482b995049ca5b43719dd015019898983189afca4914f65c1a040dcfab40",
    ),
    STAGED_MASK_PROMPT: os.environ.get(
        "FASTMRI_G10_STAGED_MASK_PROMPT_SHA256",
        "ff254a569887e340240a0132d87eb117087d3f44c3f21b137ab591ff6dcf6ea4",
    ),
    STAGED_MASK_SPECIALIST_TRAIN: os.environ.get(
        "FASTMRI_G10_STAGED_MASK_SPECIALIST_TRAIN_SHA256",
        "5ad7320df04a6ddf0259b2809d6e07d7a4de780ffb758e20f23847dc53485a90",
    ),
}
ORIGINAL_PRODUCTION_SHA256 = (
    "9cc7544e3cb54bcfeb06140130781e638afe1c1429b50478001175a558f02aae"
)
INFERENCE_SNAPSHOT_FILES = (
    Path("test_part.py"),
    Path("utils/learning/promptmr_router.py"),
    Path("utils/learning/promptmr_legal_mask.py"),
    Path("utils/learning/promptmr_mask_router.py"),
    Path("train.py"),
    Path("utils/learning/promptmr_production.py"),
    Path("utils/learning/promptmr_post_refiner.py"),
    Path("utils/learning/promptmr_mask_conditioning.py"),
    Path("utils/learning/promptmr_mask_conditioned_dc.py"),
    Path("utils/learning/promptmr_mask_conditioned_prompt.py"),
)

REGISTERED = {
    "R1": {
        "rung": "R1",
        "n_history": 0,
        "run_name": (
            "EXP_PROMPTMR_R1_G16_LEGALMASK_DELAY5_COS50_E35_SEED430_V1"
        ),
    },
    "R2": {
        "rung": "R2",
        "n_history": 0,
        "run_name": "EXP_PROMPTMR_R2_G10_FINAL_DELAY5_COS50_E40_SEED430_V1",
    },
    "R1_H11": {
        "rung": "R1",
        "n_history": 11,
        "run_name": (
            "EXP_PROMPTMR_R1_H11_G10_FINAL_DELAY5_COS50_E40_SEED430_V1"
        ),
    },
    "R2_C10": {
        "rung": "R2",
        "n_history": 0,
        "run_name": (
            "EXP_PROMPTMR_R2_C10_G20_FINAL_DELAY5_COS50_E40_SEED430_V1"
        ),
    },
    "R2_C12": {
        "rung": "R2",
        "n_history": 0,
        "run_name": (
            "EXP_PROMPTMR_R2_C12_G23_FINAL_DELAY5_COS50_E40_SEED430_V1"
        ),
    },
}
MAX_RECOVERIES = 12
POLL_SECONDS = 60
TRAINING_POLL_SECONDS = 15
# 2026-08-20 22:00 KST.  The single-final deadline path deliberately consumes
# the available training window; non-amended paths retain their own fallback
# behavior when this cutoff is reached.
FINAL_RESEARCH_CUTOFF_UNIX = float(
    os.environ.get("FASTMRI_FINAL_RESEARCH_CUTOFF_UNIX", "1787230800.0")
)

# R23 keeps the already-running cosine-51 optimization trajectory intact but
# seals the single downstream lineage at the exact E49 epoch boundary.  The
# horizon remains 51 so no scheduler/optimizer state is rewritten in flight.
GENERALIST_SCHEDULER_EPOCHS = 51
GENERALIST_SCHEDULER_STEPS = 238_272
GENERALIST_HANDOFF_EPOCH = int(
    os.environ.get("FASTMRI_GENERALIST_HANDOFF_EPOCH", "49")
)
GENERALIST_HANDOFF_STEP = 4_672 * GENERALIST_HANDOFF_EPOCH
if GENERALIST_HANDOFF_EPOCH != 49:
    raise RuntimeError("R23 requires the exact E49 generalist handoff")

R19_ACC4_SCHEMA = "vessl-score-free-acc4-late-moe-e49-r25-e2-prefix-v1"
R19_ACC8_SCHEMA = "vessl-score-free-acc8-late-moe-e49-r23-r10-prefix-v1"
R19_BBOX_LOSS_FAMILY = (
    "winner_foreground_ssim_l1_sqrt_area_plus_official384_bbox05_v2"
)
R19_PLAIN_LOSS_FAMILY = "winner_foreground_ssim_l1_sqrt_area_v1"


class FinalResearchCutoff(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def valid_embedded_contract(value: object, schema: str) -> bool:
    if not isinstance(value, dict) or value.get("schema") != schema:
        return False
    observed = value.get("contract_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    payload = dict(value)
    payload.pop("contract_sha256", None)
    return object_sha256(payload) == observed


def valid_post_training_data_contract(value: object) -> bool:
    if not valid_embedded_contract(
        value,
        "vessl-organizer-training-data-contract-v1",
    ):
        return False
    members = value.get("members")
    if not isinstance(members, list) or len(members) != 2:
        return False
    if [member.get("root") for member in members] != [
        "/root/Data/train",
        "/root/Data/val",
    ]:
        return False
    return value.get("leaderboard_data_read") is False and all(
        isinstance(member, dict)
        and isinstance(member.get("trusted_manifest_sha256"), str)
        and len(member["trusted_manifest_sha256"]) == 64
        and isinstance(member.get("file_shape_identity_inventory_sha256"), str)
        and len(member["file_shape_identity_inventory_sha256"]) == 64
        for member in members
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def snapshot_inference_sources(final_directory: Path) -> dict[str, Any]:
    """Atomically bind the exact mutable inference bytes to one candidate."""
    snapshot = final_directory / "inference-source-snapshot"
    manifest = snapshot / "MANIFEST.json"
    existing = load_json(manifest)
    if existing:
        files = existing.get("files")
        if (
            existing.get("schema")
            != "vessl-final-inference-source-snapshot-v1"
            or existing.get("state") != "SEALED"
            or not isinstance(files, list)
            or {Path(str(item.get("path"))) for item in files}
            != set(INFERENCE_SNAPSHOT_FILES)
        ):
            raise RuntimeError("existing inference source snapshot is invalid")
        for item in files:
            path = snapshot / str(item["path"])
            if (
                not path.is_file()
                or sha256(path) != str(item.get("sha256"))
            ):
                raise RuntimeError("inference source snapshot hash mismatch")
        return {
            "directory": str(snapshot),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "files": files,
        }

    temporary = final_directory / (
        f".inference-source-snapshot.{os.getpid()}.tmp"
    )
    if snapshot.exists() or temporary.exists():
        raise RuntimeError("partial inference source snapshot exists")
    temporary.mkdir()
    files: list[dict[str, Any]] = []
    try:
        for relative in INFERENCE_SNAPSHOT_FILES:
            source = ROOT / relative
            if not source.is_file():
                raise RuntimeError(
                    f"required inference source is missing: {source}"
                )
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            destination.chmod(0o444)
            files.append(
                {
                    "path": str(relative),
                    "sha256": sha256(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )
        atomic_json(
            temporary / "MANIFEST.json",
            {
                "schema": "vessl-final-inference-source-snapshot-v1",
                "state": "SEALED",
                "files": files,
                "leaderboard_data_read": False,
                "external_learned_state_imported": False,
                "created_unix": time.time(),
            },
        )
        (temporary / "MANIFEST.json").chmod(0o444)
        for directory in sorted(
            (
                path
                for path in temporary.rglob("*")
                if path.is_dir()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        os.replace(temporary, snapshot)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "directory": str(snapshot),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "files": files,
    }


def update(state: str, **extra: object) -> None:
    atomic_json(
        STATUS,
        {
            "schema": "vessl-g10-architecture-dispatcher-status-v1",
            "state": state,
            "pid": os.getpid(),
            "leaderboard_data_read": False,
            "external_learned_state_imported": False,
            "updated_unix": time.time(),
            **extra,
        },
    )


def validate_staged() -> None:
    optional_legacy_specialist_sources = {
        STAGED_SPECIALIST_TRAIN,
        STAGED_SPECIALIST_PRODUCTION,
        STAGED_SPECIALIST_COMPOSER,
        STAGED_SPECIALIST_SOUP_COMPOSER,
        STAGED_MASK_SPECIALIST_TRAIN,
    }
    for path, expected in EXPECTED_STAGED_SHA256.items():
        if SINGLE_FINAL_REQUIRED and path in optional_legacy_specialist_sources:
            continue
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"staged source mismatch: {path}")
    amendment = load_json(STAGED_RUNTIME_AMENDMENT)
    runtime = amendment.get("runtime", {})
    frame = runtime.get("training_frame_alignment", {})
    coil_batch = runtime.get("sensitivity_coil_microbatch", {})
    candidate_policy = amendment.get("candidate_policy", {})
    staged_sources = amendment.get("staged_sources", {})
    if (
        amendment.get("schema")
        != "final-c10-single-lineage-r29-inference-amendment-v1"
        or amendment.get("state") != "SEALED"
        or amendment.get("scope")
        != "INFERENCE_ONLY_FUTURE_SINGLE_FINAL_PACKAGE"
        or runtime.get("all_work_inside_official_timed_recon_slice")
        is not True
        or runtime.get("post_refiner_input_mode")
        != "recon_zero_filled_residual"
        or runtime.get("unknown_mask_outer_tta") != ["identity"]
        or runtime.get("exact_mask_route_selected_inside_recon_slice")
        is not True
        or frame.get("enabled") is not True
        or frame.get("height") != 384
        or frame.get("operation")
        != "ifft2c_center_crop_height_384_fft2c_then_reapply_official_mask"
        or coil_batch.get("enabled") is not True
        or coil_batch.get("group_size") != 8
        or candidate_policy.get("candidate_count") != 1
        or candidate_policy.get("fallback_registered") is not False
        or candidate_policy.get("multiple_candidates_allowed") is not False
        or staged_sources.get("test_part.py")
        != EXPECTED_STAGED_SHA256[STAGED_TEST_PART]
        or staged_sources.get("vessl_final_lazy_router_admission.py")
        != EXPECTED_STAGED_SHA256[STAGED_FINAL_ADMISSION]
        or staged_sources.get("promptmr_post_refiner.py")
        != EXPECTED_STAGED_SHA256[STAGED_POST_REFINER]
        or staged_sources.get("promptmr_router.py")
        != EXPECTED_STAGED_SHA256[STAGED_ROUTER]
        or staged_sources.get("promptmr_mask_router.py")
        != EXPECTED_STAGED_SHA256[STAGED_MASK_ROUTER]
    ):
        raise RuntimeError("invalid R29 inference amendment")


def validate_handoffs(
    *,
    tactics_path: Path | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    selected_tactics_path = (
        TACTICS_HANDOFF if tactics_path is None else Path(tactics_path)
    )
    while True:
        architecture = load_json(ARCH_HANDOFF)
        tactics = load_json(selected_tactics_path)
        if not architecture or not tactics:
            update(
                "WAITING_SCORE_FREE_HANDOFFS",
                architecture_handoff_present=bool(architecture),
                tactics_handoff_present=bool(tactics),
                tactics_handoff_path=str(selected_tactics_path),
            )
            time.sleep(POLL_SECONDS)
            continue
        arch_transfer = architecture.get("transfer_contract")
        tactics_transfer = tactics.get("transfer_contract")
        arch_recipe = architecture.get("selected_recipe")
        tactics_recipe = tactics.get("selected_recipe")
        winner = str(architecture.get("winner", ""))
        if (
            architecture.get("schema")
            != "vessl-architecture-scalar-handoff-v1"
            or architecture.get("state") != "SEALED"
            or architecture.get("component_role") != "generalist_architecture"
            or not isinstance(arch_transfer, dict)
            or arch_transfer.get("learned_state_included") is not False
            or arch_transfer.get("research_measurements_included") is not False
            or arch_transfer.get("final_component_training")
            != "fresh_end_to_end_on_vessl"
            or not isinstance(arch_recipe, dict)
        ):
            raise RuntimeError("invalid G10 architecture scalar handoff")
        if (
            tactics.get("schema")
            != "vessl-final-tactics-scalar-handoff-v1"
            or tactics.get("state") != "SEALED"
            or tactics.get("component_role") != "final_tactics"
            or not isinstance(tactics_transfer, dict)
            or tactics_transfer.get("learned_state_included") is not False
            or tactics_transfer.get("research_measurements_included") is not False
            or tactics_transfer.get("final_learned_state_source") != "VESSL_ONLY"
            or not isinstance(tactics_recipe, dict)
            or tactics_recipe.get("generalist") != arch_recipe
        ):
            raise RuntimeError("invalid or mismatched G11 tactics scalar handoff")
        late = tactics_recipe.get("late_finetune")
        if not isinstance(late, dict) or late.get("enabled") is not False:
            raise RuntimeError(
                "architecture dispatcher only accepts the pruned late-loss arm"
            )
        mask_family = tactics_recipe.get("training_mask_family")
        generator = (
            mask_family.get("generator")
            if isinstance(mask_family, dict)
            else None
        )
        training = (
            mask_family.get("training")
            if isinstance(mask_family, dict)
            else None
        )
        if (
            not isinstance(mask_family, dict)
            or mask_family.get("schema")
            != "vessl-score-free-legal-mask-recipe-v1"
            or mask_family.get("enabled") is not True
            or not isinstance(generator, dict)
            or generator.get("accelerations") != [4, 8]
            or generator.get("native_width") is not True
            or generator.get("center_fraction") != 0.08
            or generator.get("residue_policy")
            != "per_sample_full_cycle_across_epochs"
            or generator.get("public_frequency_weighting") is not False
            or not isinstance(training, dict)
            or training.get("source")
            != "organizer_train_full_kspace_only"
            or training.get("public_target_or_image_used") is not False
            or training.get("public_payload_required_on_vessl") is not False
        ):
            raise RuntimeError("invalid score-free legal-mask tactic")
        post_refiner = tactics_recipe.get("post_refiner")
        if not isinstance(post_refiner, dict):
            raise RuntimeError("post-refiner tactic is absent")
        if post_refiner.get("enabled") is True:
            registered_refiners = {
                "NAF_S": 72_625,
                "PLAIN_168K": 168_049,
                "NAF_M": 248_641,
                "NAF_L": 815_713,
            }
            variant = str(post_refiner.get("variant", ""))
            mask_conditioned = bool(
                post_refiner.get("mask_conditioned", False)
            )
            expected_parameter_count = registered_refiners.get(variant, -1) + (
                1_440 if mask_conditioned and variant == "NAF_S" else 0
            )
            expected_trainable_scope = (
                "naf_s_plus_mask_conditioner"
                if mask_conditioned
                else "naf_s_only"
            )
            if (
                variant not in registered_refiners
                or int(post_refiner.get("parameter_count", -1))
                != expected_parameter_count
                or int(post_refiner.get("epochs", -1)) not in {1, 3, 10, 15, 20, 21}
                or post_refiner.get("views")
                not in (
                    ["identity"],
                    ["identity", "flip_lr"],
                    ["identity", "flip_lr", "flip_ud", "rot180"],
                )
                or float(post_refiner.get("peak_lr", -1)) != 0.0001
                or float(post_refiner.get("weight_decay", -1))
                != 0.0001
                or int(post_refiner.get("seed", -1)) != 430
                or float(
                    post_refiner.get("maximum_residual_fraction", -1)
                )
                != 0.05
                or post_refiner.get("training_data")
                not in {
                    "organizer_train_only",
                    "organizer_train_plus_val_final",
                }
                or post_refiner.get("initialization")
                != "fresh_vessl_random_init_after_vessl_generalist"
                or post_refiner.get("external_learned_state_imported")
                is not False
                or tactics_recipe.get("inference", {}).get("tta_views")
                != "acc8_identity_flip_lr"
                or (
                    int(post_refiner.get("epochs", -1)) == 15
                    and (
                        post_refiner.get("role")
                        != "main_output_post_refiner"
                        or post_refiner.get("variant") != "NAF_S"
                        or post_refiner.get("late_branch_epochs") != [41, 55]
                        or post_refiner.get("parent_epoch") != 40
                        or post_refiner.get("parent_optimizer_step") != 186880
                        or post_refiner.get("trainable_parameter_scope")
                        != expected_trainable_scope
                        or post_refiner.get("frozen_parameter_scope")
                        != "main_c10_e40_all_parameters"
                        or post_refiner.get("main_parameters_updated") is not False
                        or post_refiner.get("views_batched") is not True
                        or post_refiner.get("no_c10_branch_replication") is not True
                        or post_refiner.get("one_post_refiner_checkpoint_only")
                        is not True
                    )
                )
                or (
                    int(post_refiner.get("epochs", -1)) == 20
                    and (
                        post_refiner.get("role")
                        != "main_output_post_refiner"
                        or post_refiner.get("variant") != "NAF_S"
                        or post_refiner.get("late_branch_epochs") != [52, 71]
                        or post_refiner.get("parent_epoch") != 51
                        or post_refiner.get("parent_optimizer_step") != 238272
                        or post_refiner.get("optimizer_steps") != 91141
                        or post_refiner.get("steps_per_epoch") != 4672
                        or post_refiner.get("trainable_parameter_scope")
                        != expected_trainable_scope
                        or post_refiner.get("frozen_parameter_scope")
                        != "main_c10_e51_all_parameters"
                        or post_refiner.get("main_parameters_updated") is not False
                        or post_refiner.get("views_batched") is not True
                        or post_refiner.get("no_c10_branch_replication") is not True
                        or post_refiner.get("one_post_refiner_checkpoint_only")
                        is not True
                        or post_refiner.get("training_data")
                        != "organizer_train_plus_val_final"
                        or post_refiner.get("extra_train_root")
                        != "/root/Data/val"
                        or post_refiner.get("loss_family")
                        != "winner_foreground_ssim_l1_sqrt_area_v1"
                        or float(post_refiner.get("loss_lambda_l1", -1))
                        != 0.1
                        or post_refiner.get("sqrt_area_weighting") is not True
                        or post_refiner.get(
                            "validation_used_for_checkpoint_selection"
                        )
                        is not False
                    )
                )
                or (
                    int(post_refiner.get("epochs", -1)) == 21
                    and (
                        post_refiner.get("role")
                        != "main_output_post_refiner"
                        or post_refiner.get("variant") != "NAF_S"
                        or post_refiner.get("late_branch_epochs") != [50, 70]
                        or post_refiner.get("parent_epoch")
                        != GENERALIST_HANDOFF_EPOCH
                        or post_refiner.get("parent_optimizer_step")
                        != GENERALIST_HANDOFF_STEP
                        or post_refiner.get("optimizer_steps") != 91231
                        or post_refiner.get("lr_horizon_optimizer_steps")
                        != 93567
                        or post_refiner.get("steps_per_epoch") != 4672
                        or post_refiner.get("trainable_parameter_scope")
                        != expected_trainable_scope
                        or post_refiner.get("frozen_parameter_scope")
                        != "main_c10_e49_all_parameters"
                        or post_refiner.get("main_parameters_updated") is not False
                        or post_refiner.get("views_batched") is not True
                        or post_refiner.get("no_c10_branch_replication") is not True
                        or post_refiner.get("one_post_refiner_checkpoint_only")
                        is not True
                        or post_refiner.get("training_data")
                        != "organizer_train_plus_val_final"
                        or post_refiner.get("extra_train_root")
                        != "/root/Data/val"
                        or post_refiner.get("loss_family")
                        not in {R19_PLAIN_LOSS_FAMILY, R19_BBOX_LOSS_FAMILY}
                        or float(post_refiner.get("loss_lambda_l1", -1))
                        != 0.1
                        or post_refiner.get("sqrt_area_weighting") is not True
                        or post_refiner.get(
                            "validation_used_for_checkpoint_selection"
                        )
                        is not False
                    )
                )
            ):
                raise RuntimeError("invalid score-free post-refiner tactic")
        if SINGLE_FINAL_REQUIRED:
            acc4 = tactics_recipe.get("acc4_specialist", {})
            acc8 = tactics_recipe.get("acc8_specialist", {})
            bbox_enabled = (
                post_refiner.get("loss_family") == R19_BBOX_LOSS_FAMILY
            )
            if (
                post_refiner.get("enabled") is not True
                or post_refiner.get("variant") != "NAF_S"
                or int(post_refiner.get("epochs", -1)) != 21
                or post_refiner.get("late_branch_epochs") != [50, 70]
                or post_refiner.get("parent_epoch")
                != GENERALIST_HANDOFF_EPOCH
                or post_refiner.get("parent_optimizer_step")
                != GENERALIST_HANDOFF_STEP
                or post_refiner.get("optimizer_steps") != 91231
                or post_refiner.get("lr_horizon_optimizer_steps") != 93567
                or post_refiner.get("steps_per_epoch") != 4672
                or post_refiner.get("trainable_parameter_scope") != "naf_s_only"
                or post_refiner.get("frozen_parameter_scope")
                != "main_c10_e49_all_parameters"
                or post_refiner.get("main_parameters_updated") is not False
                or post_refiner.get("views") != ["identity", "flip_lr"]
                or post_refiner.get("mask_conditioned") is not False
                or post_refiner.get("mask_conditioning") is not None
                or post_refiner.get("input_mode")
                != "recon_zero_filled_residual"
                or post_refiner.get("zero_filled_definition")
                != (
                    "rss(fftshift(ifft2(ifftshift(masked_kspace),"
                    "norm=ortho)))"
                )
                or post_refiner.get("normalization")
                != "shared_detached_reconstruction_amax"
                or post_refiner.get("spatial_match")
                != "center_crop_then_zero_pad"
                or post_refiner.get("training_data") != "organizer_train_plus_val_final"
                or post_refiner.get("extra_train_root") != "/root/Data/val"
                or post_refiner.get("loss_family")
                not in {R19_PLAIN_LOSS_FAMILY, R19_BBOX_LOSS_FAMILY}
                or float(post_refiner.get("loss_lambda_l1", -1)) != 0.1
                or post_refiner.get("sqrt_area_weighting") is not True
                or (
                    bbox_enabled
                    and (
                        float(post_refiner.get("bbox_loss_coefficient", -1)) != 0.5
                        or post_refiner.get("organizer_annotations_used_for_training")
                        is not True
                        or post_refiner.get("annotation_source")
                        != "organizer_train_plus_val_h5_annotations_training_only"
                        or post_refiner.get("bbox_coordinate_frame") != [384, 384]
                        or post_refiner.get("bbox_training_alignment")
                        != "test_part_center_crop_then_zero_pad_v1"
                    )
                )
                or (
                    not bbox_enabled
                    and (
                        post_refiner.get("bbox_loss_coefficient") is not None
                        or post_refiner.get("organizer_annotations_used_for_training")
                        is not False
                    )
                )
                or post_refiner.get("inference_annotation_access") is not False
                or post_refiner.get("validation_used_for_checkpoint_selection") is not False
                or post_refiner.get("training_base_route")
                != "exact_acceleration_specialist_before_shared_naf_s_v1"
                or post_refiner.get("sampler_policy")
                != "equal_acc_real_acc8_real80_virtual20_v1"
                or tactics_recipe.get("inference", {}).get("tta_views") != "acc8_identity_flip_lr"
                or acc4.get("enabled") is not True
                or acc4.get("schema") != R19_ACC4_SCHEMA
                or acc4.get("late_branch_parent_epoch")
                != GENERALIST_HANDOFF_EPOCH
                or acc4.get("late_branch_parent_optimizer_step")
                != GENERALIST_HANDOFF_STEP
                or acc4.get("epochs") != 2
                or acc4.get("optimizer_steps") != 4672
                or acc4.get("lr_horizon_optimizer_steps") != 35040
                or float(acc4.get("peak_lr", -1)) != 2.5e-5
                or acc4.get("loss_family") != "exact_upstream_ssim"
                or acc4.get("mraugment") != "conservative_immediate"
                or acc4.get("train_acceleration") != "acc4"
                or acc4.get("deployment_scope") != "acc4_only_router"
                or acc4.get("trainable_parameter_scope") != "all_registered_trainable_parameters"
                or acc8.get("enabled") is not True
                or acc8.get("schema") != R19_ACC8_SCHEMA
                or acc8.get("late_branch_parent_epoch")
                != GENERALIST_HANDOFF_EPOCH
                or acc8.get("late_branch_parent_optimizer_step")
                != GENERALIST_HANDOFF_STEP
                or acc8.get("epochs") != 1
                or acc8.get("optimizer_steps") != 1158
                or acc8.get("lr_horizon_optimizer_steps") != 2315
                or float(acc8.get("peak_lr", -1)) != 5e-5
                or acc8.get("train_acceleration") != "acc8"
                or acc8.get("deployment_scope") != "acc8_only_router"
                or acc8.get("trainable_parameter_scope") != "all_registered_trainable_parameters"
                or acc8.get("loss_family")
                != "r10_image_masked_ssim_valid_windows_mean"
                or acc8.get("mraugment") != "off"
            ):
                raise RuntimeError(
                    "single-final policy requires the sealed R25 E49 + ACC4-E2-prefix + ACC8-R10-prefix + NAF_S-91231/93567-horizon routed package"
                )
        candidate_policy = tactics.get("candidate_policy")
        if SINGLE_FINAL_REQUIRED and (
            not isinstance(candidate_policy, dict)
            or candidate_policy.get("candidate_count_max") != 1
            or candidate_policy.get("fallback_registered") is not False
            or candidate_policy.get("fallback_on_post_refiner_failure") is not False
            or candidate_policy.get("multiple_candidates_allowed") is not False
        ):
            raise RuntimeError("single-final candidate policy is absent or invalid")
        if winner not in {"R1", *REGISTERED}:
            raise RuntimeError(f"unregistered G10 winner: {winner}")
        return winner, arch_recipe, tactics_recipe


def refresh_terminal_tactics(
    expected_winner: str,
    expected_recipe: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Reload score-free terminal tactics after generalist training.

    Architecture identity is immutable once VESSL scratch training starts, but
    score-free specialist/refiner decisions may be sealed while that long run is
    in progress.  Reusing the process-start copy would silently discard those
    decisions.  Revalidate both handoffs and accept only a tactics-only refresh.
    """
    if TERMINAL_TACTICS_WAIT_SECONDS < 0:
        raise RuntimeError("terminal tactics wait must be non-negative")
    deadline = time.monotonic() + TERMINAL_TACTICS_WAIT_SECONDS
    while (
        not TERMINAL_TACTICS_HANDOFF.is_file()
        and time.monotonic() < deadline
    ):
        remaining = max(0.0, deadline - time.monotonic())
        update(
            "WAITING_TERMINAL_TACTICS_HANDOFF",
            winner=expected_winner,
            terminal_tactics_handoff=str(TERMINAL_TACTICS_HANDOFF),
            fallback_tactics_handoff=str(TACTICS_HANDOFF),
            wait_remaining_seconds=remaining,
        )
        time.sleep(min(POLL_SECONDS, remaining))
    if TERMINAL_TACTICS_HANDOFF.is_file():
        selected_path = TERMINAL_TACTICS_HANDOFF
        refresh_source = "TERMINAL_G11_HANDOFF"
    else:
        if REQUIRE_TERMINAL_TACTICS_HANDOFF:
            update(
                "FINAL_TACTICS_REQUIRED_NO_FALLBACK",
                winner=expected_winner,
                terminal_tactics_handoff=str(TERMINAL_TACTICS_HANDOFF),
                wait_seconds=TERMINAL_TACTICS_WAIT_SECONDS,
            )
            raise RuntimeError(
                "terminal G11 tactics handoff is required; refusing the "
                "base tactics fallback"
            )
        selected_path = TACTICS_HANDOFF
        refresh_source = "BASE_G11_TIMEOUT_FALLBACK"
    winner, recipe, tactics = validate_handoffs(
        tactics_path=selected_path,
    )
    if winner != expected_winner:
        raise RuntimeError(
            "generalist architecture winner changed after scratch training"
        )
    if recipe != expected_recipe:
        raise RuntimeError(
            "generalist architecture recipe changed after scratch training"
        )
    update(
        "TERMINAL_TACTICS_REFRESHED",
        winner=winner,
        refresh_source=refresh_source,
        tactics_handoff_path=str(selected_path),
        architecture_handoff_sha256=sha256(ARCH_HANDOFF),
        tactics_handoff_sha256=sha256(selected_path),
    )
    return tactics, selected_path


def wait_fallback() -> tuple[Path, str]:
    while True:
        for contract_path, schema in (
            (
                R1_E35_CONTRACT,
                "vessl-generalist-e35-model-only-parent-v1",
            ),
            (
                R1_HARDSTART_CONTRACT,
                "vessl-generalist-hardstart-model-only-parent-v1",
            ),
        ):
            contract = load_json(contract_path)
            checkpoint = Path(str(contract.get("checkpoint", "")))
            digest = str(contract.get("checkpoint_sha256", ""))
            if (
                contract.get("schema") == schema
                and contract.get("state") == "SEALED"
                and checkpoint.is_file()
                and len(digest) == 64
                and sha256(checkpoint) == digest
            ):
                return checkpoint, digest
        update(
            "WAITING_R1_FALLBACK",
            accepted_contracts=[
                str(R1_E35_CONTRACT),
                str(R1_HARDSTART_CONTRACT),
            ],
        )
        time.sleep(POLL_SECONDS)


def command_line(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(
            b"\0", b" "
        ).decode(errors="replace")
    except OSError:
        return None


def process_argv(pid: int) -> list[str]:
    try:
        return [
            item.decode(errors="replace")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
    except OSError:
        return []


def argv_matches_fragment(argv: list[str], fragment: str) -> bool:
    """Match an owned process without accepting diagnostic shell text."""
    if not argv:
        return False
    executable = Path(argv[0]).name.lower()
    if fragment.endswith(".py"):
        return "python" in executable and any(
            Path(item).name == fragment for item in argv[1:]
        )
    if fragment.endswith(".sh"):
        return executable in {"bash", "sh"} and any(
            Path(item).name == fragment for item in argv[1:]
        )
    # Run identities such as --net-name values must be distinct argv items.
    # A bash/ssh diagnostic whose single command-string merely mentions the
    # identity is not an owned trainer.
    return fragment in argv[1:]


def matching_pids(fragment: str) -> list[int]:
    result = []
    for entry in Path("/proc").glob("[0-9]*"):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        argv = process_argv(int(entry.name))
        if argv_matches_fragment(argv, fragment):
            result.append(int(entry.name))
    return sorted(result)


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return False
    return state != "Z"


def stop_legacy_dispatcher() -> dict[str, list[int]]:
    dispatchers = matching_pids(LEGACY_DISPATCHER_FRAGMENT)
    loops = matching_pids(LEGACY_LOOP_FRAGMENT)
    targets = dispatchers + loops
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while any(alive(pid) for pid in targets) and time.monotonic() < deadline:
        time.sleep(0.5)
    remaining = [pid for pid in targets if alive(pid)]
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)
    if any(alive(pid) for pid in targets):
        raise RuntimeError("legacy post-E35 dispatcher did not stop")
    return {"dispatcher_pids": dispatchers, "loop_pids": loops}


def install_sources(mapping: dict[Path, Path]) -> dict[str, str]:
    backup = CONTROL / "source-backup"
    backup.mkdir(parents=True, exist_ok=True)
    observed = {}
    for source, destination in mapping.items():
        expected = EXPECTED_STAGED_SHA256[source]
        if sha256(source) != expected:
            raise RuntimeError(f"source changed before install: {source}")
        if destination.exists():
            prior = backup / destination.name
            if not prior.exists():
                shutil.copy2(destination, prior)
                prior.chmod(0o444)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        py_compile.compile(str(destination), doraise=True)
        observed[str(destination)] = sha256(destination)
    return observed


def latest_checkpoint(run: Path) -> tuple[Path, str] | None:
    descriptor = load_json(run / "checkpoints/latest.json")
    raw = descriptor.get("path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        return None
    return path, sha256(path)


def is_safe_prestep_specialist_run(run: Path) -> bool:
    """Allow one idempotent retry only before any learned state was written."""
    if not run.is_dir():
        return False
    safe_entries = {
        "checkpoints",
        "resolved_config.json",
        "legal-mask-family-manifest.json",
        "feasibility.json",
    }
    observed_entries = {path.name for path in run.iterdir()}
    if not observed_entries.issubset(safe_entries):
        return False
    checkpoint_dir = run / "checkpoints"
    return not checkpoint_dir.is_dir() or not any(checkpoint_dir.iterdir())


def terminal_checkpoint(
    recipe: dict[str, Any],
    run: Path,
) -> tuple[Path, str] | None:
    """Return the exact E49 handoff on the unchanged cosine-51 horizon."""
    target_step = GENERALIST_HANDOFF_STEP
    target_epoch = GENERALIST_HANDOFF_EPOCH
    expected_step = 4672 * target_epoch
    if target_step != expected_step:
        raise RuntimeError(
            "terminal optimizer step does not match the full balanced epoch"
        )
    if (
        int(recipe["epochs"]) != GENERALIST_SCHEDULER_EPOCHS
        or int(recipe["total_steps"]) != GENERALIST_SCHEDULER_STEPS
    ):
        raise RuntimeError("selected architecture must retain cosine-51 horizon")

    receipt_path = run / "controller-terminal-checkpoint.json"
    receipt = load_json(receipt_path)
    checkpoint = Path(str(receipt.get("checkpoint", "")))
    digest = str(receipt.get("checkpoint_sha256", ""))
    if (
        receipt.get("state") == "SEALED"
        and int(receipt.get("epoch", -1)) == target_epoch
        and int(receipt.get("optimizer_step", -1)) == target_step
        and checkpoint.is_file()
        and len(digest) == 64
        and sha256(checkpoint) == digest
    ):
        return checkpoint, digest

    candidates = sorted(
        (run / "checkpoints").glob(
            f"checkpoint-*-{target_step:09d}.pt"
        )
    )
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(
            f"ambiguous terminal checkpoints at step {target_step}: "
            f"{candidates}"
        )
    checkpoint = candidates[0]
    digest = sha256(checkpoint)
    atomic_json(
        receipt_path,
        {
            "schema": "vessl-g10-generalist-terminal-checkpoint-v1",
            "state": "SEALED",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "epoch": target_epoch,
            "optimizer_step": target_step,
            "scheduler_horizon_epochs": int(recipe["epochs"]),
            "scheduler_total_steps": int(recipe["total_steps"]),
            "leaderboard_data_used": False,
            "external_learned_state_imported": False,
        },
    )
    return checkpoint, digest


def trainer_pids(run_name: str) -> list[int]:
    return [
        pid
        for pid in matching_pids(run_name)
        if (command_line(pid) or "").find("train.py") >= 0
    ]


def stop_trainer(pid: int) -> None:
    """Stop only the selected training child after its terminal checkpoint."""
    try:
        group = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        if group == pid:
            os.killpg(group, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 120
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if alive(pid):
        try:
            if group == pid:
                os.killpg(group, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if alive(pid):
        raise RuntimeError(f"selected trainer {pid} did not stop")


def train_command(
    recipe: dict[str, Any],
    run: Path,
    resume: tuple[Path, str] | None,
) -> list[str]:
    command = [
        "/bin/python",
        "-u",
        "train.py",
        "--model-family",
        "promptmr-plus",
        "--promptmr-production",
        "--promptmr-rung",
        str(recipe["rung"]),
        "--promptmr-num-cascades",
        str(int(recipe["num_cascades"])),
        "--promptmr-n-history",
        str(int(recipe["n_history"])),
        "--promptmr-compact-fallback",
        "--promptmr-train-acceleration",
        "all",
        "--promptmr-mraugment",
        str(recipe["mraugment"]),
        "--promptmr-legal-mask-family",
        "--promptmr-lr-schedule",
        "cos50",
        "--promptmr-skip-validation",
        "--precision",
        "fp32",
        "--require-cuda-device-name",
        "NVIDIA GeForce GTX 1080",
        "--GPU-NUM",
        "0",
        "--batch-size",
        "1",
        "--num-epochs",
        str(int(recipe["epochs"])),
        "--lr",
        repr(float(recipe["learning_rate"])),
        "--seed",
        str(int(recipe["seed"])),
        "--net-name",
        run.name,
        "--data-path-train",
        "/root/Data/train",
        "--trusted-data-manifest",
        (
            "/root/result/EXP_FI_ACC8_CKPT_BASE_E30_R1/"
            "fi-acc8-full-training/provenance.json"
        ),
        "--report-interval",
        "50",
    ]
    if resume is not None:
        command.extend(
            [
                "--resume-checkpoint",
                str(resume[0]),
                "--resume-checkpoint-sha256",
                resume[1],
            ]
        )
    return command


def train_generalist(
    winner: str,
    recipe: dict[str, Any],
) -> tuple[Path, str, Path]:
    registration = REGISTERED[winner]
    run = RESULT / registration["run_name"]
    recoveries = 0
    while True:
        terminal = terminal_checkpoint(recipe, run)
        if terminal is not None:
            for pid in trainer_pids(run.name):
                stop_trainer(pid)
            update(
                "SELECTED_ARCHITECTURE_E49_SEALED",
                winner=winner,
                checkpoint=str(terminal[0]),
                checkpoint_sha256=terminal[1],
                terminal_epoch=GENERALIST_HANDOFF_EPOCH,
                scheduler_horizon_epochs=int(recipe["epochs"]),
            )
            return terminal[0], terminal[1], run
        if time.time() >= FINAL_RESEARCH_CUTOFF_UNIX:
            for pid in trainer_pids(run.name):
                stop_trainer(pid)
            update(
                "SELECTED_ARCHITECTURE_CUTOFF_R1_FALLBACK",
                winner=winner,
                cutoff_unix=FINAL_RESEARCH_CUTOFF_UNIX,
            )
            raise FinalResearchCutoff(
                "selected architecture missed the final research cutoff"
            )
        terminal = load_json(run / "terminal.json")
        if terminal.get("status") == "COMPLETED":
            raise RuntimeError(
                "selected generalist completed without its exact E49 "
                "terminal checkpoint being sealed"
            )
        existing = trainer_pids(run.name)
        if len(existing) > 1:
            raise RuntimeError(f"duplicate selected trainers: {existing}")
        if existing:
            update(
                "ADOPTING_SELECTED_ARCHITECTURE_TRAINER",
                winner=winner,
                trainer_pid=existing[0],
                recoveries=recoveries,
            )
            while alive(existing[0]):
                terminal = terminal_checkpoint(recipe, run)
                if terminal is not None:
                    stop_trainer(existing[0])
                    break
                if time.time() >= FINAL_RESEARCH_CUTOFF_UNIX:
                    stop_trainer(existing[0])
                    raise FinalResearchCutoff(
                        "adopted selected trainer reached final cutoff"
                    )
                time.sleep(TRAINING_POLL_SECONDS)
            continue
        resume = latest_checkpoint(run) if run.exists() else None
        if run.exists() and resume is None:
            raise RuntimeError("selected run exists without recoverable checkpoint")
        command = train_command(recipe, run, resume)
        LOGS.mkdir(parents=True, exist_ok=True)
        update(
            "TRAINING_SELECTED_ARCHITECTURE",
            winner=winner,
            run=str(run),
            recoveries=recoveries,
            resume_checkpoint=str(resume[0]) if resume else None,
        )
        with (LOGS / f"{run.name}.log").open("ab") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            while process.poll() is None:
                terminal = terminal_checkpoint(recipe, run)
                if terminal is not None:
                    stop_trainer(process.pid)
                    break
                if time.time() >= FINAL_RESEARCH_CUTOFF_UNIX:
                    stop_trainer(process.pid)
                    raise FinalResearchCutoff(
                        "selected trainer reached final cutoff"
                    )
                time.sleep(TRAINING_POLL_SECONDS)
            returncode = process.wait()
        if terminal_checkpoint(recipe, run) is not None:
            continue
        terminal = load_json(run / "terminal.json")
        if terminal.get("status") == "COMPLETED":
            continue
        recoveries += 1
        if recoveries > MAX_RECOVERIES or latest_checkpoint(run) is None:
            raise RuntimeError(
                "selected architecture exhausted recovery: "
                f"returncode={returncode}, terminal={terminal}"
            )
        time.sleep(30)


def specialist_command(
    generalist: dict[str, Any],
    specialist: dict[str, Any],
    run: Path,
    parent: tuple[Path, str],
    resume: tuple[Path, str] | None,
) -> list[str]:
    acceleration = str(specialist.get("train_acceleration", ""))
    if acceleration not in {"acc4", "acc8"}:
        raise RuntimeError("specialist acceleration route is invalid")
    kind = mask_conditioning_kind(specialist)
    recipe_scope = specialist.get("trainable_parameter_scope")
    scope_id = {
        None: "all",
        "all_registered_trainable_parameters": "all",
        "core.cascades.8-11": "last4",
        "core.sens_net+core.cascades.8-11": "sens_last4",
        (
            "core.sens_net+core.cascades.8-11+core.mask_conditioner"
        ): "sens_last4",
        (
            "core.sens_net+core.cascades.8-11+"
            "core.mask_prompt_conditioner"
        ): "sens_last4",
        # The wrapper first asks the pinned production runtime for its
        # registered sens_last4 contract, then replaces ownership with the
        # exact adapter-only prefixes from the sealed scalar profile.
        "core.mask_prompt_conditioner": "sens_last4",
    }.get(recipe_scope)
    if recipe_scope == "core.mask_prompt_conditioner":
        scope_id = (
            "c10_last_cascade_adapter"
            if kind == "c10_last_cascade_prompt"
            else "sens_last4"
        )
    if scope_id is None:
        raise RuntimeError("specialist trainable parameter scope is invalid")
    if kind is not None:
        for composition_key in (
            "composition",
            "post_specialist_composition",
        ):
            composition = specialist.get(composition_key)
            if composition is not None and (
                not isinstance(composition, dict)
                or float(composition.get("reconstruction_alpha", -1)) != 1.0
                or float(composition.get("sensitivity_alpha", -1)) != 1.0
            ):
                raise RuntimeError(
                    "conditioned specialists cannot be folded into an "
                    "unconditioned parent"
                )
    command = ["/bin/python", "-u"]
    if kind is None:
        command.append("train.py")
    else:
        command.extend(
            [
                str(STAGED_MASK_SPECIALIST_TRAIN),
                "--mask-conditioning-profile-json",
                json.dumps(
                    specialist,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            ]
        )
    command.extend([
        "--model-family",
        "promptmr-plus",
        "--promptmr-production",
        "--promptmr-rung",
        str(generalist["rung"]),
        "--promptmr-num-cascades",
        str(int(generalist["num_cascades"])),
        "--promptmr-n-history",
        str(int(generalist["n_history"])),
        "--promptmr-compact-fallback",
        "--promptmr-train-acceleration",
        acceleration,
        "--promptmr-mraugment",
        str(specialist["mraugment"]),
        "--promptmr-legal-mask-family",
        "--promptmr-lr-schedule",
        str(specialist["lr_schedule"]),
        "--promptmr-skip-validation",
        "--precision",
        "fp32",
        "--require-cuda-device-name",
        "NVIDIA GeForce GTX 1080",
        "--GPU-NUM",
        "0",
        "--batch-size",
        "1",
        "--num-epochs",
        str(int(specialist["epochs"])),
        "--promptmr-stop-after-optimizer-steps",
        str(int(specialist["optimizer_steps"])),
        "--lr",
        repr(float(specialist["peak_lr"])),
        "--seed",
        str(int(specialist["seed"])),
        "--net-name",
        run.name,
        "--data-path-train",
        "/root/Data/train",
        "--data-path-val",
        "/root/Data/val",
        "--trusted-data-manifest",
        (
            "/root/result/EXP_FI_ACC8_CKPT_BASE_E30_R1/"
            "fi-acc8-full-training/provenance.json"
        ),
        "--report-interval",
        "50",
    ])
    optional_scalar_flags = (
        ("--promptmr-mraugment-seed", "mraugment_seed"),
        ("--promptmr-legal-mask-seed", "legal_mask_seed"),
        (
            "--promptmr-specialist-lr-horizon-optimizer-steps",
            "lr_horizon_optimizer_steps",
        ),
    )
    for flag, key in optional_scalar_flags:
        value = specialist.get(key)
        if value is not None:
            command.extend([flag, str(int(value))])
    command.extend(
        [
        "--promptmr-specialist-loss-family",
        str(specialist.get("loss_family", "exact_upstream_ssim")),
        "--promptmr-specialist-trainable-scope",
        scope_id,
        ]
    )
    if resume is None:
        command.extend(
            [
                "--promptmr-vessl-model-only-import",
                str(parent[0]),
                "--promptmr-vessl-model-only-import-sha256",
                parent[1],
            ]
        )
    else:
        command.extend(
            [
                "--resume-checkpoint",
                str(resume[0]),
                "--resume-checkpoint-sha256",
                resume[1],
            ]
        )
    return command


def install_specialization_sources() -> dict[str, str]:
    expected_current = {
        TRAIN_SOURCE: EXPECTED_STAGED_SHA256[STAGED_TRAIN],
        PRODUCTION_SOURCE: EXPECTED_STAGED_SHA256[STAGED_PRODUCTION],
    }
    expected_specialist = {
        TRAIN_SOURCE: EXPECTED_STAGED_SHA256[STAGED_SPECIALIST_TRAIN],
        PRODUCTION_SOURCE: EXPECTED_STAGED_SHA256[
            STAGED_SPECIALIST_PRODUCTION
        ],
    }
    current = {
        TRAIN_SOURCE: sha256(TRAIN_SOURCE),
        PRODUCTION_SOURCE: sha256(PRODUCTION_SOURCE),
    }
    if current == expected_specialist:
        return {str(path): digest for path, digest in current.items()}
    if current != expected_current:
        raise RuntimeError(
            f"unexpected source before specialist install: {current}"
        )
    installed = install_sources(
        {
            STAGED_SPECIALIST_TRAIN: TRAIN_SOURCE,
            STAGED_SPECIALIST_PRODUCTION: PRODUCTION_SOURCE,
        }
    )
    if installed != {
        str(path): digest for path, digest in expected_specialist.items()
    }:
        raise RuntimeError("specialization source installation mismatch")
    return installed


def mask_conditioning_kind(specialist: dict[str, Any]) -> str | None:
    present = [
        kind
        for kind, key in (
            ("dc", "mask_conditioned_dc"),
            ("prompt", "mask_conditioned_prompt"),
            (
                "c10_last_cascade_prompt",
                "mask_conditioned_c10_last_cascade_prompt",
            ),
        )
        if specialist.get(key) is not None
    ]
    if len(present) > 1:
        raise RuntimeError("only one mask-conditioning tactic may be active")
    return present[0] if present else None


def install_mask_conditioning_sources() -> dict[str, str]:
    """Install code-only runtime bytes; never import research learned state."""

    mapping = {
        STAGED_MASK_CONDITIONING: MASK_CONDITIONING_DESTINATION,
        STAGED_MASK_DC: MASK_DC_DESTINATION,
        STAGED_MASK_PROMPT: MASK_PROMPT_DESTINATION,
    }
    current = {
        destination: sha256(destination)
        for destination in mapping.values()
        if destination.is_file()
    }
    expected = {
        destination: EXPECTED_STAGED_SHA256[source]
        for source, destination in mapping.items()
    }
    if current == expected:
        return {str(path): digest for path, digest in current.items()}
    if current and any(
        path in current and current[path] != digest
        for path, digest in expected.items()
    ):
        raise RuntimeError(
            f"unexpected mask-conditioning runtime before install: {current}"
        )
    installed = install_sources(mapping)
    if installed != {
        str(path): digest for path, digest in expected.items()
    }:
        raise RuntimeError("mask-conditioning runtime installation mismatch")
    return installed


def specialist_composition_command(
    parent: tuple[Path, str],
    specialist_checkpoint: tuple[Path, str],
    composition: dict[str, Any],
    output: Path,
) -> list[str]:
    if (
        composition.get("operation")
        != "post_specialist_model_only_delta_merge"
        or composition.get("source_weights")
        != "VESSL_TRAINED_GENERALIST_AND_SPECIALIST_ONLY"
    ):
        raise RuntimeError("unsupported ACC8 specialist composition contract")
    reconstruction_alpha = float(composition.get("reconstruction_alpha", -1))
    sensitivity_alpha = float(composition.get("sensitivity_alpha", -1))
    if not (
        0.0 <= reconstruction_alpha <= 1.0
        and 0.0 <= sensitivity_alpha <= 1.0
    ):
        raise RuntimeError("ACC8 specialist composition alpha is invalid")
    return [
        "/bin/python",
        "-u",
        str(STAGED_SPECIALIST_COMPOSER),
        "--generalist-checkpoint",
        str(parent[0]),
        "--generalist-sha256",
        parent[1],
        "--specialist-checkpoint",
        str(specialist_checkpoint[0]),
        "--specialist-sha256",
        specialist_checkpoint[1],
        "--reconstruction-alpha",
        repr(reconstruction_alpha),
        "--sensitivity-alpha",
        repr(sensitivity_alpha),
        "--output",
        str(output),
    ]


def checkpoint_at_step(run: Path, step: int) -> tuple[Path, str]:
    candidates = sorted(
        (run / "checkpoints").glob(f"checkpoint-*-{int(step):09d}.pt")
    )
    if not candidates:
        raise RuntimeError(f"specialist checkpoint step {step} is absent")
    preferred = sorted(
        candidates,
        key=lambda path: (
            "-budget-" in path.name,
            "-last-" in path.name,
            path.stat().st_mtime_ns,
        ),
        reverse=True,
    )[0]
    return preferred, sha256(preferred)


def specialist_soup_command(
    earlier: tuple[Path, str],
    later: tuple[Path, str],
    soup: dict[str, Any],
    output: Path,
) -> list[str]:
    if (
        soup.get("enabled") is not True
        or soup.get("source")
        != "VESSL_TRAINED_SPECIALIST_CHECKPOINTS_ONLY"
    ):
        raise RuntimeError("unsupported specialist checkpoint soup contract")
    return [
        "/bin/python",
        "-u",
        str(STAGED_SPECIALIST_SOUP_COMPOSER),
        "--earlier-checkpoint",
        str(earlier[0]),
        "--earlier-sha256",
        earlier[1],
        "--later-checkpoint",
        str(later[0]),
        "--later-sha256",
        later[1],
        "--later-weight",
        repr(float(soup["later_weight"])),
        "--output",
        str(output),
    ]


def compose_specialist_soup(
    run: Path,
    latest: tuple[Path, str],
    specialist: dict[str, Any],
) -> tuple[Path, str]:
    soup = specialist.get("checkpoint_soup")
    if not isinstance(soup, dict) or soup.get("enabled") is not True:
        return latest
    earlier_step = int(soup.get("earlier_optimizer_step", -1))
    later_step = int(soup.get("later_optimizer_step", -1))
    later_weight = float(soup.get("later_weight", -1))
    if (
        (earlier_step, later_step, later_weight)
        not in {
            (500, 1000, 0.5),
            (1000, 1158, 0.5),
            (1000, 1158, 0.75),
        }
        or int(specialist.get("optimizer_steps", -1)) != later_step
    ):
        raise RuntimeError("selected specialist soup is invalid")
    earlier = checkpoint_at_step(run, earlier_step)
    if not latest[0].name.endswith(f"-{later_step:09d}.pt"):
        raise RuntimeError(
            "terminal specialist checkpoint is not the selected soup endpoint"
        )
    # Use the terminal checkpoint as the later endpoint. A milestone and the
    # terminal save can contain identical learned tensors but different
    # serialization/provenance bytes.
    later = latest
    output = run / "derived" / (
        f"specialist-soup-{earlier_step}-{later_step}-"
        f"late{str(later_weight).replace('.', 'p')}.pt"
    )
    command = specialist_soup_command(earlier, later, soup, output)
    update(
        "COMPOSING_VESSL_SPECIALIST_CHECKPOINT_SOUP",
        specialist_soup_command=command,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    receipt = load_json(output.with_suffix(output.suffix + ".receipt.json"))
    if (
        completed.returncode != 0
        or receipt.get("state") != "PASS"
        or receipt.get("output") != str(output.resolve())
        or receipt.get("output_sha256") != sha256(output)
        or int(receipt.get("earlier_optimizer_step", -1)) != earlier_step
        or int(receipt.get("later_optimizer_step", -1)) != later_step
        or float(receipt.get("later_weight", -1)) != later_weight
        or receipt.get("all_source_weights_trained_on_vessl") is not True
        or receipt.get("external_learned_state_imported") is not False
    ):
        raise RuntimeError(
            "VESSL specialist checkpoint soup failed: "
            f"rc={completed.returncode} output={completed.stdout[-2000:]}"
        )
    return output, str(receipt["output_sha256"])


def compose_specialist_checkpoint(
    parent: tuple[Path, str],
    specialist_checkpoint: tuple[Path, str],
    specialist: dict[str, Any],
    run: Path,
) -> tuple[Path, str]:
    composition = specialist.get("composition")
    if not isinstance(composition, dict):
        return specialist_checkpoint
    reconstruction_alpha = float(
        composition.get("reconstruction_alpha", -1)
    )
    sensitivity_alpha = float(composition.get("sensitivity_alpha", -1))
    if reconstruction_alpha == 1.0 and sensitivity_alpha == 1.0:
        return specialist_checkpoint
    output = run / "composed" / "checkpoint-acc8-model-only-composed.pt"
    receipt_path = run / "composed" / "receipt.json"
    command = specialist_composition_command(
        parent,
        specialist_checkpoint,
        composition,
        output,
    )
    if output.is_file() or receipt_path.is_file():
        receipt = load_json(receipt_path)
        if (
            not output.is_file()
            or receipt.get("state") != "PASS"
            or receipt.get("output_sha256") != sha256(output)
            or receipt.get("generalist_sha256") != parent[1]
            or receipt.get("specialist_sha256") != specialist_checkpoint[1]
            or receipt.get("composition") != composition
        ):
            raise RuntimeError("existing specialist composition is invalid")
        return output, str(receipt["output_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / "compose.log"
    with log_path.open("ab") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"ACC8 specialist composition failed: {completed.returncode}"
        )
    output_sha256 = sha256(output)
    atomic_json(
        receipt_path,
        {
            "schema": "vessl-acc8-specialist-composition-receipt-v1",
            "state": "PASS",
            "output": str(output),
            "output_sha256": output_sha256,
            "generalist": str(parent[0]),
            "generalist_sha256": parent[1],
            "specialist": str(specialist_checkpoint[0]),
            "specialist_sha256": specialist_checkpoint[1],
            "composition": composition,
            "all_source_weights_trained_on_vessl": True,
            "external_learned_state_imported": False,
            "created_unix": time.time(),
        },
    )
    return output, output_sha256


def train_specialist(
    winner: str,
    generalist: dict[str, Any],
    specialist: dict[str, Any],
    parent: tuple[Path, str],
) -> tuple[Path, str, Path, dict[str, str]]:
    if specialist.get("enabled") is not True:
        return parent[0], parent[1], Path(), {}
    if str(specialist.get("rung")) != str(generalist["rung"]):
        raise RuntimeError("specialist/generalist rung mismatch")
    if int(specialist.get("n_history", -1)) != int(
        generalist["n_history"]
    ):
        raise RuntimeError("specialist/generalist history mismatch")
    acceleration = str(specialist.get("train_acceleration", ""))
    optimizer_steps = int(specialist.get("optimizer_steps", 0))
    requested_epochs = int(specialist.get("epochs", 0))
    late_acc8_moe_budgets = {
        1: 2_315,
        3: 6_945,
        5: 11_575,
        7: 16_205,
        10: 23_150,
        15: 34_725,
    }
    r19_acc8_moe = (
        acceleration == "acc8"
        and specialist.get("schema") == R19_ACC8_SCHEMA
        and requested_epochs == 1
        and optimizer_steps == 1_158
    )
    late_acc8_moe = (
        r19_acc8_moe
        or (
            acceleration == "acc8"
            and specialist.get("schema")
            in {"vessl-score-free-acc8-late-moe-e40-e55-v1", "vessl-score-free-acc8-late-moe-e51-asymmetric-v1", "vessl-score-free-acc8-late-moe-e51-quality-recovery-v1"}
            and late_acc8_moe_budgets.get(requested_epochs) == optimizer_steps
        )
    )
    late_acc4_moe_budgets = {
        1: 2_336,
        2: 4_672,
        3: 7_008,
        5: 11_680,
        7: 16_352,
        10: 23_360,
        15: 35_040,
    }
    late_acc4_moe = (
        acceleration == "acc4"
        and specialist.get("schema")
        in {"vessl-score-free-acc4-late-moe-e40-e55-v1", "vessl-score-free-acc4-late-moe-e51-asymmetric-v1", R19_ACC4_SCHEMA}
        and late_acc4_moe_budgets.get(requested_epochs) == optimizer_steps
    )
    c10_adapter_late = (
        acceleration in {"acc4", "acc8"}
        and specialist.get("schema")
        == "vessl-score-free-c10-last-cascade-adapter-e40-e55-v1"
        and requested_epochs == 15
        and optimizer_steps
        == (35_040 if acceleration == "acc4" else 34_725)
    )
    late_moe = late_acc4_moe or late_acc8_moe or c10_adapter_late
    asymmetric_e51 = specialist.get("schema") in {
        "vessl-score-free-acc4-late-moe-e51-asymmetric-v1",
        "vessl-score-free-acc8-late-moe-e51-asymmetric-v1",
        "vessl-score-free-acc8-late-moe-e51-quality-recovery-v1",
        R19_ACC4_SCHEMA,
        R19_ACC8_SCHEMA,
    }
    if asymmetric_e51 and (
        specialist.get("late_branch_parent_epoch")
        != GENERALIST_HANDOFF_EPOCH
        or specialist.get("late_branch_parent_optimizer_step")
        != GENERALIST_HANDOFF_STEP
    ):
        raise RuntimeError("asymmetric specialist parent contract is invalid")
    expected_epochs = (
        requested_epochs
        if late_moe
        else {"acc4": 3, "acc8": 1}.get(acceleration)
    )
    allowed_steps = (
        {optimizer_steps}
        if late_moe
        else {
            "acc4": {3504},
            "acc8": {500, 1000, 1158, 1500, 2000, 2315},
        }.get(acceleration)
    )
    if expected_epochs is None or allowed_steps is None:
        raise RuntimeError("selected specialist acceleration is invalid")
    deployment_scope = specialist.get(
        "deployment_scope",
        f"{acceleration}_only_router",
    )
    if deployment_scope not in {
        "acc4_only_router",
        "acc8_only_router",
        "global_successor",
    }:
        raise RuntimeError("selected specialist deployment scope is invalid")
    if (
        acceleration == "acc4"
        and deployment_scope == "acc8_only_router"
    ) or (
        acceleration == "acc8"
        and deployment_scope == "acc4_only_router"
    ):
        raise RuntimeError("specialist route and deployment scope mismatch")
    if (
        requested_epochs != expected_epochs
        or optimizer_steps not in allowed_steps
    ):
        raise RuntimeError(
            f"selected {acceleration} specialist budget is invalid"
        )
    conditioning_kind = mask_conditioning_kind(specialist)
    adapter_only_prompt = (
        conditioning_kind == "prompt"
        and specialist.get("trainable_parameter_scope")
        == "core.mask_prompt_conditioner"
    )
    r25_acc4_prefix = (
        acceleration == "acc4"
        and specialist.get("schema") == R19_ACC4_SCHEMA
        and requested_epochs == 2
        and optimizer_steps == 4_672
    )
    maximum_peak_lr = (
        5e-5
        if r19_acc8_moe
        else 2.5e-5
        if r25_acc4_prefix
        else 2e-5
        if late_moe
        else 3e-4 if adapter_only_prompt else 1e-4
    )
    if not 1e-6 <= float(specialist.get("peak_lr", 0)) <= maximum_peak_lr:
        raise RuntimeError("selected specialist LR is outside contract")
    if specialist.get("mraugment") not in {
        "off",
        "conservative_immediate",
        "conservative_delay2",
        "conservative_delay5",
    }:
        raise RuntimeError("selected specialist MRAugment is unsupported")
    expected_schedule = (
        "specialist_warmup1_cosine"
        if late_moe
        else "specialist_warmup1"
    )
    if specialist.get("lr_schedule") != expected_schedule:
        raise RuntimeError("selected specialist schedule is unsupported")
    quality_recovery_acc8 = (
        specialist.get("schema")
        in {
            "vessl-score-free-acc8-late-moe-e51-quality-recovery-v1",
            R19_ACC8_SCHEMA,
        }
    )
    expected_mraugment = (
        "off" if quality_recovery_acc8 else "conservative_immediate"
    )
    if late_moe and specialist.get("mraugment") != expected_mraugment:
        raise RuntimeError("late MoE MRAugment contract mismatch")
    if quality_recovery_acc8 and specialist.get("loss_family") != (
        "r10_image_masked_ssim_valid_windows_mean"
    ):
        raise RuntimeError("selected R10-prefix ACC8 loss contract mismatch")
    conditioning_contracts = {
        "dc": {
            "key": "mask_conditioned_dc",
            "schema": "promptmr-mask-conditioned-dc-v1",
            "parameter_count": 64_535_833,
            "trainable_parameter_count": 22_755_473,
            "scope": (
                "core.sens_net+core.cascades.8-11+core.mask_conditioner"
            ),
        },
        "prompt": {
            "key": "mask_conditioned_prompt",
            "schema": "promptmr-mask-conditioned-prompt-logits-v1",
            "parameter_count": 64_536_649,
            "trainable_variants": {
                (
                    "core.sens_net+core.cascades.8-11+"
                    "core.mask_prompt_conditioner"
                ): (22_756_289, "core.cascades.0-7"),
                "core.mask_prompt_conditioner": (
                    1_100,
                    "core.sens_net+core.cascades.0-11",
                ),
            },
        },
        "c10_last_cascade_prompt": {
            "key": "mask_conditioned_c10_last_cascade_prompt",
            "schema": "promptmr-c10-last-cascade-mask-prompt-conditioner-v1",
            "parameter_count": 54_090_794,
            "trainable_variants": {
                "core.mask_prompt_conditioner": (
                    335,
                    "core.sens_net+core.cascades.0-9",
                ),
            },
        },
    }
    if conditioning_kind is not None:
        contract = conditioning_contracts[conditioning_kind]
        condition = specialist.get(contract["key"])
        variants = contract.get("trainable_variants")
        if variants is None:
            variants = {
                contract["scope"]: (
                    contract["trainable_parameter_count"],
                    "core.cascades.0-7",
                )
            }
        variant = variants.get(specialist.get("trainable_parameter_scope"))
        c10_conditioned = conditioning_kind == "c10_last_cascade_prompt"
        expected_num_cascades = 10 if c10_conditioned else 12
        if (
            (acceleration not in {"acc4", "acc8"} if c10_conditioned else acceleration != "acc8")
            or str(generalist["rung"]) != "R2"
            or int(generalist["num_cascades"]) != expected_num_cascades
            or int(generalist["n_history"]) != 0
            or not isinstance(condition, dict)
            or condition.get("schema") != contract["schema"]
            or condition.get("identity_initialized") is not True
            or condition.get("unknown_mask_route")
            != "exact_generalist_identity"
            or specialist.get("unknown_mask_route")
            != "exact_generalist_identity"
            or (
                c10_conditioned
                and specialist.get("parameter_efficient_specialization")
                is not True
            )
            or int(specialist.get("parameter_count", -1))
            != contract["parameter_count"]
            or variant is None
            or int(specialist.get("trainable_parameter_count", -1))
            != int(variant[0])
            or specialist.get("frozen_parameter_scope") != variant[1]
        ):
            raise RuntimeError(
                "selected mask-conditioned specialist contract is invalid"
            )
    loss_family = specialist.get("loss_family", "exact_upstream_ssim")
    if loss_family not in {
        "exact_upstream_ssim",
        "r10_image_masked_ssim_valid_windows_mean",
    }:
        raise RuntimeError("selected specialist loss family is unsupported")
    if (
        loss_family == "r10_image_masked_ssim_valid_windows_mean"
        and not (
            acceleration == "acc8"
            and str(generalist["rung"]) == "R2"
            and int(generalist["num_cascades"]) in {10, 12}
            and int(generalist["n_history"]) == 0
        )
    ):
        raise RuntimeError(
            "R10 image-masked SSIM requires the R2/C10-or-C12/H0 ACC8 specialist"
        )
    scope_contracts = {
        None: ("all", None, None),
        "all_registered_trainable_parameters": ("all", None, None),
        "core.cascades.8-11": (
            "last4",
            20_890_272,
            "core.sens_net+core.cascades.0-7",
        ),
        "core.sens_net+core.cascades.8-11": (
            "sens_last4",
            22_755_189,
            "core.cascades.0-7",
        ),
        (
            "core.sens_net+core.cascades.8-11+core.mask_conditioner"
        ): (
            "sens_last4",
            22_755_473,
            "core.cascades.0-7",
        ),
        (
            "core.sens_net+core.cascades.8-11+"
            "core.mask_prompt_conditioner"
        ): (
            "sens_last4",
            22_756_289,
            "core.cascades.0-7",
        ),
        "core.mask_prompt_conditioner": (
            "mask_prompt_only",
            1_100,
            "core.sens_net+core.cascades.0-11",
        ),
    }
    trainable_scope = specialist.get("trainable_parameter_scope")
    scope_contract = scope_contracts.get(trainable_scope)
    if (
        conditioning_kind == "c10_last_cascade_prompt"
        and trainable_scope == "core.mask_prompt_conditioner"
    ):
        scope_contract = (
            "c10_last_cascade_adapter",
            335,
            "core.sens_net+core.cascades.0-9",
        )
    if scope_contract is None:
        raise RuntimeError("selected specialist trainable scope is unsupported")
    scope_id, scope_count, frozen_scope = scope_contract
    late_loss_valid = (
        loss_family == "exact_upstream_ssim"
        or (
            quality_recovery_acc8
            and loss_family
            == "r10_image_masked_ssim_valid_windows_mean"
        )
    )
    if (late_acc4_moe or late_acc8_moe) and (
        scope_id != "all" or not late_loss_valid
    ):
        raise RuntimeError(
            "late MoE full-model loss contract is invalid"
        )
    c10_scope_valid = (
        c10_adapter_late
        and conditioning_kind == "c10_last_cascade_prompt"
        and str(generalist["rung"]) == "R2"
        and int(generalist["num_cascades"]) == 10
        and int(generalist["n_history"]) == 0
        and loss_family == "exact_upstream_ssim"
        and scope_id == "c10_last_cascade_adapter"
        and specialist.get("parameter_efficient_specialization") is True
        and int(specialist.get("trainable_parameter_count", -1))
        == scope_count
        and specialist.get("frozen_parameter_scope") == frozen_scope
    )
    legacy_scope_valid = (
        acceleration == "acc8"
        and str(generalist["rung"]) == "R2"
        and int(generalist["num_cascades"]) == 12
        and int(generalist["n_history"]) == 0
        and loss_family == "r10_image_masked_ssim_valid_windows_mean"
        and specialist.get("parameter_efficient_specialization") is True
        and int(specialist.get("trainable_parameter_count", -1))
        == scope_count
        and specialist.get("frozen_parameter_scope") == frozen_scope
    )
    if scope_id != "all" and not (c10_scope_valid or legacy_scope_valid):
        raise RuntimeError(
            "parameter-efficient specialist scope contract is invalid"
        )
    if conditioning_kind is not None:
        contract = conditioning_contracts[conditioning_kind]
        variants = contract.get("trainable_variants")
        expected_count = (
            variants[trainable_scope][0]
            if variants is not None
            else contract["trainable_parameter_count"]
        )
        if scope_count != expected_count:
            raise RuntimeError("mask-conditioning scope/count binding mismatch")
    for key in (
        "mraugment_seed",
        "legal_mask_seed",
        "lr_horizon_optimizer_steps",
    ):
        value = specialist.get(key)
        if value is not None and int(value) < 0:
            raise RuntimeError(f"selected specialist {key} is invalid")
    lr_horizon = specialist.get("lr_horizon_optimizer_steps")
    if (
        lr_horizon is not None
        and int(lr_horizon) < int(specialist["optimizer_steps"])
    ):
        raise RuntimeError(
            "selected specialist LR horizon cannot precede its stop step"
        )
    expected_late_horizon = (
        2_315
        if r19_acc8_moe
        else 35_040
        if r25_acc4_prefix
        else optimizer_steps
        if asymmetric_e51
        else 35_040 if acceleration == "acc4" else 34_725
    )
    if late_moe and int(lr_horizon or -1) != expected_late_horizon:
        raise RuntimeError(
            "late MoE prefix requires its presealed LR horizon"
        )
    soup = specialist.get("checkpoint_soup")
    if soup is not None:
        if not isinstance(soup, dict) or soup.get("source") != (
            "VESSL_TRAINED_SPECIALIST_CHECKPOINTS_ONLY"
        ):
            raise RuntimeError("selected specialist soup contract is invalid")
        if soup.get("enabled") is True:
            soup_key = (
                int(soup.get("earlier_optimizer_step", -1)),
                int(soup.get("later_optimizer_step", -1)),
                float(soup.get("later_weight", -1)),
            )
            if (
                soup_key
                not in {
                    (500, 1000, 0.5),
                    (1000, 1158, 0.5),
                    (1000, 1158, 0.75),
                }
                or soup_key[1] != optimizer_steps
            ):
                raise RuntimeError("selected specialist soup is unregistered")
        elif not (
            soup.get("enabled") is False
            and int(soup.get("earlier_optimizer_step", -1))
            == int(soup.get("later_optimizer_step", -1))
            == optimizer_steps
            and float(soup.get("later_weight", -1)) == 1.0
        ):
            raise RuntimeError("selected pure specialist identity is invalid")

    if conditioning_kind is not None:
        for composition_key in (
            "composition",
            "post_specialist_composition",
        ):
            composition = specialist.get(composition_key)
            if composition is not None and (
                not isinstance(composition, dict)
                or float(composition.get("reconstruction_alpha", -1)) != 1.0
                or float(composition.get("sensitivity_alpha", -1)) != 1.0
            ):
                raise RuntimeError(
                    "conditioned specialists cannot be folded into an "
                    "unconditioned parent"
                )

    installed = install_specialization_sources()
    if conditioning_kind is not None:
        installed.update(install_mask_conditioning_sources())
        installed[str(STAGED_MASK_SPECIALIST_TRAIN)] = (
            EXPECTED_STAGED_SHA256[STAGED_MASK_SPECIALIST_TRAIN]
        )
    conditioning_suffix = {
        None: "",
        "dc": "_MASKDC",
        "prompt": "_MASKPROMPT",
        "c10_last_cascade_prompt": "_C10MASKPROMPT",
    }[conditioning_kind]
    specialist_identity = object_sha256(
        {
            "schema": "vessl-r19-specialist-run-identity-v1",
            "parent_sha256": parent[1],
            "recipe": specialist,
            "train_source_sha256": EXPECTED_STAGED_SHA256[
                STAGED_SPECIALIST_TRAIN
            ],
            "production_source_sha256": EXPECTED_STAGED_SHA256[
                STAGED_SPECIALIST_PRODUCTION
            ],
        }
    )
    run = RESULT / (
        f"EXP_PROMPTMR_{winner}_{acceleration.upper()}_G10_FINAL_"
        f"E{expected_epochs}_S{optimizer_steps}{conditioning_suffix}_"
        f"SEED430_R19_{specialist_identity[:16]}"
    )
    recoveries = 0
    while True:
        terminal = load_json(run / "terminal.json")
        if terminal.get("status") == "COMPLETED":
            latest = latest_checkpoint(run)
            if latest is None:
                raise RuntimeError("completed specialist has no checkpoint")
            soupped = compose_specialist_soup(run, latest, specialist)
            selected = compose_specialist_checkpoint(
                parent,
                soupped,
                specialist,
                run,
            )
            return selected[0], selected[1], run, installed
        existing = trainer_pids(run.name)
        if len(existing) > 1:
            raise RuntimeError(f"duplicate specialist trainers: {existing}")
        if existing:
            update(
                f"ADOPTING_{acceleration.upper()}_SPECIALIST_TRAINER",
                winner=winner,
                trainer_pid=existing[0],
                recoveries=recoveries,
            )
            while alive(existing[0]):
                time.sleep(POLL_SECONDS)
            continue
        resume = latest_checkpoint(run) if run.exists() else None
        if (
            run.exists()
            and resume is None
            and not is_safe_prestep_specialist_run(run)
        ):
            raise RuntimeError(
                "specialist run exists without recoverable checkpoint"
            )
        command = specialist_command(
            generalist, specialist, run, parent, resume
        )
        update(
            f"TRAINING_{acceleration.upper()}_SPECIALIST",
            winner=winner,
            specialist_run=str(run),
            recoveries=recoveries,
            resume_checkpoint=str(resume[0]) if resume else None,
        )
        LOGS.mkdir(parents=True, exist_ok=True)
        with (LOGS / f"{run.name}.log").open("ab") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            returncode = process.wait()
        terminal = load_json(run / "terminal.json")
        if terminal.get("status") == "COMPLETED":
            continue
        recoveries += 1
        if recoveries > MAX_RECOVERIES or latest_checkpoint(run) is None:
            raise RuntimeError(
                f"{acceleration.upper()} specialist exhausted recovery: "
                f"returncode={returncode}, terminal={terminal}"
            )
        time.sleep(30)


def train_post_refiner(
    winner: str,
    recipe: dict[str, Any],
    refiner_recipe: dict[str, Any],
    parent: tuple[Path, str],
    acc4_parent: tuple[Path, str],
    acc8_parent: tuple[Path, str],
) -> tuple[Path, str, Path, dict[str, str]]:
    if refiner_recipe.get("enabled") is not True:
        raise RuntimeError("disabled post-refiner was sent to trainer")
    if time.time() >= FINAL_RESEARCH_CUTOFF_UNIX:
        raise FinalResearchCutoff(
            "post-refiner training would begin after the research cutoff"
        )
    restored_generalist_sources = install_sources(
        {
            STAGED_TRAIN: TRAIN_SOURCE,
            STAGED_PRODUCTION: PRODUCTION_SOURCE,
        }
    )
    expected_generalist = {
        TRAIN_SOURCE: EXPECTED_STAGED_SHA256[STAGED_TRAIN],
        PRODUCTION_SOURCE: EXPECTED_STAGED_SHA256[STAGED_PRODUCTION],
    }
    expected_specialist = {
        TRAIN_SOURCE: EXPECTED_STAGED_SHA256[STAGED_SPECIALIST_TRAIN],
        PRODUCTION_SOURCE: EXPECTED_STAGED_SHA256[
            STAGED_SPECIALIST_PRODUCTION
        ],
    }
    current_training_sources = {
        TRAIN_SOURCE: sha256(TRAIN_SOURCE),
        PRODUCTION_SOURCE: sha256(PRODUCTION_SOURCE),
    }
    if current_training_sources == expected_specialist:
        installed = install_sources(
            {
                STAGED_TRAIN: TRAIN_SOURCE,
                STAGED_PRODUCTION: PRODUCTION_SOURCE,
            }
        )
    elif current_training_sources == expected_generalist:
        installed = {
            str(path): digest
            for path, digest in expected_generalist.items()
        }
    else:
        raise RuntimeError(
            "post-refiner found unregistered training sources"
        )
    installed.update(
        install_sources(
            {STAGED_POST_REFINER: POST_REFINER_DESTINATION}
        )
    )
    installed.update(restored_generalist_sources)
    variant = str(refiner_recipe["variant"])
    refiner_epochs = int(refiner_recipe["epochs"])
    bbox_aligned_full_data = (
        refiner_recipe.get("training_data")
        == "organizer_train_plus_val_final"
        and refiner_recipe.get("loss_family")
        == R19_BBOX_LOSS_FAMILY
    )
    mask_conditioned_tta4 = (
        refiner_recipe.get("mask_conditioned") is True
        and refiner_recipe.get("views")
        == ["identity", "flip_lr", "flip_ud", "rot180"]
    )
    refiner_identity = object_sha256(
        {
            "schema": "vessl-r29-zf-context-post-refiner-run-identity-v1",
            "generalist_sha256": parent[1],
            "acc4_sha256": acc4_parent[1],
            "acc8_sha256": acc8_parent[1],
            "recipe": refiner_recipe,
            "trainer_source_sha256": EXPECTED_STAGED_SHA256[
                STAGED_POST_REFINER_TRAIN
            ],
            "module_source_sha256": EXPECTED_STAGED_SHA256[
                STAGED_POST_REFINER
            ],
        }
    )
    run = RESULT / (
        f"VESSL_POST_REFINER_{winner}_{variant}_"
        f"E{refiner_epochs}_"
        f"{'BBOX05_FULLDATA_' if bbox_aligned_full_data else ''}"
        f"SEED430_R29_{refiner_identity[:16]}"
    )
    recoveries = 0
    while True:
        receipt = load_json(run / "receipt.json")
        checkpoint = Path(str(receipt.get("checkpoint", "")))
        checkpoint_sha256 = str(receipt.get("checkpoint_sha256", ""))
        if (
            receipt.get("schema")
            == "vessl-post-refiner-training-receipt-v1"
            and receipt.get("state") == "PASS"
            and receipt.get("base_checkpoint_sha256") == parent[1]
            and receipt.get("variant") == variant
            and receipt.get("views") == refiner_recipe["views"]
            and bool(receipt.get("mask_conditioned", False))
            is bool(refiner_recipe.get("mask_conditioned", False))
            and receipt.get("input_mode")
            == refiner_recipe.get("input_mode")
            and receipt.get("zero_filled_definition")
            == refiner_recipe.get("zero_filled_definition")
            and receipt.get("normalization")
            == refiner_recipe.get("normalization")
            and receipt.get("spatial_match")
            == refiner_recipe.get("spatial_match")
            and int(receipt.get("epochs", -1)) == refiner_epochs
            and receipt.get("loss_family")
            == refiner_recipe.get("loss_family", "exact_upstream_ssim")
            and receipt.get("bbox_loss_coefficient")
            == refiner_recipe.get("bbox_loss_coefficient")
            and receipt.get("organizer_annotations_used_for_training")
            is refiner_recipe.get("organizer_annotations_used_for_training")
            and receipt.get("inference_annotation_access") is False
            and receipt.get("training_data")
            == refiner_recipe.get("training_data", "organizer_train_only")
            and receipt.get("validation_used_for_checkpoint_selection")
            is False
            and receipt.get("training_base_route")
            == "exact_acceleration_specialist_before_shared_naf_s_v1"
            and receipt.get("routed_branch_sha256") == {
                "acc4": acc4_parent[1],
                "acc8": acc8_parent[1],
            }
            and receipt.get("sampler_policy")
            == "equal_acc_real_acc8_real80_virtual20_v1"
            and int(receipt.get("lr_horizon_optimizer_steps", -1))
            == int(refiner_recipe.get("lr_horizon_optimizer_steps", -1))
            and valid_post_training_data_contract(
                receipt.get("training_data_contract")
            )
            and (
                refiner_recipe.get("loss_family") != R19_BBOX_LOSS_FAMILY
                or (
                    isinstance(receipt.get("annotation_contract"), dict)
                    and receipt["annotation_contract"].get("schema")
                    == "organizer-train-val-official384-bbox-cell-weighting-v2"
                    and receipt["annotation_contract"].get(
                        "source_coordinate_frame"
                    )
                    == [384, 384]
                    and receipt["annotation_contract"].get(
                        "training_tensor_alignment"
                    )
                    == "test_part_center_crop_then_zero_pad_v1"
                )
            )
            and (
                refiner_recipe.get("optimizer_steps") is None
                or int(receipt.get("optimizer_steps", -1))
                == int(refiner_recipe["optimizer_steps"])
            )
            and checkpoint.is_file()
            and sha256(checkpoint) == checkpoint_sha256
        ):
            for pid in trainer_pids(run.name):
                stop_trainer(pid)
            return checkpoint, checkpoint_sha256, run, installed
        if time.time() >= FINAL_RESEARCH_CUTOFF_UNIX:
            for pid in trainer_pids(run.name):
                stop_trainer(pid)
            raise FinalResearchCutoff(
                "post-refiner reached the final research cutoff"
            )
        existing = trainer_pids(run.name)
        if len(existing) > 1:
            raise RuntimeError(f"duplicate post-refiner trainers: {existing}")
        if existing:
            update(
                "ADOPTING_POST_REFINER_TRAINER",
                winner=winner,
                variant=variant,
                trainer_pid=existing[0],
                recoveries=recoveries,
            )
            while alive(existing[0]):
                if time.time() >= FINAL_RESEARCH_CUTOFF_UNIX:
                    stop_trainer(existing[0])
                    raise FinalResearchCutoff(
                        "adopted post-refiner reached final cutoff"
                    )
                time.sleep(TRAINING_POLL_SECONDS)
            continue
        recovery = run / "recovery.pt"
        if run.exists() and any(run.iterdir()) and not recovery.is_file():
            raise RuntimeError(
                "partial post-refiner run has no recoverable checkpoint"
            )
        command = [
            "/bin/python",
            "-u",
            str(STAGED_POST_REFINER_TRAIN),
            "--base-checkpoint",
            str(parent[0]),
            "--base-checkpoint-sha256",
            parent[1],
            "--acc4-checkpoint",
            str(acc4_parent[0]),
            "--acc4-checkpoint-sha256",
            acc4_parent[1],
            "--acc8-checkpoint",
            str(acc8_parent[0]),
            "--acc8-checkpoint-sha256",
            acc8_parent[1],
            "--variant",
            variant,
            "--input-mode",
            str(refiner_recipe["input_mode"]),
            "--views",
            *map(str, refiner_recipe["views"]),
            "--epochs",
            str(refiner_epochs),
            "--output-dir",
            str(run),
            "--train-root",
            "/root/Data/train",
            "--trusted-data-manifest",
            (
                "/root/result/EXP_FI_ACC8_CKPT_BASE_E30_R1/"
                "fi-acc8-full-training/provenance.json"
            ),
            "--loss-family",
            str(
                refiner_recipe.get(
                    "loss_family", "exact_upstream_ssim"
                )
            ),
            "--peak-lr",
            repr(float(refiner_recipe["peak_lr"])),
            "--weight-decay",
            repr(float(refiner_recipe["weight_decay"])),
            "--seed",
            str(int(refiner_recipe["seed"])),
        ]
        if refiner_recipe.get("training_data") == "organizer_train_plus_val_final":
            command.extend(
                [
                    "--extra-train-root",
                    str(refiner_recipe["extra_train_root"]),
                    "--extra-trusted-data-manifest",
                    (
                        "/root/result/EXP_FI_ACC8_CKPT_BASE_E30_R1/"
                        "fi-acc8-full-training/provenance.json"
                    ),
                ]
            )
        if recovery.is_file():
            command.extend(["--resume", str(recovery)])
        if refiner_recipe.get("optimizer_steps") is not None:
            command.extend(
                [
                    "--optimizer-steps",
                    str(int(refiner_recipe["optimizer_steps"])),
                ]
            )
        if refiner_recipe.get("lr_horizon_optimizer_steps") is not None:
            command.extend(
                [
                    "--lr-horizon-optimizer-steps",
                    str(int(refiner_recipe["lr_horizon_optimizer_steps"])),
                ]
            )
        if refiner_recipe.get("mask_conditioned") is True:
            command.append("--mask-conditioned")
        update(
            "TRAINING_VESSL_POST_REFINER",
            winner=winner,
            variant=variant,
            run=str(run),
            recoveries=recoveries,
            resume=str(recovery) if recovery.is_file() else None,
        )
        LOGS.mkdir(parents=True, exist_ok=True)
        with (LOGS / f"{run.name}.log").open("ab") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            while process.poll() is None:
                if time.time() >= FINAL_RESEARCH_CUTOFF_UNIX:
                    stop_trainer(process.pid)
                    raise FinalResearchCutoff(
                        "post-refiner trainer reached final cutoff"
                    )
                time.sleep(TRAINING_POLL_SECONDS)
            returncode = process.wait()
        recoveries += 1
        if recoveries > MAX_RECOVERIES or not recovery.is_file():
            raise RuntimeError(
                "post-refiner exhausted recovery: "
                f"returncode={returncode}"
            )
        time.sleep(30)


def build_shared_final(
    checkpoint: Path,
    checkpoint_sha256: str,
    *,
    acc4_checkpoint: Path | None = None,
    acc4_sha256: str | None = None,
    acc8_checkpoint: Path | None = None,
    acc8_sha256: str | None = None,
    post_refiner_checkpoint: Path | None = None,
    post_refiner_sha256: str | None = None,
    tta_views: str,
    suffix: str,
) -> tuple[Path, dict[str, Any], Path]:
    build_identity = {
        "schema": "vessl-r20-single-final-build-identity-v1",
        "generalist_sha256": checkpoint_sha256,
        "acc4_sha256": acc4_sha256 or checkpoint_sha256,
        "acc8_sha256": acc8_sha256 or checkpoint_sha256,
        "post_refiner_sha256": post_refiner_sha256,
        "tta_views": tta_views,
        "builder_source_sha256": EXPECTED_STAGED_SHA256[STAGED_BUILDER],
        "router_source_sha256": EXPECTED_STAGED_SHA256[STAGED_ROUTER],
        "test_part_source_sha256": EXPECTED_STAGED_SHA256[STAGED_TEST_PART],
        "post_refiner_source_sha256": EXPECTED_STAGED_SHA256[
            STAGED_POST_REFINER
        ],
        "final_admission_source_sha256": EXPECTED_STAGED_SHA256[
            STAGED_FINAL_ADMISSION
        ],
        "inference_amendment_sha256": EXPECTED_STAGED_SHA256[
            STAGED_RUNTIME_AMENDMENT
        ],
    }
    build_identity_sha256 = object_sha256(build_identity)
    final = RESULT / (
        f"VESSL_FINAL_G10_G11_ROUTED_{suffix}_{build_identity_sha256[:16]}"
    )
    final_checkpoint = final / "best_model.pt"
    build_receipt_path = final / "r20-build-receipt.json"
    final.mkdir(parents=True, exist_ok=True)
    install_mask_conditioning_sources()
    install_sources(
        {
            STAGED_ROUTER: ROUTER_DESTINATION,
            STAGED_TEST_PART: TEST_PART_DESTINATION,
            STAGED_LEGAL_MASK: LEGAL_MASK_SOURCE,
            STAGED_MASK_ROUTER: MASK_ROUTER_SOURCE,
            STAGED_POST_REFINER: POST_REFINER_DESTINATION,
        }
    )
    if final_checkpoint.exists():
        build_receipt = load_json(build_receipt_path)
        if (
            build_receipt.get("schema")
            != "vessl-r20-single-final-build-receipt-v1"
            or build_receipt.get("state") != "PASS"
            or build_receipt.get("build_identity") != build_identity
            or build_receipt.get("build_identity_sha256")
            != build_identity_sha256
            or build_receipt.get("final_checkpoint_sha256")
            != sha256(final_checkpoint)
        ):
            raise RuntimeError(
                "content-addressed final package exists without its exact build receipt"
            )
    else:
        if build_receipt_path.exists():
            raise RuntimeError("orphaned final build receipt blocks package reuse")
        command = [
            "/bin/python",
            "-u",
            str(STAGED_BUILDER),
            "--acc4-checkpoint",
            str(acc4_checkpoint or checkpoint),
            "--acc4-sha256",
            acc4_sha256 or checkpoint_sha256,
            "--acc8-checkpoint",
            str(acc8_checkpoint or checkpoint),
            "--acc8-sha256",
            acc8_sha256 or checkpoint_sha256,
            "--tta-views",
            tta_views,
            "--output",
            str(final_checkpoint),
        ]
        if (
            (acc4_sha256 or checkpoint_sha256) != checkpoint_sha256
            or (acc8_sha256 or checkpoint_sha256) != checkpoint_sha256
        ):
            command.extend([
                "--generalist-checkpoint", str(checkpoint),
                "--generalist-sha256", checkpoint_sha256,
            ])
        if post_refiner_checkpoint is not None:
            if post_refiner_sha256 is None:
                raise RuntimeError("post-refiner SHA-256 is absent")
            command.extend(
                [
                    "--post-refiner-checkpoint",
                    str(post_refiner_checkpoint),
                    "--post-refiner-sha256",
                    post_refiner_sha256,
                ]
            )
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )
        if not final_checkpoint.is_file():
            raise RuntimeError("final builder did not create best_model.pt")
        atomic_json(
            build_receipt_path,
            {
                "schema": "vessl-r20-single-final-build-receipt-v1",
                "state": "PASS",
                "build_identity": build_identity,
                "build_identity_sha256": build_identity_sha256,
                "final_checkpoint": str(final_checkpoint),
                "final_checkpoint_sha256": sha256(final_checkpoint),
                "candidate_count": 1,
                "created_unix": time.time(),
            },
        )
    final_checkpoint_sha256 = sha256(final_checkpoint)
    admission = final / f"admission-v2-{final_checkpoint_sha256[:16]}.json"
    completed = subprocess.run(
        [
            "/bin/python",
            "-u",
            str(STAGED_FINAL_ADMISSION),
            "--exp-dir",
            str(final),
            "--output",
            str(admission),
        ],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode not in {0, 2}:
        raise RuntimeError(
            f"final admission crashed: {completed.returncode}"
        )
    value = load_json(admission)
    if (
        value.get("schema") != "vessl-final-lazy-router-admission-v2"
        or value.get("state") not in {"PASS", "FAIL"}
        or value.get("final_checkpoint_sha256")
        != final_checkpoint_sha256
    ):
        raise RuntimeError("selected architecture final admission is invalid")
    return final_checkpoint, value, final


def build_admitted_ladder(
    generalist_checkpoint: Path,
    generalist_sha256: str,
    fallback_checkpoint: Path | None,
    fallback_sha256: str | None,
    *,
    acc4_checkpoint: Path | None,
    acc4_sha256: str | None,
    acc8_checkpoint: Path | None,
    acc8_sha256: str | None,
    post_refiner_checkpoint: Path | None,
    post_refiner_sha256: str | None,
    tta_views: str,
    suffix: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], str, str]:
    primary_acc4 = acc4_checkpoint or generalist_checkpoint
    primary_acc4_sha = acc4_sha256 or generalist_sha256
    primary_acc8 = acc8_checkpoint or generalist_checkpoint
    primary_acc8_sha = acc8_sha256 or generalist_sha256
    proposals = [
        {
            "mode": "PRIMARY_REQUESTED",
            "acc4": primary_acc4,
            "acc4_sha": primary_acc4_sha,
            "acc8": primary_acc8,
            "acc8_sha": primary_acc8_sha,
            "post_refiner": post_refiner_checkpoint,
            "post_refiner_sha": post_refiner_sha256,
            "tta": tta_views,
            "suffix": suffix,
        }
    ]
    if not SINGLE_FINAL_REQUIRED and post_refiner_checkpoint is not None:
        proposals.append(
            {
                **proposals[0],
                "mode": "SELECTED_GENERALIST_NO_POST_REFINER_IDENTITY",
                "post_refiner": None,
                "post_refiner_sha": None,
                "tta": "identity",
                "suffix": f"{suffix}_NO_POST_REFINER_IDENTITY_R1",
            }
        )
    if not SINGLE_FINAL_REQUIRED and tta_views != "identity":
        proposals.append(
            {
                **proposals[0],
                "mode": "ROUTED_IDENTITY",
                "tta": "identity",
                "suffix": f"{suffix}_IDENTITY_R1",
            }
        )
    if not SINGLE_FINAL_REQUIRED and (
        primary_acc4_sha != generalist_sha256
        or primary_acc8_sha != generalist_sha256
    ):
        proposals.append(
            {
                "mode": "SELECTED_GENERALIST_IDENTITY",
                "acc4": generalist_checkpoint,
                "acc4_sha": generalist_sha256,
                "acc8": generalist_checkpoint,
                "acc8_sha": generalist_sha256,
                "post_refiner": None,
                "post_refiner_sha": None,
                "tta": "identity",
                "suffix": f"{suffix}_SELECTED_GENERALIST_IDENTITY_R1",
            }
        )
    if not SINGLE_FINAL_REQUIRED and generalist_sha256 != fallback_sha256:
        proposals.append(
            {
                "mode": "SEALED_R1_GENERALIST_IDENTITY",
                "acc4": fallback_checkpoint,
                "acc4_sha": fallback_sha256,
                "acc8": fallback_checkpoint,
                "acc8_sha": fallback_sha256,
                "post_refiner": None,
                "post_refiner_sha": None,
                "tta": "identity",
                "suffix": f"{suffix}_SEALED_R1_IDENTITY_R1",
            }
        )

    attempts = []
    observed: set[tuple[str, str, str, str]] = set()
    for proposal in proposals:
        identity = (
            str(proposal["acc4_sha"]),
            str(proposal["acc8_sha"]),
            str(proposal["tta"]),
            str(proposal["post_refiner_sha"]),
        )
        if identity in observed:
            continue
        observed.add(identity)
        final_checkpoint, admission, final = build_shared_final(
            generalist_checkpoint,
            generalist_sha256,
            acc4_checkpoint=Path(proposal["acc4"]),
            acc4_sha256=str(proposal["acc4_sha"]),
            acc8_checkpoint=Path(proposal["acc8"]),
            acc8_sha256=str(proposal["acc8_sha"]),
            post_refiner_checkpoint=(
                Path(proposal["post_refiner"])
                if proposal["post_refiner"] is not None
                else None
            ),
            post_refiner_sha256=(
                str(proposal["post_refiner_sha"])
                if proposal["post_refiner_sha"] is not None
                else None
            ),
            tta_views=str(proposal["tta"]),
            suffix=str(proposal["suffix"]),
        )
        attempt = {
            "mode": proposal["mode"],
            "final_directory": str(final),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": sha256(final_checkpoint),
            "acc4_checkpoint_sha256": proposal["acc4_sha"],
            "acc8_checkpoint_sha256": proposal["acc8_sha"],
            "tta_views": proposal["tta"],
            "post_refiner_checkpoint_sha256": proposal[
                "post_refiner_sha"
            ],
            "admission": admission,
        }
        attempts.append(attempt)
        if admission.get("state") == "PASS":
            return (
                final_checkpoint,
                admission,
                attempts,
                str(proposal["tta"]),
                str(proposal["mode"]),
            )
    atomic_json(
        CONTROL / f"{suffix}-final-admission-ladder-failure.json",
        {
            "schema": "vessl-final-admission-ladder-failure-v1",
            "state": "FAIL",
            "attempts": attempts,
            "leaderboard_data_read": False,
            "external_learned_state_imported": False,
            "created_unix": time.time(),
        },
    )
    raise RuntimeError("all final admission ladder candidates failed")


def main() -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # A healthy authoritative controller owns the immutable workflow.
        # A watchdog probe must not overwrite its status with a false failure.
        return 0
    existing = load_json(RECEIPT)
    if existing.get("state") == "PASS":
        update("PASS", receipt=str(RECEIPT))
        return 0
    validate_staged()
    winner, recipe, tactics = validate_handoffs()
    active_tactics_handoff = TACTICS_HANDOFF
    # Claim the only post-E35 CUDA lane as soon as the score-free G10/G11
    # handoffs are valid.  The protected generalist does not use this lock;
    # only the two final dispatchers are serialized.
    FINAL_GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    final_gpu_lock_handle = FINAL_GPU_LOCK.open("a+")
    update(
        "WAITING_FINAL_GPU_OWNERSHIP",
        final_gpu_lock=str(FINAL_GPU_LOCK),
        winner=winner,
    )
    fcntl.flock(final_gpu_lock_handle, fcntl.LOCK_EX)
    claim = {
        "schema": "vessl-final-gpu-authority-claim-v1",
        "state": "CLAIMED",
        "owner": "G10_ARCHITECTURE_DISPATCHER",
        "winner": winner,
        "architecture_handoff_sha256": sha256(ARCH_HANDOFF),
        "tactics_handoff_sha256": sha256(TACTICS_HANDOFF),
        "leaderboard_data_read": False,
        "external_learned_state_imported": False,
        "created_unix": time.time(),
    }
    existing_claim = load_json(AUTHORITY_CLAIM)
    if existing_claim:
        comparable = dict(existing_claim)
        comparable["created_unix"] = claim["created_unix"]
        if comparable != claim:
            raise RuntimeError("existing G10 authority claim differs")
    else:
        atomic_json(AUTHORITY_CLAIM, claim)
        AUTHORITY_CLAIM.chmod(0o444)
    update(
        "FINAL_GPU_OWNERSHIP_ACQUIRED",
        final_gpu_lock=str(FINAL_GPU_LOCK),
        authority_claim=str(AUTHORITY_CLAIM),
        winner=winner,
    )
    # The amended NAF path has exactly one final package and no registered
    # fallback.  In that mode, waiting for the historical R1 fallback would
    # violate the contract and could block an otherwise valid E50->E51
    # transition.  The legacy value is retained only for non-amended paths.
    if SINGLE_FINAL_REQUIRED:
        fallback, fallback_sha256 = None, None
        update(
            "SINGLE_FINAL_NO_FALLBACK_REGISTERED",
            fallback_registered=False,
            candidate_count_max=1,
        )
    else:
        fallback, fallback_sha256 = wait_fallback()
    stopped = stop_legacy_dispatcher()
    admission_path = CONTROL / f"{winner}-admission.json"
    if not admission_path.exists():
        if ARCHITECTURE_ADMISSION_SOURCE is not None:
            source_admission = load_json(ARCHITECTURE_ADMISSION_SOURCE)
            if (
                source_admission.get("state") != "PASS"
                or source_admission.get("winner") != winner
                or source_admission.get("external_learned_state_imported")
                is not False
                or source_admission.get("leaderboard_data_read") is not False
            ):
                raise RuntimeError(
                    "configured architecture admission source is not a sealed PASS"
                )
            atomic_json(admission_path, source_admission)
        else:
            if sha256(PRODUCTION_SOURCE) != ORIGINAL_PRODUCTION_SHA256:
                raise RuntimeError(
                    "pre-admission VESSL production source mismatch"
                )
            completed = subprocess.run(
                [
                    "/bin/python",
                    "-u",
                    str(STAGED_ADMISSION),
                    "--handoff",
                    str(ARCH_HANDOFF),
                    "--output",
                    str(admission_path),
                    "--lock",
                    str(CONTROL / "admission.lock"),
                ],
                cwd=ROOT,
                check=False,
            )
            if completed.returncode not in {0, 2}:
                raise RuntimeError(
                    f"architecture admission crashed: {completed.returncode}"
                )
    admission = load_json(admission_path)
    tta_views = str(tactics["inference"]["tta_views"])
    selected_run = None
    acc4_specialist_run = None
    acc8_specialist_run = None
    acc4_checkpoint = None
    acc4_sha256 = None
    acc8_checkpoint = None
    acc8_sha256 = None
    specialization_sources: dict[str, str] = {}
    post_refiner_run = None
    post_refiner_checkpoint = None
    post_refiner_sha256 = None
    post_refiner_sources: dict[str, str] = {}
    post_refiner_error = None
    final_admission_attempts: list[dict[str, Any]] = []
    final_mode = "PENDING"
    effective_tta_views = tta_views
    if admission.get("state") != "PASS":
        if SINGLE_FINAL_REQUIRED:
            raise RuntimeError(
                "main C10 admission failed and single-final policy forbids fallback"
            )
        (
            final_checkpoint,
            final_admission,
            final_admission_attempts,
            effective_tta_views,
            final_mode,
        ) = build_admitted_ladder(
            fallback,
            fallback_sha256,
            fallback,
            fallback_sha256,
            acc4_checkpoint=None,
            acc4_sha256=None,
            acc8_checkpoint=None,
            acc8_sha256=None,
            post_refiner_checkpoint=None,
            post_refiner_sha256=None,
            tta_views=tta_views,
            suffix="R1_ADMISSION_FALLBACK_R1",
        )
        decision = "SELECTED_ARCHITECTURE_ADMISSION_FAILED_R1_FALLBACK"
        selected_run = None
        selected_checkpoint = fallback
        selected_sha256 = fallback_sha256
    else:
        installed = install_sources(
            {
                STAGED_TRAIN: TRAIN_SOURCE,
                STAGED_PRODUCTION: PRODUCTION_SOURCE,
                STAGED_LEGAL_MASK: LEGAL_MASK_SOURCE,
            }
        )
        atomic_json(CONTROL / "installed-training-sources.json", installed)
        try:
            (
                selected_checkpoint,
                selected_sha256,
                selected_run,
            ) = train_generalist(winner, recipe)
        except FinalResearchCutoff:
            if SINGLE_FINAL_REQUIRED:
                raise RuntimeError(
                    "main C10 missed the cutoff and single-final policy forbids fallback"
                )
            (
                final_checkpoint,
                final_admission,
                final_admission_attempts,
                effective_tta_views,
                final_mode,
            ) = build_admitted_ladder(
                fallback,
                fallback_sha256,
                fallback,
                fallback_sha256,
                acc4_checkpoint=None,
                acc4_sha256=None,
                acc8_checkpoint=None,
                acc8_sha256=None,
                post_refiner_checkpoint=None,
                post_refiner_sha256=None,
                tta_views=tta_views,
                suffix="R1_DEADLINE_FALLBACK_R1",
            )
            decision = "SELECTED_ARCHITECTURE_CUTOFF_R1_FALLBACK"
            selected_run = None
            selected_checkpoint = fallback
            selected_sha256 = fallback_sha256
            acc4_specialist_run = None
            acc8_specialist_run = None
            acc4_checkpoint = None
            acc4_sha256 = None
            acc8_checkpoint = None
            acc8_sha256 = None
            specialization_sources = {}
        else:
            tactics, active_tactics_handoff = refresh_terminal_tactics(
                winner,
                recipe,
            )
            tta_views = str(tactics["inference"]["tta_views"])
            effective_tta_views = tta_views
            acc4_specialist = tactics.get("acc4_specialist", {})
            acc8_specialist = tactics.get("acc8_specialist", {})
            if not isinstance(acc4_specialist, dict):
                raise RuntimeError("selected ACC4 specialist recipe is absent")
            if not isinstance(acc8_specialist, dict):
                raise RuntimeError("selected ACC8 specialist recipe is absent")
            if (
                acc4_specialist.get("enabled") is True
                and acc8_specialist.get("enabled") is True
            ):
                dual_late_moe = (
                    acc4_specialist.get("schema")
                    == R19_ACC4_SCHEMA
                    and acc8_specialist.get("schema")
                    == R19_ACC8_SCHEMA
                    and int(acc4_specialist.get("late_branch_parent_epoch", -1))
                    == GENERALIST_HANDOFF_EPOCH
                    and int(acc8_specialist.get("late_branch_parent_epoch", -1))
                    == GENERALIST_HANDOFF_EPOCH
                    and acc4_specialist.get("deployment_scope")
                    == "acc4_only_router"
                    and acc8_specialist.get("deployment_scope")
                    == "acc8_only_router"
                )
                if not dual_late_moe:
                    raise RuntimeError(
                        "dual specialists require the registered shared-E49 "
                        "ACC4/ACC8 late-MoE contract"
                    )
            acc4_checkpoint = selected_checkpoint
            acc4_sha256 = selected_sha256
            acc8_checkpoint = selected_checkpoint
            acc8_sha256 = selected_sha256
            if acc4_specialist.get("enabled") is True:
                (
                    acc4_checkpoint,
                    acc4_sha256,
                    acc4_specialist_run,
                    acc4_specialization_sources,
                ) = train_specialist(
                    winner,
                    recipe,
                    acc4_specialist,
                    (selected_checkpoint, selected_sha256),
                )
                specialization_sources.update(acc4_specialization_sources)
            if acc8_specialist.get("enabled") is True:
                (
                    acc8_checkpoint,
                    acc8_sha256,
                    acc8_specialist_run,
                    acc8_specialization_sources,
                ) = train_specialist(
                    winner,
                    recipe,
                    acc8_specialist,
                    (selected_checkpoint, selected_sha256),
                )
                for source, digest in acc8_specialization_sources.items():
                    existing_digest = specialization_sources.get(source)
                    if existing_digest not in {None, digest}:
                        raise RuntimeError(
                            "dual late-MoE specialization source mismatch"
                        )
                    specialization_sources[source] = digest
                if (
                    acc8_specialist.get("deployment_scope")
                    == "global_successor"
                ):
                    if acc4_specialist.get("enabled") is True:
                        raise RuntimeError(
                            "global ACC8 successor conflicts with ACC4 specialist"
                        )
                    acc4_checkpoint = acc8_checkpoint
                    acc4_sha256 = acc8_sha256
            post_refiner_recipe = tactics.get("post_refiner", {})
            if not isinstance(post_refiner_recipe, dict):
                raise RuntimeError("selected post-refiner recipe is absent")
            if post_refiner_recipe.get("enabled") is True:
                try:
                    (
                        post_refiner_checkpoint,
                        post_refiner_sha256,
                        post_refiner_run,
                        post_refiner_sources,
                    ) = train_post_refiner(
                        winner,
                        recipe,
                        post_refiner_recipe,
                        (selected_checkpoint, selected_sha256),
                        (acc4_checkpoint, acc4_sha256),
                        (acc8_checkpoint, acc8_sha256),
                    )
                except Exception as error:
                    post_refiner_error = (
                        f"{type(error).__name__}: {error}"
                    )
                    if SINGLE_FINAL_REQUIRED:
                        update(
                            "POST_REFINER_FAILED_SINGLE_FINAL_BLOCKED",
                            winner=winner,
                            error=post_refiner_error,
                        )
                        raise RuntimeError(
                            "NAF_S post-refiner failed and single-final policy forbids base-only fallback"
                        ) from error
                    update(
                        "POST_REFINER_FALLBACK_TO_BASE",
                        winner=winner,
                        error=post_refiner_error,
                    )
                    post_refiner_checkpoint = None
                    post_refiner_sha256 = None
                    post_refiner_run = None
            (
                final_checkpoint,
                final_admission,
                final_admission_attempts,
                effective_tta_views,
                final_mode,
            ) = build_admitted_ladder(
                selected_checkpoint,
                selected_sha256,
                fallback,
                fallback_sha256,
                acc4_checkpoint=acc4_checkpoint,
                acc4_sha256=acc4_sha256,
                acc8_checkpoint=acc8_checkpoint,
                acc8_sha256=acc8_sha256,
                post_refiner_checkpoint=post_refiner_checkpoint,
                post_refiner_sha256=post_refiner_sha256,
                tta_views=tta_views,
                suffix=f"{winner}_DEADLINE_MAX_SCORE_R29",
            )
            decision = "SELECTED_ARCHITECTURE_VESSL_SCRATCH_COMPLETE"
    if final_mode != "PRIMARY_REQUESTED":
        decision = f"{decision}_{final_mode}"
    inference_source_snapshot = snapshot_inference_sources(
        final_checkpoint.parent
    )

    acc4_specialist = tactics.get("acc4_specialist", {})
    acc8_specialist = tactics.get("acc8_specialist", {})
    routed_specialist_packaged = final_mode in {
        "PRIMARY_REQUESTED",
        "ROUTED_IDENTITY",
    }
    effective_post_refiner_sha256 = (
        final_admission_attempts[-1].get(
            "post_refiner_checkpoint_sha256"
        )
        if final_admission_attempts
        else None
    )
    post_refiner_packaged = (
        post_refiner_sha256 is not None
        and effective_post_refiner_sha256 == post_refiner_sha256
    )
    payload = {
        "schema": "vessl-g10-architecture-dispatcher-receipt-v1",
        "state": "PASS",
        "decision": decision,
        "winner": winner,
        "active_tactics_handoff": str(active_tactics_handoff),
        "active_tactics_handoff_sha256": sha256(active_tactics_handoff),
        "fallback_checkpoint": (
            str(fallback) if fallback is not None else None
        ),
        "fallback_checkpoint_sha256": fallback_sha256,
        "selected_run": str(selected_run) if selected_run else None,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": selected_sha256,
        "acc4_specialist_run": (
            str(acc4_specialist_run)
            if admission.get("state") == "PASS"
            and acc4_specialist.get("enabled") is True
            else None
        ),
        "acc4_specialist_checkpoint": (
            str(acc4_checkpoint)
            if admission.get("state") == "PASS"
            and acc4_specialist.get("enabled") is True
            else None
        ),
        "acc4_specialist_checkpoint_sha256": (
            acc4_sha256
            if admission.get("state") == "PASS"
            and acc4_specialist.get("enabled") is True
            else None
        ),
        "acc8_specialist_run": (
            str(acc8_specialist_run)
            if admission.get("state") == "PASS"
            and acc8_specialist.get("enabled") is True
            else None
        ),
        "acc8_specialist_checkpoint": (
            str(acc8_checkpoint)
            if admission.get("state") == "PASS"
            and acc8_specialist.get("enabled") is True
            else None
        ),
        "acc8_specialist_checkpoint_sha256": (
            acc8_sha256
            if admission.get("state") == "PASS"
            and acc8_specialist.get("enabled") is True
            else None
        ),
        "specialization_sources": (
            specialization_sources
            if admission.get("state") == "PASS"
            else {}
        ),
        "post_refiner_requested": (
            tactics.get("post_refiner", {}).get("enabled") is True
        ),
        "post_refiner_run": (
            str(post_refiner_run) if post_refiner_run else None
        ),
        "post_refiner_checkpoint": (
            str(post_refiner_checkpoint)
            if post_refiner_checkpoint is not None
            else None
        ),
        "post_refiner_checkpoint_sha256": post_refiner_sha256,
        "post_refiner_sources": post_refiner_sources,
        "post_refiner_training_error": post_refiner_error,
        "post_refiner_status": (
            "TRAINED_ON_VESSL_AND_PACKAGED"
            if post_refiner_packaged
            else "TRAINED_NOT_PACKAGED_ADMISSION_FALLBACK"
            if post_refiner_checkpoint is not None
            else "FALLBACK_TO_BASE"
            if tactics.get("post_refiner", {}).get("enabled") is True
            else "NOT_SELECTED"
        ),
        "inference_source_snapshot": inference_source_snapshot,
        "inference_amendment": str(STAGED_RUNTIME_AMENDMENT),
        "inference_amendment_sha256": EXPECTED_STAGED_SHA256[
            STAGED_RUNTIME_AMENDMENT
        ],
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "final_admission": final_admission,
        "tta_views_requested": tta_views,
        "tta_views": effective_tta_views,
        "final_admission_mode": final_mode,
        "final_admission_attempts": final_admission_attempts,
        "acc4_specialist_requested": (
            acc4_specialist.get("enabled") is True
        ),
        "acc4_specialist_status": (
            "TRAINED_ON_VESSL_AND_ROUTED"
            if (
                acc4_specialist.get("enabled") is True
                and acc4_specialist_run is not None
                and routed_specialist_packaged
            )
            else "CUTOFF_FALLBACK_NOT_TRAINED"
            if decision.startswith(
                "SELECTED_ARCHITECTURE_CUTOFF_R1_FALLBACK"
            )
            else "TRAINED_NOT_PACKAGED_ADMISSION_FALLBACK"
            if (
                acc4_specialist.get("enabled") is True
                and acc4_specialist_run is not None
            )
            else "NOT_SELECTED"
        ),
        "acc8_specialist_requested": (
            acc8_specialist.get("enabled") is True
        ),
        "acc8_specialist_deployment_scope": (
            acc8_specialist.get("deployment_scope")
            if acc8_specialist.get("enabled") is True
            else None
        ),
        "acc8_specialist_status": (
            (
                "TRAINED_ON_VESSL_AND_GLOBAL_SUCCESSOR"
                if acc8_specialist.get("deployment_scope")
                == "global_successor"
                else "TRAINED_ON_VESSL_AND_ROUTED"
            )
            if (
                acc8_specialist.get("enabled") is True
                and acc8_specialist_run is not None
                and routed_specialist_packaged
            )
            else "CUTOFF_FALLBACK_NOT_TRAINED"
            if decision.startswith(
                "SELECTED_ARCHITECTURE_CUTOFF_R1_FALLBACK"
            )
            else "TRAINED_NOT_PACKAGED_ADMISSION_FALLBACK"
            if (
                acc8_specialist.get("enabled") is True
                and acc8_specialist_run is not None
            )
            else "NOT_SELECTED"
        ),
        "legacy_dispatcher_stop": stopped,
        "architecture_admission": admission,
        "leaderboard_data_read": False,
        "external_learned_state_imported": False,
        "all_final_learned_state_vessl_only": True,
        "created_unix": time.time(),
    }
    atomic_json(RECEIPT, payload)
    update("PASS", receipt=str(RECEIPT), decision=decision)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        update("HARD_STOP", error=f"{type(error).__name__}: {error}")
        raise
    raise SystemExit(exit_code)
