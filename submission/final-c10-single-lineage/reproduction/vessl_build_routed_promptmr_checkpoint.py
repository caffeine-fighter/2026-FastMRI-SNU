#!/usr/bin/env python3
"""Package two scratch VESSL PromptMR+ arms into one routed checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

try:
    from utils.learning.promptmr_mask_conditioning import validate_profile
except ModuleNotFoundError:
    from ops.vessl_promptmr_mask_conditioning import validate_profile


SCHEMA = "vessl-acceleration-routed-promptmr-v1"
SCHEMA_V2 = "vessl-acceleration-routed-promptmr-v2"
PINNED_SOURCE_COMMIT = "934eeda6d4d18cd39e406fa1eee9e1f70603cb5e"
VESSL_RESULT_ROOT = Path("/root/result")
REGISTERED_ARCHITECTURES = {
    ("R1", 8, 0): (36_219_065, 36_219_065),
    ("R2", 8, 0): (43_645_369, 36_595_129),
    ("R1", 8, 11): (41_028_793, 36_190_393),
    ("R2", 10, 0): (54_090_459, 45_381_339),
    ("R2", 12, 0): (64_535_549, 54_167_549),
}
REGISTERED_REFINERS = {
    "NAF_S": 72_625,
    "PLAIN_168K": 168_049,
    "NAF_M": 248_641,
    "NAF_L": 815_713,
}
MASK_CONDITIONER_PARAMETERS = {"NAF_S": 1_440}
MASK_CONDITIONING_CONTRACT = {
    "enabled": True,
    "source": "input_kspace_mask_exact_route",
    "routes": {"unknown": 0, "acc4": 4, "acc8": 8},
    "parameter_count": 1_440,
    "zero_initialized": True,
    "application": "stem_and_each_naf_block_film",
    "maximum_scale_shift": 0.1,
}
R19_BBOX_LOSS_FAMILY = (
    "winner_foreground_ssim_l1_sqrt_area_plus_official384_bbox05_v2"
)
R19_PLAIN_LOSS_FAMILY = "winner_foreground_ssim_l1_sqrt_area_v1"
R29_INPUT_MODE = "recon_zero_filled_residual"
R29_ZERO_FILLED_DEFINITION = (
    "rss(fftshift(ifft2(ifftshift(masked_kspace),norm=ortho)))"
)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def valid_embedded_contract(value: object, schema: str) -> bool:
    if not isinstance(value, dict) or value.get("schema") != schema:
        return False
    observed = value.get("contract_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    payload = dict(value)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest() == observed


def mask_conditioning_profile(
    checkpoint: dict,
    config: dict,
    *,
    acceleration: int | None,
    rung: str,
    n_history: int,
) -> dict | None:
    """Recover and validate a VESSL-trained conditioned component contract."""

    keys = [
        key
        for key in (
            "mask_conditioned_dc",
            "mask_conditioned_prompt",
            "mask_conditioned_c10_last_cascade_prompt",
        )
        if config.get(key) is not None
    ]
    if not keys:
        return None
    if len(keys) != 1:
        raise RuntimeError(
            "conditioned component must contain exactly one registration"
        )
    key = keys[0]
    if key == "mask_conditioned_c10_last_cascade_prompt":
        if acceleration not in {4, 8}:
            raise RuntimeError("C10 mask conditioning requires an acceleration route")
    elif acceleration != 8:
        raise RuntimeError(
            "legacy mask conditioning is registered only for the ACC8 component"
        )
    profile = {
        "enabled": True,
        "rung": rung,
        "n_history": n_history,
        "train_acceleration": f"acc{acceleration}",
        "parameter_efficient_specialization": (
            True if key == "mask_conditioned_c10_last_cascade_prompt"
            else config.get("parameter_efficient_specialization", False)
        ),
        "parameter_count": int(config.get("parameter_count", -1)),
        "trainable_parameter_count": int(
            config.get("specialist_trainable_parameter_count", -1)
        ),
        "trainable_parameter_scope": config.get(
            "specialist_trainable_scope"
        ),
        "frozen_parameter_scope": config.get("specialist_frozen_scope"),
        "unknown_mask_route": "exact_generalist_identity",
        key: config.get(key),
    }
    registration = validate_profile(profile)
    if registration is None:
        raise RuntimeError("conditioned component lost its registration")
    if (
        checkpoint.get(key) != registration["contract"]
        or checkpoint.get("mask_conditioning_kind")
        != registration["kind"]
    ):
        raise RuntimeError("conditioned checkpoint metadata is inconsistent")
    model = checkpoint.get("model", {})
    marker = (
        ".mask_conditioner."
        if registration["kind"] == "dc"
        else ".mask_prompt_conditioner."
    )
    conditioner_keys = [name for name in model if marker in name]
    if len(conditioner_keys) != 4:
        raise RuntimeError("conditioned checkpoint state namespace mismatch")
    if registration["kind"] == "dc":
        original_dc = [
            name
            for name in model
            if name.endswith("parametrizations.dc_weight.original")
        ]
        if len(original_dc) != 12:
            raise RuntimeError("conditioned DC originals are incomplete")
    return profile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component(
    path: Path,
    expected_sha256: str,
    acceleration: int | None,
) -> dict:
    label = "generalist" if acceleration is None else f"acc{acceleration}"
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(VESSL_RESULT_ROOT):
        raise RuntimeError(
            f"{label} source is outside the VESSL result root"
        )
    observed = sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"{label} checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    sampler = checkpoint.get("sampler")
    model_config = config.get("model") if isinstance(config, dict) else None
    rung = str(checkpoint.get("rung", ""))
    n_history = (
        int(model_config.get("n_history", -1))
        if isinstance(model_config, dict)
        else -1
    )
    num_cascades = (
        int(model_config.get("num_cascades", -1))
        if isinstance(model_config, dict)
        else -1
    )
    registered_counts = REGISTERED_ARCHITECTURES.get(
        (rung, num_cascades, n_history)
    )
    conditioning = (
        mask_conditioning_profile(
            checkpoint,
            config,
            acceleration=acceleration,
            rung=rung,
            n_history=n_history,
        )
        if isinstance(config, dict)
        else None
    )
    expected_parameter_count = (
        int(conditioning["parameter_count"])
        if conditioning is not None
        else registered_counts[0] if registered_counts is not None else -1
    )
    expected_trainable_count = (
        int(conditioning["trainable_parameter_count"])
        if conditioning is not None
        else registered_counts[1] if registered_counts is not None else -1
    )
    observed_trainable = (
        int(
            config.get(
                "trainable_parameter_count",
                expected_trainable_count,
            )
        )
        if isinstance(config, dict)
        else -1
    )
    train_route = (
        str(config.get("train_acceleration", ""))
        if isinstance(config, dict)
        else ""
    )
    sampler_route = (
        str(sampler.get("route", ""))
        if isinstance(sampler, dict)
        else ""
    )
    if (
        checkpoint.get("format_version") != 2
        or checkpoint.get("model_family") != "promptmr-plus-reduced"
        or registered_counts is None
        or checkpoint.get("scratch") is not True
        or checkpoint.get("external_learned_state") is not False
        or not isinstance(checkpoint.get("model"), dict)
        or not isinstance(config, dict)
        or config.get("rung") != rung
        or config.get("scratch") is not True
        or config.get("external_learned_state") is not False
        or config.get("source_commit") != PINNED_SOURCE_COMMIT
        or not isinstance(sampler, dict)
        or train_route
        not in ({"all"} if acceleration is None else {"all", label})
        or sampler_route
        not in ({"all"} if acceleration is None else {"all", label})
        or int(config.get("parameter_count", -1))
        != expected_parameter_count
        or observed_trainable != expected_trainable_count
    ):
        raise RuntimeError(f"{label} checkpoint is not VESSL scratch")
    return {
        "acceleration": acceleration,
        "role": label,
        "rung": rung,
        "num_cascades": num_cascades,
        "n_history": n_history,
        "parameter_count": expected_parameter_count,
        "trainable_parameter_count": expected_trainable_count,
        "mask_conditioning": conditioning,
        "train_acceleration": train_route,
        "scratch": True,
        "external_learned_state": False,
        "trained_on_vessl": True,
        "source_checkpoint": str(resolved_path),
        "source_checkpoint_sha256": observed,
        "source_commit": PINNED_SOURCE_COMMIT,
        "source_config": config,
        "source_step": checkpoint.get("global_optimizer_step"),
        "source_epoch": checkpoint.get("epoch"),
        "model": checkpoint["model"],
    }


def post_refiner_component(
    path: Path,
    expected_sha256: str,
    *,
    base_checkpoint_sha256: str,
    acc4_route_sha256: str,
    acc8_route_sha256: str,
) -> dict:
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(VESSL_RESULT_ROOT):
        raise RuntimeError("post-refiner source is outside /root/result")
    observed = sha256(path)
    if observed != expected_sha256:
        raise RuntimeError("post-refiner checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    variant = str(checkpoint.get("variant", ""))
    views = checkpoint.get("views")
    module_sha256 = checkpoint.get("post_refiner_module_sha256")
    mask_conditioned = bool(checkpoint.get("mask_conditioned", False))
    expected_parameter_count = REGISTERED_REFINERS.get(variant, -1) + (
        MASK_CONDITIONER_PARAMETERS.get(variant, 0)
        if mask_conditioned
        else 0
    )
    expected_trainable_scope = (
        "naf_s_plus_mask_conditioner" if mask_conditioned else "naf_s_only"
    )
    if (
        checkpoint.get("schema") != "vessl-base-once-post-refiner-v1"
        or checkpoint.get("state") != "COMPLETE"
        or variant not in REGISTERED_REFINERS
        or int(checkpoint.get("parameter_count", -1))
        != expected_parameter_count
        or (
            mask_conditioned
            and checkpoint.get("mask_conditioning")
            != MASK_CONDITIONING_CONTRACT
        )
        or views
        not in (
            ["identity"],
            ["identity", "flip_lr"],
            ["identity", "flip_lr", "flip_ud", "rot180"],
        )
        or not isinstance(checkpoint.get("post_refiner_state"), dict)
        or checkpoint.get("base_checkpoint_sha256")
        != base_checkpoint_sha256
        or checkpoint.get("training_base_route")
        != "exact_acceleration_specialist_before_shared_naf_s_v1"
        or checkpoint.get("routed_branch_sha256") != {
            "acc4": acc4_route_sha256,
            "acc8": acc8_route_sha256,
        }
        or checkpoint.get("sampler_policy")
        != "equal_acc_real_acc8_real80_virtual20_v1"
        or not isinstance(module_sha256, str)
        or len(module_sha256) != 64
        or float(checkpoint.get("maximum_residual_fraction", -1)) != 0.05
        or float(checkpoint.get("peak_lr", -1)) != 0.0001
        or float(checkpoint.get("weight_decay", -1)) != 0.0001
        or int(checkpoint.get("seed", -1)) != 430
        or int(checkpoint.get("epoch", -1)) not in {1, 3, 10, 15, 20, 21}
        or checkpoint.get("training_data")
        not in {
            "organizer_train_only",
            "organizer_train_plus_val_final",
        }
        or checkpoint.get("leaderboard_data_read") is not False
        or checkpoint.get("checkpoint_selection_influence") is not False
        or checkpoint.get("trained_on_vessl") is not True
        or checkpoint.get("external_learned_state_imported") is not False
        or not valid_embedded_contract(
            checkpoint.get("training_data_contract"),
            "vessl-organizer-training-data-contract-v1",
        )
        or (
            int(checkpoint.get("epoch", -1)) == 10
            and (
                checkpoint.get("role") != "main_output_post_refiner"
                or checkpoint.get("parent_epoch") != 51
                or checkpoint.get("late_branch_epochs") != [72, 81]
                or checkpoint.get("trainable_parameter_scope") != expected_trainable_scope
                or checkpoint.get("frozen_parameter_scope") != "main_c10_e51_all_parameters"
                or checkpoint.get("main_parameters_updated") is not False
                or checkpoint.get("views_batched") is not True
                or int(checkpoint.get("optimizer_step", -1)) != 44_736
                or int(checkpoint.get("steps_per_epoch", -1)) != 4_672
                or checkpoint.get("partial_terminal_epoch") is not True
                or checkpoint.get("training_data") != "organizer_train_plus_val_final"
                or checkpoint.get("loss_family") != "winner_foreground_ssim_l1_sqrt_area_plus_bbox05_v1"
                or float(checkpoint.get("loss_lambda_l1", -1)) != 0.1
                or checkpoint.get("sqrt_area_weighting") is not True
                or float(checkpoint.get("bbox_loss_coefficient", -1)) != 0.5
                or checkpoint.get("organizer_annotations_used_for_training") is not True
                or checkpoint.get("inference_annotation_access") is not False
                or not isinstance(checkpoint.get("annotation_contract"), dict)
                or checkpoint.get("validation_used_for_checkpoint_selection") is not False
            )
        )
        or (
            int(checkpoint.get("epoch", -1)) == 15
            and (
                checkpoint.get("role") != "main_output_post_refiner"
                or checkpoint.get("parent_epoch") != 40
                or checkpoint.get("late_branch_epochs") != [41, 55]
                or checkpoint.get("trainable_parameter_scope")
                != expected_trainable_scope
                or checkpoint.get("frozen_parameter_scope")
                != "main_c10_e40_all_parameters"
                or checkpoint.get("main_parameters_updated") is not False
                or checkpoint.get("views_batched") is not True
            )
        )
        or (
            int(checkpoint.get("epoch", -1)) == 20
            and (
                checkpoint.get("role") != "main_output_post_refiner"
                or checkpoint.get("parent_epoch") != 50
                or checkpoint.get("late_branch_epochs") != [51, 70]
                or checkpoint.get("trainable_parameter_scope")
                != expected_trainable_scope
                or checkpoint.get("frozen_parameter_scope")
                != "main_c10_e50_all_parameters"
                or checkpoint.get("main_parameters_updated") is not False
                or checkpoint.get("views_batched") is not True
                or int(checkpoint.get("optimizer_step", -1)) != 91_141
                or int(checkpoint.get("steps_per_epoch", -1)) != 4_672
                or checkpoint.get("partial_terminal_epoch") is not True
                or checkpoint.get("training_data")
                != "organizer_train_plus_val_final"
                or checkpoint.get("loss_family")
                != "winner_foreground_ssim_l1_sqrt_area_v1"
                or float(checkpoint.get("loss_lambda_l1", -1)) != 0.1
                or checkpoint.get("sqrt_area_weighting") is not True
                or checkpoint.get(
                    "validation_used_for_checkpoint_selection"
                )
                is not False
            )
        )
        or (
            int(checkpoint.get("epoch", -1)) == 21
            and (
                checkpoint.get("role") != "main_output_post_refiner"
                or checkpoint.get("parent_epoch") != 49
                or checkpoint.get("late_branch_epochs") != [50, 70]
                or checkpoint.get("trainable_parameter_scope")
                != expected_trainable_scope
                or checkpoint.get("frozen_parameter_scope")
                != "main_c10_e49_all_parameters"
                or checkpoint.get("main_parameters_updated") is not False
                or checkpoint.get("views_batched") is not True
                or int(checkpoint.get("optimizer_step", -1)) != 91_231
                or int(
                    checkpoint.get("lr_horizon_optimizer_steps", -1)
                )
                != 93_567
                or checkpoint.get("input_mode") != R29_INPUT_MODE
                or checkpoint.get("zero_filled_definition")
                != R29_ZERO_FILLED_DEFINITION
                or checkpoint.get("normalization")
                != "shared_detached_reconstruction_amax"
                or checkpoint.get("spatial_match")
                != "center_crop_then_zero_pad"
                or int(checkpoint.get("steps_per_epoch", -1)) != 4_672
                or checkpoint.get("partial_terminal_epoch") is not True
                or checkpoint.get("training_data")
                != "organizer_train_plus_val_final"
                or checkpoint.get("loss_family")
                not in {R19_PLAIN_LOSS_FAMILY, R19_BBOX_LOSS_FAMILY}
                or float(checkpoint.get("loss_lambda_l1", -1)) != 0.1
                or checkpoint.get("sqrt_area_weighting") is not True
                or checkpoint.get("validation_used_for_checkpoint_selection")
                is not False
                or (
                    checkpoint.get("loss_family") == R19_BBOX_LOSS_FAMILY
                    and (
                        float(checkpoint.get("bbox_loss_coefficient", -1))
                        != 0.5
                        or checkpoint.get(
                            "organizer_annotations_used_for_training"
                        )
                        is not True
                        or not isinstance(
                            checkpoint.get("annotation_contract"), dict
                        )
                        or checkpoint["annotation_contract"].get("schema")
                        != "organizer-train-val-official384-bbox-cell-weighting-v2"
                        or checkpoint["annotation_contract"].get(
                            "source_coordinate_frame"
                        )
                        != [384, 384]
                        or checkpoint["annotation_contract"].get(
                            "training_tensor_alignment"
                        )
                        != "test_part_center_crop_then_zero_pad_v1"
                    )
                )
                or (
                    checkpoint.get("loss_family") == R19_PLAIN_LOSS_FAMILY
                    and (
                        checkpoint.get("bbox_loss_coefficient") is not None
                        or checkpoint.get(
                            "organizer_annotations_used_for_training"
                        )
                        is not False
                    )
                )
            )
        )
    ):
        raise RuntimeError("invalid VESSL-scratch post-refiner checkpoint")
    return {
        "enabled": True,
        "role": checkpoint.get("role", "base_once_post_refiner"),
        "variant": variant,
        "views": views,
        "views_batched": checkpoint.get("views_batched", False),
        "mask_conditioned": mask_conditioned,
        "mask_conditioning": (
            checkpoint.get("mask_conditioning") if mask_conditioned else None
        ),
        "input_mode": checkpoint.get("input_mode"),
        "zero_filled_definition": checkpoint.get("zero_filled_definition"),
        "normalization": checkpoint.get("normalization"),
        "spatial_match": checkpoint.get("spatial_match"),
        "epoch": int(checkpoint["epoch"]),
        "parent_epoch": checkpoint.get("parent_epoch"),
        "late_branch_epochs": checkpoint.get("late_branch_epochs"),
        "optimizer_steps": checkpoint.get("optimizer_step"),
        "lr_horizon_optimizer_steps": checkpoint.get(
            "lr_horizon_optimizer_steps"
        ),
        "steps_per_epoch": checkpoint.get("steps_per_epoch"),
        "partial_terminal_epoch": checkpoint.get("partial_terminal_epoch"),
        "completed_epoch_equivalent": checkpoint.get(
            "completed_epoch_equivalent"
        ),
        "training_data": checkpoint.get("training_data"),
        "loss_family": checkpoint.get("loss_family"),
        "loss_lambda_l1": checkpoint.get("loss_lambda_l1"),
        "sqrt_area_weighting": checkpoint.get("sqrt_area_weighting"),
        "bbox_loss_coefficient": checkpoint.get("bbox_loss_coefficient"),
        "organizer_annotations_used_for_training": checkpoint.get(
            "organizer_annotations_used_for_training"
        ),
        "annotation_contract": checkpoint.get("annotation_contract"),
        "training_data_contract": checkpoint.get("training_data_contract"),
        "inference_annotation_access": checkpoint.get(
            "inference_annotation_access"
        ),
        "validation_used_for_checkpoint_selection": checkpoint.get(
            "validation_used_for_checkpoint_selection"
        ),
        "trainable_parameter_scope": checkpoint.get("trainable_parameter_scope"),
        "frozen_parameter_scope": checkpoint.get("frozen_parameter_scope"),
        "main_parameters_updated": checkpoint.get("main_parameters_updated"),
        "parameter_count": expected_parameter_count,
        "maximum_residual_fraction": 0.05,
        "base_forward_count_per_slice": 1,
        "refiner_forward_count_per_slice": 1,
        "trained_on_vessl": True,
        "external_learned_state_imported": False,
        "source_checkpoint": str(resolved_path),
        "source_checkpoint_sha256": observed,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "training_base_route": checkpoint.get("training_base_route"),
        "routed_branch_sha256": checkpoint.get("routed_branch_sha256"),
        "sampler_policy": checkpoint.get("sampler_policy"),
        "post_refiner_module_sha256": module_sha256,
        "post_refiner_state": checkpoint["post_refiner_state"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for acceleration in (4, 8):
        parser.add_argument(
            f"--acc{acceleration}-checkpoint",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--acc{acceleration}-sha256",
            required=True,
        )
    parser.add_argument("--generalist-checkpoint", type=Path)
    parser.add_argument("--generalist-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-refiner-checkpoint", type=Path)
    parser.add_argument("--post-refiner-sha256")
    parser.add_argument(
        "--tta-views",
        choices=(
            "identity",
            "identity_flip_lr",
            "identity_dihedral4",
            "acc4_identity_flip_lr",
            "acc8_identity_flip_lr",
        ),
        default="identity",
    )
    args = parser.parse_args()
    if (args.post_refiner_checkpoint is None) != (
        args.post_refiner_sha256 is None
    ):
        parser.error(
            "post-refiner checkpoint and SHA-256 must be provided together"
        )
    if (args.generalist_checkpoint is None) != (
        args.generalist_sha256 is None
    ):
        parser.error(
            "generalist checkpoint and SHA-256 must be provided together"
        )
    return args


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    acc4 = component(args.acc4_checkpoint, args.acc4_sha256, 4)
    acc8 = component(args.acc8_checkpoint, args.acc8_sha256, 8)
    generalist = (
        component(
            args.generalist_checkpoint,
            args.generalist_sha256,
            None,
        )
        if args.generalist_checkpoint is not None
        else None
    )
    architecture4 = (
        acc4["rung"],
        acc4["num_cascades"],
        acc4["n_history"],
    )
    architecture8 = (
        acc8["rung"],
        acc8["num_cascades"],
        acc8["n_history"],
    )
    if architecture4 != architecture8:
        raise RuntimeError(
            "routed components must share one registered architecture"
        )
    if generalist is not None:
        architecture_generalist = (
            generalist["rung"],
            generalist["num_cascades"],
            generalist["n_history"],
        )
        if architecture_generalist != architecture4:
            raise RuntimeError(
                "dedicated generalist and specialists must share architecture"
            )
        if (
            generalist["train_acceleration"] != "all"
            or acc4["train_acceleration"] != "acc4"
            or acc8["train_acceleration"] != "acc8"
        ):
            raise RuntimeError(
                "three-component routing requires one generalist and two exact specialists"
            )
    if (
        args.acc4_sha256 == args.acc8_sha256
        and (
            acc4["train_acceleration"] != "all"
            or acc8["train_acceleration"] != "all"
        )
    ):
        raise RuntimeError(
            "a shared routed source must be an all-acceleration generalist"
        )
    post_refiner = None
    if args.post_refiner_checkpoint is not None:
        if generalist is None and args.acc4_sha256 != args.acc8_sha256:
            raise RuntimeError(
                "routed post-refiner packaging requires a dedicated generalist base"
            )
        if args.tta_views not in {
            "identity",
            "acc8_identity_flip_lr",
        }:
            raise RuntimeError(
                "post-refiner permits only identity or sealed ACC8 routed TTA2"
            )
        post_refiner = post_refiner_component(
            args.post_refiner_checkpoint,
            args.post_refiner_sha256,
            base_checkpoint_sha256=(args.generalist_sha256 or args.acc4_sha256),
            acc4_route_sha256=args.acc4_sha256,
            acc8_route_sha256=args.acc8_sha256,
        )
    tta_views = {
        "identity": ["identity"],
        "identity_flip_lr": ["identity", "flip_lr"],
        "identity_dihedral4": [
            "identity",
            "flip_lr",
            "flip_ud",
            "rot180",
        ],
        "acc4_identity_flip_lr": ["identity"],
        "acc8_identity_flip_lr": ["identity"],
    }[args.tta_views]
    routed_tta_views = {
        "acc4_identity_flip_lr": {
            "acc4": ["identity", "flip_lr"],
            "acc8": ["identity"],
        },
        "acc8_identity_flip_lr": {
            "acc4": ["identity"],
            "acc8": ["identity", "flip_lr"],
        },
    }
    tta_views_by_acceleration = routed_tta_views.get(args.tta_views)
    generalists = [
        key
        for key, value in (("acc4", acc4), ("acc8", acc8))
        if value["train_acceleration"] == "all"
    ]
    if generalist is None and not generalists:
        raise RuntimeError(
            "routed package requires a VESSL-trained generalist fail-safe"
        )
    generalist_component = (
        "generalist"
        if generalist is not None
        else "acc8" if "acc8" in generalists else generalists[0]
    )
    payload = {
        "schema": SCHEMA_V2 if generalist is not None else SCHEMA,
        "all_components_trained_on_vessl": True,
        "architecture": {
            "rung": acc4["rung"],
            "num_cascades": acc4["num_cascades"],
            "n_history": acc4["n_history"],
            "parameter_count": acc4["parameter_count"],
            "trainable_parameter_count": acc4[
                "trainable_parameter_count"
            ],
        },
        "routing_feature": (
            "exact_legal_mask_family_with_generalist_fail_safe"
        ),
        "routing_contract": {
                "schema": (
                    "promptmr-legal-mask-routing-contract-v2"
                    if generalist is not None
                    else "promptmr-legal-mask-routing-contract-v1"
                ),
            "generator": {
                "center_fraction": 0.08,
                "acs_width": "round(native_width*0.08)",
                "acs_start": "(native_width-acs_width+1)//2",
                "outer_lines": "column%acceleration==residue",
            },
            "supported_accelerations": [4, 8],
            "generalist_component": generalist_component,
            "unknown_or_mismatch": generalist_component,
            "public_frequency_weighting": False,
        },
        "tta_views": tta_views,
        **(
            {
                "tta_views_by_acceleration": (
                    tta_views_by_acceleration
                )
            }
            if tta_views_by_acceleration is not None
            else {}
        ),
        "post_refiner": (
            post_refiner
            if post_refiner is not None
            else {
                "enabled": False,
                "role": "base_once_post_refiner",
            }
        ),
        "components": (
            {
                "generalist": generalist,
                "acc4": acc4,
                "acc8": acc8,
            }
            if generalist is not None
            else {"acc4": acc4, "acc8": acc8}
        ),
        "created_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        f".{args.output.name}.{os.getpid()}.tmp"
    )
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(
        {
            "output": str(args.output),
            "sha256": sha256(args.output),
            "acc4_source_sha256": args.acc4_sha256,
            "acc8_source_sha256": args.acc8_sha256,
            "generalist_source_sha256": args.generalist_sha256,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
