"""Acceleration-aware PromptMR+ inference router for the final VESSL model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import torch

try:
    from utils.learning.promptmr_mask_conditioning import validate_profile
except ModuleNotFoundError:
    from ops.vessl_promptmr_mask_conditioning import validate_profile
from utils.learning.promptmr_mask_router import select_component


ROUTED_SCHEMA = "vessl-acceleration-routed-promptmr-v1"
ROUTED_SCHEMA_V2 = "vessl-acceleration-routed-promptmr-v2"
EFFECTIVE_ACCELERATION_THRESHOLD = 4.0
ALLOWED_TTA_VIEW_SEQUENCES = (
    ["identity"],
    ["identity", "flip_lr"],
    ["identity", "flip_lr", "flip_ud", "rot180"],
)
R19_BBOX_LOSS_FAMILY = (
    "winner_foreground_ssim_l1_sqrt_area_plus_official384_bbox05_v2"
)
R19_PLAIN_LOSS_FAMILY = "winner_foreground_ssim_l1_sqrt_area_v1"
R29_INPUT_MODE = "recon_zero_filled_residual"
R29_ZERO_FILLED_DEFINITION = (
    "rss(fftshift(ifft2(ifftshift(masked_kspace),norm=ortho)))"
)


def _valid_embedded_contract(value: object, schema: str) -> bool:
    if not isinstance(value, dict) or value.get("schema") != schema:
        return False
    observed = value.get("contract_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    payload = dict(value)
    payload.pop("contract_sha256", None)
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    return hashlib.sha256(canonical).hexdigest() == observed


def infer_acceleration_from_mask(mask: np.ndarray | torch.Tensor) -> int:
    """Classify organizer Cartesian masks without relying on filenames."""
    if isinstance(mask, torch.Tensor):
        values = mask.detach()
        if values.is_cuda:
            values = values.cpu()
        array = values.numpy()
    else:
        array = np.asarray(mask)
    array = np.squeeze(array).astype(bool, copy=False)
    if array.ndim != 1 or array.size < 16:
        raise RuntimeError("acceleration router requires a 1-D sampling mask")
    acquired = int(np.count_nonzero(array))
    if acquired <= 0:
        raise RuntimeError("sampling mask contains no acquired columns")
    effective_acceleration = float(array.size) / acquired
    if not 2.0 <= effective_acceleration <= 12.0:
        raise RuntimeError(
            "sampling-mask acceleration is outside the supported range: "
            f"{effective_acceleration:.6f}"
        )
    return (
        4
        if effective_acceleration < EFFECTIVE_ACCELERATION_THRESHOLD
        else 8
    )


@dataclass(frozen=True)
class RoutedCheckpointContract:
    acc4_source_sha256: str
    acc8_source_sha256: str
    generalist_source_sha256: str | None
    rung: str
    num_cascades: int
    n_history: int
    generalist_component: str
    component_keys: tuple[str, ...]
    post_refiner_enabled: bool
    tta_views_by_acceleration: dict[int, tuple[str, ...]] | None

    @classmethod
    def validate(cls, checkpoint: dict[str, Any]) -> "RoutedCheckpointContract":
        schema = checkpoint.get("schema")
        if schema not in {ROUTED_SCHEMA, ROUTED_SCHEMA_V2}:
            raise RuntimeError("invalid routed PromptMR+ checkpoint schema")
        if checkpoint.get("all_components_trained_on_vessl") is not True:
            raise RuntimeError("routed checkpoint is not VESSL-final eligible")
        components = checkpoint.get("components")
        expected_components = (
            {"acc4", "acc8"}
            if schema == ROUTED_SCHEMA
            else {"generalist", "acc4", "acc8"}
        )
        if not isinstance(components, dict) or set(components) != expected_components:
            raise RuntimeError(
                "routed checkpoint contains an invalid component set"
            )
        tta_views = checkpoint.get("tta_views", ["identity"])
        if tta_views not in ALLOWED_TTA_VIEW_SEQUENCES:
            raise RuntimeError("invalid routed PromptMR+ TTA contract")
        raw_tta_by_acceleration = checkpoint.get(
            "tta_views_by_acceleration"
        )
        if raw_tta_by_acceleration is None:
            tta_views_by_acceleration = None
        else:
            if (
                not isinstance(raw_tta_by_acceleration, dict)
                or set(raw_tta_by_acceleration) != {"acc4", "acc8"}
                or tta_views != ["identity"]
            ):
                raise RuntimeError(
                    "invalid acceleration-specific TTA contract"
                )
            tta_views_by_acceleration = {}
            for key, acceleration in (("acc4", 4), ("acc8", 8)):
                views = raw_tta_by_acceleration[key]
                if views not in ALLOWED_TTA_VIEW_SEQUENCES:
                    raise RuntimeError(
                        f"invalid {key} TTA view sequence"
                    )
                tta_views_by_acceleration[acceleration] = tuple(views)
        post_refiner = checkpoint.get(
            "post_refiner",
            {"enabled": False},
        )
        if not isinstance(post_refiner, dict):
            raise RuntimeError("invalid post-refiner checkpoint contract")
        post_refiner_enabled = post_refiner.get("enabled") is True
        if post_refiner_enabled and (
            tta_views != ["identity"]
            or post_refiner.get("variant")
            not in {"NAF_S", "PLAIN_168K", "NAF_M", "NAF_L"}
            or post_refiner.get("views")
            not in (
                ["identity"],
                ["identity", "flip_lr"],
                ["identity", "flip_lr", "flip_ud", "rot180"],
            )
            or int(post_refiner.get("epoch", -1)) not in {1, 3, 10, 15, 20, 21}
            or not isinstance(post_refiner.get("post_refiner_state"), dict)
            or int(post_refiner.get("base_forward_count_per_slice", -1))
            != 1
            or int(post_refiner.get("refiner_forward_count_per_slice", -1))
            != 1
            or post_refiner.get("trained_on_vessl") is not True
            or post_refiner.get("external_learned_state_imported") is not False
            or not _valid_embedded_contract(
                post_refiner.get("training_data_contract"),
                "vessl-organizer-training-data-contract-v1",
            )
            or (
                int(post_refiner.get("epoch", -1)) == 10
                and (
                    post_refiner.get("role") != "main_output_post_refiner"
                    or post_refiner.get("parent_epoch") != 51
                    or post_refiner.get("late_branch_epochs") != [72, 81]
                    or post_refiner.get("trainable_parameter_scope")
                    != ("naf_s_plus_mask_conditioner" if bool(post_refiner.get("mask_conditioned", False)) else "naf_s_only")
                    or post_refiner.get("frozen_parameter_scope") != "main_c10_e51_all_parameters"
                    or post_refiner.get("main_parameters_updated") is not False
                    or post_refiner.get("views_batched") is not True
                    or int(post_refiner.get("optimizer_steps", -1)) != 44_736
                    or int(post_refiner.get("steps_per_epoch", -1)) != 4_672
                    or post_refiner.get("partial_terminal_epoch") is not True
                    or post_refiner.get("training_data") != "organizer_train_plus_val_final"
                    or post_refiner.get("loss_family") != "winner_foreground_ssim_l1_sqrt_area_plus_bbox05_v1"
                    or float(post_refiner.get("loss_lambda_l1", -1)) != 0.1
                    or post_refiner.get("sqrt_area_weighting") is not True
                    or float(post_refiner.get("bbox_loss_coefficient", -1)) != 0.5
                    or post_refiner.get("organizer_annotations_used_for_training") is not True
                    or not isinstance(post_refiner.get("annotation_contract"), dict)
                    or post_refiner["annotation_contract"].get("schema")
                    != "organizer-train-val-bbox-cell-weighting-v1"
                    or post_refiner["annotation_contract"].get("source")
                    != "organizer_train_plus_val_h5_annotations_training_only"
                    or post_refiner["annotation_contract"].get("slices")
                    != {"4": 2743, "8": 2699}
                    or post_refiner["annotation_contract"].get("accepted_boxes")
                    != {"4": 1524, "8": 1230}
                    or post_refiner["annotation_contract"].get("inference_annotation_access") is not False
                    or post_refiner.get("inference_annotation_access") is not False
                    or post_refiner.get("validation_used_for_checkpoint_selection") is not False
                    or post_refiner.get("training_base_route")
                    != "exact_acceleration_specialist_before_shared_naf_s_v1"
                    or post_refiner.get("sampler_policy")
                    != "equal_acc_real_acc8_real80_virtual20_v1"
                )
            )
            or (
                int(post_refiner.get("epoch", -1)) == 20
                and (
                    post_refiner.get("role") != "main_output_post_refiner"
                    or post_refiner.get("parent_epoch") != 50
                    or post_refiner.get("late_branch_epochs") != [51, 70]
                    or post_refiner.get("trainable_parameter_scope")
                    != (
                        "naf_s_plus_mask_conditioner"
                        if bool(post_refiner.get("mask_conditioned", False))
                        else "naf_s_only"
                    )
                    or post_refiner.get("frozen_parameter_scope")
                    != "main_c10_e50_all_parameters"
                    or post_refiner.get("main_parameters_updated") is not False
                    or post_refiner.get("views_batched") is not True
                    or int(post_refiner.get("optimizer_steps", -1)) != 91_141
                    or int(post_refiner.get("steps_per_epoch", -1)) != 4_672
                    or post_refiner.get("partial_terminal_epoch") is not True
                    or post_refiner.get("training_data")
                    != "organizer_train_plus_val_final"
                    or post_refiner.get("loss_family")
                    != "winner_foreground_ssim_l1_sqrt_area_v1"
                    or float(post_refiner.get("loss_lambda_l1", -1)) != 0.1
                    or post_refiner.get("sqrt_area_weighting") is not True
                    or post_refiner.get(
                        "validation_used_for_checkpoint_selection"
                    )
                    is not False
                )
            )
            or (
                int(post_refiner.get("epoch", -1)) == 21
                and (
                    post_refiner.get("role") != "main_output_post_refiner"
                    or post_refiner.get("parent_epoch") != 49
                    or post_refiner.get("late_branch_epochs") != [50, 70]
                    or post_refiner.get("trainable_parameter_scope")
                    != (
                        "naf_s_plus_mask_conditioner"
                        if bool(post_refiner.get("mask_conditioned", False))
                        else "naf_s_only"
                    )
                    or post_refiner.get("frozen_parameter_scope")
                    != "main_c10_e49_all_parameters"
                    or post_refiner.get("main_parameters_updated") is not False
                    or post_refiner.get("views_batched") is not True
                    or int(post_refiner.get("optimizer_steps", -1)) != 91_231
                    or int(
                        post_refiner.get("lr_horizon_optimizer_steps", -1)
                    )
                    != 93_567
                    or post_refiner.get("input_mode") != R29_INPUT_MODE
                    or post_refiner.get("zero_filled_definition")
                    != R29_ZERO_FILLED_DEFINITION
                    or post_refiner.get("normalization")
                    != "shared_detached_reconstruction_amax"
                    or post_refiner.get("spatial_match")
                    != "center_crop_then_zero_pad"
                    or int(post_refiner.get("steps_per_epoch", -1)) != 4_672
                    or post_refiner.get("partial_terminal_epoch") is not True
                    or post_refiner.get("training_data")
                    != "organizer_train_plus_val_final"
                    or post_refiner.get("loss_family")
                    not in {R19_PLAIN_LOSS_FAMILY, R19_BBOX_LOSS_FAMILY}
                    or float(post_refiner.get("loss_lambda_l1", -1)) != 0.1
                    or post_refiner.get("sqrt_area_weighting") is not True
                    or post_refiner.get(
                        "validation_used_for_checkpoint_selection"
                    )
                    is not False
                    or (
                        post_refiner.get("loss_family")
                        == R19_BBOX_LOSS_FAMILY
                        and (
                            float(
                                post_refiner.get("bbox_loss_coefficient", -1)
                            )
                            != 0.5
                            or post_refiner.get(
                                "organizer_annotations_used_for_training"
                            )
                            is not True
                            or not isinstance(
                                post_refiner.get("annotation_contract"), dict
                            )
                            or post_refiner["annotation_contract"].get(
                                "schema"
                            )
                            != "organizer-train-val-official384-bbox-cell-weighting-v2"
                            or post_refiner["annotation_contract"].get(
                                "source_coordinate_frame"
                            )
                            != [384, 384]
                            or post_refiner["annotation_contract"].get(
                                "training_tensor_alignment"
                            )
                            != "test_part_center_crop_then_zero_pad_v1"
                        )
                    )
                    or (
                        post_refiner.get("loss_family")
                        == R19_PLAIN_LOSS_FAMILY
                        and (
                            post_refiner.get("bbox_loss_coefficient")
                            is not None
                            or post_refiner.get(
                                "organizer_annotations_used_for_training"
                            )
                            is not False
                        )
                    )
                )
            )
        ):
            raise RuntimeError("invalid enabled post-refiner contract")
        routing = checkpoint.get("routing_contract")
        generalist_component = (
            routing.get("generalist_component")
            if isinstance(routing, dict)
            else None
        )
        generalist_valid = (
            generalist_component in {"acc4", "acc8"}
            if schema == ROUTED_SCHEMA
            else generalist_component == "generalist"
        )
        if (
            not isinstance(routing, dict)
            or routing.get("schema")
            != (
                "promptmr-legal-mask-routing-contract-v1"
                if schema == ROUTED_SCHEMA
                else "promptmr-legal-mask-routing-contract-v2"
            )
            or not generalist_valid
            or routing.get("unknown_or_mismatch")
            != routing.get("generalist_component")
            or routing.get("public_frequency_weighting") is not False
        ):
            raise RuntimeError("invalid legal-mask routing contract")
        source_hashes = {}
        architectures = set()
        component_keys = (
            ("acc4", "acc8")
            if schema == ROUTED_SCHEMA
            else ("generalist", "acc4", "acc8")
        )
        for component_key in component_keys:
            component = components[component_key]
            rung = component.get("rung") if isinstance(component, dict) else None
            n_history = (
                component.get("n_history")
                if isinstance(component, dict)
                else None
            )
            num_cascades = (
                component.get("num_cascades")
                if isinstance(component, dict)
                else None
            )
            if (
                not isinstance(component, dict)
                or (rung, num_cascades, n_history)
                not in {
                    ("R1", 8, 0),
                    ("R2", 8, 0),
                    ("R1", 8, 11),
                    ("R2", 10, 0),
                    ("R2", 12, 0),
                }
                or component.get("scratch") is not True
                or component.get("external_learned_state") is not False
                or component.get("trained_on_vessl") is not True
                or not isinstance(component.get("model"), dict)
            ):
                raise RuntimeError(
                    f"invalid {component_key} VESSL component contract"
                )
            mask_conditioning = component.get("mask_conditioning")
            if mask_conditioning is not None:
                if component_key != "acc8" or not isinstance(
                    mask_conditioning, dict
                ):
                    raise RuntimeError(
                        "mask conditioning is valid only on the ACC8 component"
                    )
                registration = validate_profile(mask_conditioning)
                if registration is None:
                    raise RuntimeError("empty mask-conditioning registration")
                if (
                    int(component.get("parameter_count", -1))
                    != int(registration["parameter_count"])
                    or int(component.get("trainable_parameter_count", -1))
                    != int(registration["trainable_parameter_count"])
                ):
                    raise RuntimeError(
                        "packaged mask-conditioning parameter contract mismatch"
                    )
            source_sha256 = component.get("source_checkpoint_sha256")
            if (
                not isinstance(source_sha256, str)
                or len(source_sha256) != 64
            ):
                raise RuntimeError(
                    f"invalid {component_key} source checkpoint SHA-256"
                )
            source_hashes[component_key] = source_sha256
            architectures.add(
                (str(rung), int(num_cascades), int(n_history))
            )
        if len(architectures) != 1:
            raise RuntimeError(
                "routed PromptMR+ components have incompatible architectures"
            )
        rung, num_cascades, n_history = architectures.pop()
        return cls(
            acc4_source_sha256=source_hashes["acc4"],
            acc8_source_sha256=source_hashes["acc8"],
            generalist_source_sha256=source_hashes.get("generalist"),
            rung=rung,
            num_cascades=num_cascades,
            n_history=n_history,
            generalist_component=str(generalist_component),
            component_keys=component_keys,
            post_refiner_enabled=post_refiner_enabled,
            tta_views_by_acceleration=tta_views_by_acceleration,
        )


class AccelerationRoutedPromptMR(torch.nn.Module):
    """Keep both experts on CPU and make exactly one GPU-resident at a time.

    The branches deliberately live in a plain dictionary instead of a
    ``ModuleDict``.  This prevents an evaluator's incidental ``model.to(cuda)``
    call from materializing both experts and exceeding the 8 GiB deployment
    budget.  This class is inference-only; final checkpoints store the two
    source state dictionaries, not this wrapper's ``state_dict``.
    """

    def __init__(
        self,
        acc4_model: torch.nn.Module,
        acc8_model: torch.nn.Module,
        *,
        generalist_component: str,
        generalist_model: torch.nn.Module | None = None,
    ):
        super().__init__()
        if generalist_component not in {"generalist", "acc4", "acc8"}:
            raise ValueError("invalid generalist component")
        if (generalist_component == "generalist") != (
            generalist_model is not None
        ):
            raise ValueError(
                "a dedicated generalist route requires exactly one model"
            )
        self.branches: dict[str, torch.nn.Module] = {
            "acc4": acc4_model.cpu().eval(),
            "acc8": acc8_model.cpu().eval(),
        }
        if generalist_model is not None:
            self.branches["generalist"] = generalist_model.cpu().eval()
        self.last_route: int | str | None = None
        self.last_route_receipt: dict[str, object] | None = None
        self.generalist_component = generalist_component
        self.volume_route: str | None = None
        self.active_branch: str | None = None
        self.active_device = torch.device("cpu")

    def train(self, mode: bool = True):
        # Final reconstruction is inference-only.  Preserve normal Module API
        # behavior while applying the flag to the intentionally unregistered
        # experts.
        super().train(mode)
        for branch in self.branches.values():
            branch.train(mode)
        return self

    def to(self, *args, **kwargs):
        # Do not eagerly move both branches.  The actual input device at the
        # first forward is authoritative and activates one branch lazily.
        return self

    def _activate(self, key: str, device: torch.device) -> torch.nn.Module:
        if self.active_branch == key and self.active_device == device:
            return self.branches[key]
        if self.active_branch is not None:
            self.branches[self.active_branch].to(device="cpu")
            if self.active_device.type == "cuda":
                torch.cuda.empty_cache()
        branch = self.branches[key]
        branch.to(device=device)
        branch.eval()
        self.active_branch = key
        self.active_device = device
        return branch

    def release_active(self) -> None:
        """Reset volume routing while retaining the one selected expert.

        The next volume reuses the resident branch when its exact mask route
        matches. ``_activate`` still offloads on an actual route change, so at
        most one PromptMR expert is resident on CUDA.
        """
        self.volume_route = None
        self.last_route_receipt = None

    def forward(self, prepared, **kwargs):
        acceleration = int(prepared.acceleration)
        if acceleration not in (4, 8):
            raise RuntimeError(
                f"unsupported routed acceleration: {acceleration}"
            )
        if self.volume_route is None:
            key, receipt = select_component(
                prepared.mask,
                declared_acceleration=acceleration,
                generalist_component=self.generalist_component,
            )
            self.volume_route = key
            self.last_route_receipt = receipt
        else:
            key = self.volume_route
        device = prepared.kspace.device
        branch = self._activate(key, device)
        self.last_route = (
            "generalist" if key == "generalist" else int(key[-1])
        )
        return branch(prepared, **kwargs)
