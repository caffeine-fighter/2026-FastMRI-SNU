#!/usr/bin/env python3
"""Train a bounded image refiner after a VESSL-scratch PromptMR generalist.

The expensive base reconstruction is frozen and evaluated exactly once per
sample.  Only the registered image-domain refiner is optimized.  This file is
intended for the final VESSL lineage: no RunPod weights, leaderboard payload,
or validation-driven checkpoint selection is accepted.
"""

from __future__ import annotations

import argparse
import h5py
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from utils.learning.promptmr_post_refiner import (
    BaseOnceRefinerTTA,
    INPUT_MODE_DIRECTIONAL,
    INPUT_MODE_ZF_CONTEXT,
    NAF_REFINER_VARIANTS,
    NAFResidualImageRefiner,
    ResidualImageRefiner,
    ZERO_FILLED_DEFINITION,
)
from utils.learning.promptmr_production import (
    BalancedAccelerationSampler,
    PromptMRProductionDataset,
    UpstreamSSIMObjective,
    build_rung_model,
)
from utils.model.promptmr_plus_adapter import (
    PromptMRInput,
    PromptMRPlusAdapter,
)


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
ALLOWED_VIEWS = ("identity", "flip_lr", "flip_ud", "rot180")
BBOX_ALIGNED_LOSS_FAMILY = (
    "winner_foreground_ssim_l1_sqrt_area_plus_official384_bbox05_v2"
)
BBOX_LOSS_COEFFICIENT = 0.5
WINNER_LOSS_FAMILIES = {
    "winner_foreground_ssim_l1_sqrt_area_v1",
    BBOX_ALIGNED_LOSS_FAMILY,
}


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_embedded_contract(value: object, schema: str) -> bool:
    if not isinstance(value, dict) or value.get("schema") != schema:
        return False
    observed = value.get("contract_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    payload = dict(value)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest() == observed


def valid_specialist_training_data_contract(value: object) -> bool:
    return (
        valid_embedded_contract(
            value,
            "vessl-specialist-organizer-training-data-contract-v1",
        )
        and value.get("root") == "/root/Data/train"
        and value.get("leaderboard_data_read") is False
        and isinstance(value.get("trusted_manifest_sha256"), str)
        and len(value["trusted_manifest_sha256"]) == 64
        and isinstance(value.get("file_shape_identity_inventory_sha256"), str)
        and len(value["file_shape_identity_inventory_sha256"]) == 64
    )


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--acc4-checkpoint", type=Path)
    parser.add_argument("--acc4-checkpoint-sha256")
    parser.add_argument("--acc8-checkpoint", type=Path)
    parser.add_argument("--acc8-checkpoint-sha256")
    parser.add_argument(
        "--variant",
        choices=tuple(REGISTERED_REFINERS),
        required=True,
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=ALLOWED_VIEWS,
        required=True,
    )
    parser.add_argument("--mask-conditioned", action="store_true")
    parser.add_argument(
        "--input-mode",
        choices=(INPUT_MODE_DIRECTIONAL, INPUT_MODE_ZF_CONTEXT),
        default=INPUT_MODE_DIRECTIONAL,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        choices=(1, 3, 10, 15, 20, 21),
        required=True,
    )
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=None,
        help="Presealed terminal optimizer-step budget; required by the E50 deadline path",
    )
    parser.add_argument(
        "--lr-horizon-optimizer-steps",
        type=int,
        default=None,
        help=(
            "Presealed LR horizon. The R25 path stops before this horizon "
            "without compressing the cosine tail."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--trusted-data-manifest", type=Path, required=True)
    parser.add_argument("--extra-train-root", type=Path)
    parser.add_argument("--extra-trusted-data-manifest", type=Path)
    parser.add_argument(
        "--loss-family",
        choices=(
            "exact_upstream_ssim",
            "winner_foreground_ssim_l1_sqrt_area_v1",
            BBOX_ALIGNED_LOSS_FAMILY,
        ),
        default="exact_upstream_ssim",
    )
    parser.add_argument("--peak-lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    hashes = {
        "base": args.base_checkpoint_sha256,
        "acc4": args.acc4_checkpoint_sha256,
        "acc8": args.acc8_checkpoint_sha256,
    }
    for label, value in hashes.items():
        if value is not None and (
            value.lower() != value or len(value) != 64
        ):
            parser.error(f"{label} checkpoint SHA-256 is malformed")
    routed_values = (
        args.acc4_checkpoint,
        args.acc4_checkpoint_sha256,
        args.acc8_checkpoint,
        args.acc8_checkpoint_sha256,
    )
    if any(value is not None for value in routed_values) and not all(
        value is not None for value in routed_values
    ):
        parser.error("routed ACC4/ACC8 checkpoints must be supplied together")
    if (
        args.views[0] != "identity"
        or len(set(args.views)) != len(args.views)
        or tuple(args.views)
        not in (
            ("identity",),
            ("identity", "flip_lr"),
            ("identity", "flip_lr", "flip_ud", "rot180"),
        )
    ):
        parser.error(
            "views must be identity, identity+flip_lr, "
            "or identity-first dihedral4"
        )
    if args.peak_lr != 0.0001 or args.weight_decay != 0.0001:
        parser.error("post-refiner optimizer scalar contract changed")
    if args.seed != 430:
        parser.error("post-refiner seed contract changed")
    if args.mask_conditioned and args.variant != "NAF_S":
        parser.error("mask conditioning is registered only for NAF_S")
    if args.epochs == 10:
        if args.optimizer_steps != 44_736:
            parser.error("the asymmetric terminal path requires exactly 44736 optimizer steps")
        if args.acc4_checkpoint is None or args.acc8_checkpoint is None:
            parser.error("the R13 NAF_S path requires both routed specialists")
    elif args.epochs == 20:
        if args.optimizer_steps != 91_141:
            parser.error("the legacy deadline path requires exactly 91141 optimizer steps")
    elif args.epochs == 21:
        if (
            args.optimizer_steps != 91_231
            or args.lr_horizon_optimizer_steps != 93_567
            or args.input_mode != INPUT_MODE_ZF_CONTEXT
        ):
            parser.error(
                "the R25 terminal path requires stop=91231 and "
                "LR-horizon=93567 optimizer steps with R29 ZF context"
            )
        if args.acc4_checkpoint is None or args.acc8_checkpoint is None:
            parser.error("the R25 NAF_S path requires both routed specialists")
    elif args.optimizer_steps is not None:
        parser.error("an optimizer-step override is not registered for this epoch path")
    if args.epochs != 21 and args.lr_horizon_optimizer_steps is not None:
        parser.error("an independent LR horizon is registered only for R25")
    if (args.extra_train_root is None) != (
        args.extra_trusted_data_manifest is None
    ):
        parser.error(
            "extra train root and trusted manifest must be supplied together"
        )
    if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY and (
        args.epochs != 21
        or args.extra_train_root is None
        or args.mask_conditioned
        or tuple(args.views) != ("identity", "flip_lr")
    ):
        parser.error(
            "bbox-aligned loss is registered only for the final plain NAF_S R19 full-data path"
        )
    return args


def registered_parameter_count(variant: str, mask_conditioned: bool) -> int:
    return REGISTERED_REFINERS[variant] + (
        MASK_CONDITIONER_PARAMETERS.get(variant, 0)
        if mask_conditioned
        else 0
    )


def make_refiner(
    variant: str,
    *,
    mask_conditioned: bool = False,
    input_mode: str = INPUT_MODE_DIRECTIONAL,
) -> torch.nn.Module:
    if variant == "PLAIN_168K":
        refiner = ResidualImageRefiner()
    else:
        refiner = NAFResidualImageRefiner(
            variant=variant,
            mask_conditioned=mask_conditioned,
            input_mode=input_mode,
        )
    observed = sum(parameter.numel() for parameter in refiner.parameters())
    if observed != registered_parameter_count(variant, mask_conditioned):
        raise RuntimeError("registered post-refiner parameter count drifted")
    return refiner


class WinnerForegroundSSIML1SqrtArea(torch.nn.Module):
    """Target-foreground SSIM+L1 with winner-style sqrt-area weighting."""

    def __init__(self, lambda_l1: float = 0.1, win_size: int = 7):
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.win_size = int(win_size)

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        data_range: torch.Tensor,
        foreground: torch.Tensor,
    ) -> torch.Tensor:
        x = output.unsqueeze(1)
        y = target.unsqueeze(1)
        mask = foreground.unsqueeze(1).float()
        if not bool(mask.any()):
            raise RuntimeError("winner-style foreground mask is empty")
        pad = self.win_size // 2
        ux = F.avg_pool2d(x, self.win_size, 1, pad)
        uy = F.avg_pool2d(y, self.win_size, 1, pad)
        uxx = F.avg_pool2d(x * x, self.win_size, 1, pad)
        uyy = F.avg_pool2d(y * y, self.win_size, 1, pad)
        uxy = F.avg_pool2d(x * y, self.win_size, 1, pad)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy
        ranges = data_range.reshape(-1, 1, 1, 1)
        c1 = (0.01 * ranges) ** 2
        c2 = (0.03 * ranges) ** 2
        ssim_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
            (ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-12
        )
        mass = mask.sum(dim=(-3, -2, -1)).clamp_min(1.0)
        ssim_loss = ((1.0 - ssim_map) * mask).sum(
            dim=(-3, -2, -1)
        ) / mass
        l1 = (torch.abs(output - target) * foreground.float()).sum(
            dim=(-2, -1)
        ) / foreground.float().sum(dim=(-2, -1)).clamp_min(1.0)
        l1 = l1 / data_range.clamp_min(1e-8)
        area_fraction = foreground.float().mean(dim=(-2, -1))
        area_weight = torch.sqrt(area_fraction.clamp_min(1e-8))
        return (area_weight * (ssim_loss + self.lambda_l1 * l1)).mean()



def _parse_training_annotations(raw, *, name: str, slices: int):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as error:
        raise ValueError(f"malformed organizer annotations in {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"organizer annotations must be an object: {name}")
    result = {index: [] for index in range(slices)}
    required = ("x", "y", "width", "height")
    for key, boxes in value.items():
        if (
            not isinstance(key, str)
            or not key.isascii()
            or not key.isdigit()
            or key != str(int(key))
            or not 0 <= int(key) < slices
            or not isinstance(boxes, list)
        ):
            raise ValueError(f"malformed organizer annotation slice in {name}")
        for box in boxes:
            if (
                not isinstance(box, dict)
                or any(field not in box or type(box[field]) is not int for field in required)
                or box["width"] <= 0
                or box["height"] <= 0
            ):
                raise ValueError(f"malformed organizer annotation box in {name}")
            x0, y0 = max(0, box["x"]), max(0, box["y"])
            x1 = min(384, box["x"] + box["width"])
            y1 = min(384, box["y"] + box["height"])
            if x1 - x0 >= 7 and y1 - y0 >= 7:
                result[int(key)].append(
                    (box["x"], box["y"], box["width"], box["height"])
                )
    return result


class AnnotationBoundDataset:
    """Attach organizer-only training boxes without changing reconstruction data."""

    def __init__(self, member):
        self.member = member
        self._boxes = {}
        self._max_boxes = 1
        slices = {4: 0, 8: 0}
        boxes = {4: 0, 8: 0}
        annotation_digest = hashlib.sha256()
        for record in member.records:
            with h5py.File(record.image, "r") as handle:
                if "annotations" not in handle.attrs:
                    raise ValueError(f"missing organizer annotations: {record.name}")
                raw_annotations = handle.attrs["annotations"]
                if isinstance(raw_annotations, bytes):
                    annotation_bytes = raw_annotations
                elif isinstance(raw_annotations, str):
                    annotation_bytes = raw_annotations.encode("utf-8")
                else:
                    annotation_bytes = canonical_bytes(raw_annotations)
                annotation_digest.update(record.name.encode("utf-8"))
                annotation_digest.update(b"\0")
                annotation_digest.update(annotation_bytes)
                annotation_digest.update(b"\0")
                parsed = _parse_training_annotations(
                    raw_annotations,
                    name=record.name,
                    slices=record.slices,
                )
            self._boxes[record.name] = parsed
            slices[int(record.acceleration)] += int(record.slices)
            boxes[int(record.acceleration)] += sum(len(value) for value in parsed.values())
            self._max_boxes = max(
                self._max_boxes,
                max((len(value) for value in parsed.values()), default=0),
            )
        self.annotation_counts = {"slices": slices, "boxes": boxes}
        self.annotation_inventory_sha256 = annotation_digest.hexdigest()

    def __getattr__(self, name):
        return getattr(self.member, name)

    def __len__(self):
        return len(self.member)

    def set_epoch(self, epoch):
        self.member.set_epoch(epoch)

    def __getitem__(self, index):
        sample = dict(self.member[index])
        active = self._boxes[str(sample["fname"])][int(sample["slice_num"])]
        tensor = torch.zeros((self._max_boxes, 4), dtype=torch.int64)
        if active:
            tensor[:len(active)] = torch.tensor(active, dtype=torch.int64)
        sample["score_boxes"] = tensor
        sample["score_box_count"] = torch.tensor(len(active), dtype=torch.int64)
        return sample


def derive_bbox_annotation_contract(datasets):
    slices = {4: 0, 8: 0}
    boxes = {4: 0, 8: 0}
    for member in datasets:
        for acceleration in (4, 8):
            slices[acceleration] += int(
                member.annotation_counts["slices"][acceleration]
            )
            boxes[acceleration] += int(
                member.annotation_counts["boxes"][acceleration]
            )
    expected = {
        "slices": {4: 2743, 8: 2699},
        "boxes": {4: 1524, 8: 1230},
    }
    if slices != expected["slices"] or boxes != expected["boxes"]:
        raise RuntimeError(
            f"organizer annotation inventory drifted: slices={slices}, boxes={boxes}"
        )
    acc4_boxes_per_slice = boxes[4] / slices[4]
    acc8_boxes_per_slice = (
        0.80 * boxes[8] / slices[8]
        + 0.20 * boxes[4] / slices[4]
    )
    weights = {4: 1.0 / acc4_boxes_per_slice, 8: 1.0 / acc8_boxes_per_slice}
    annotation_inventories = [
        member.annotation_inventory_sha256 for member in datasets
    ]
    return {
        "schema": "organizer-train-val-official384-bbox-cell-weighting-v2",
        "source": "organizer_train_plus_val_h5_annotations_training_only",
        "source_coordinate_frame": [384, 384],
        "training_tensor_alignment": "test_part_center_crop_then_zero_pad_v1",
        "bbox_ssim_frame": [384, 384],
        "slices": {str(key): value for key, value in slices.items()},
        "accepted_boxes": {str(key): value for key, value in boxes.items()},
        "bbox_weight_by_acceleration": {
            str(key): value for key, value in weights.items()
        },
        "acc8_real_fraction": 0.80,
        "acc8_virtual_acc4_fraction": 0.20,
        "annotation_inventory_sha256": hashlib.sha256(
            canonical_bytes(annotation_inventories)
        ).hexdigest(),
        "inference_annotation_access": False,
    }


def derive_training_data_contract(datasets, annotation_contract):
    members = []
    for member in datasets:
        value = {
            "root": str(member.root.resolve()),
            "trusted_manifest": str(member.trusted_manifest.resolve()),
            "trusted_manifest_sha256": member.trusted_manifest_sha256,
            "file_shape_identity_inventory_sha256": member.inventory_sha256,
            "record_count": len(member.records),
            "base_example_count": member.base_example_count,
            "training_example_count": member.training_example_count,
            "virtual_acc8_example_count": member.virtual_acc8_example_count,
            "base_acceleration_counts": {
                str(key): int(count)
                for key, count in member.base_acceleration_counts.items()
            },
            "training_acceleration_counts": {
                str(key): int(count)
                for key, count in member.training_acceleration_counts.items()
            },
        }
        if hasattr(member, "annotation_inventory_sha256"):
            value["annotation_inventory_sha256"] = (
                member.annotation_inventory_sha256
            )
        members.append(value)
    contract = {
        "schema": "vessl-organizer-training-data-contract-v1",
        "members": members,
        "annotation_contract_sha256": (
            hashlib.sha256(canonical_bytes(annotation_contract)).hexdigest()
            if annotation_contract is not None
            else None
        ),
        "leaderboard_data_read": False,
    }
    contract["contract_sha256"] = hashlib.sha256(
        canonical_bytes(contract)
    ).hexdigest()
    return contract


class EvaluatorBBoxSSIM(torch.nn.Module):
    def __init__(self, win_size: int = 7):
        super().__init__()
        self.win_size = int(win_size)
        self.cov_norm = self.win_size**2 / (self.win_size**2 - 1)
        self.register_buffer(
            "window",
            torch.ones(1, 1, self.win_size, self.win_size) / self.win_size**2,
        )

    def forward(self, x, y, data_range):
        x = x.reshape(1, 1, *x.shape)
        y = y.reshape(1, 1, *y.shape)
        ux = F.conv2d(x, self.window)
        uy = F.conv2d(y, self.window)
        uxx = F.conv2d(x * x, self.window)
        uyy = F.conv2d(y * y, self.window)
        uxy = F.conv2d(x * y, self.window)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        return (
            ((2 * ux * uy + c1) * (2 * vxy + c2))
            / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
        )[0, 0]


def center_crop_or_zero_pad(
    value: torch.Tensor,
    target_shape: tuple[int, int] = (384, 384),
) -> torch.Tensor:
    """Match the official test_part crop/pad transform on trailing dimensions."""
    target_height, target_width = map(int, target_shape)
    height, width = map(int, value.shape[-2:])
    if height > target_height:
        top = (height - target_height) // 2
        value = value[..., top:top + target_height, :]
        height = target_height
    if width > target_width:
        left = (width - target_width) // 2
        value = value[..., :, left:left + target_width]
        width = target_width
    pad_height = target_height - height
    pad_width = target_width - width
    if pad_height or pad_width:
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        value = F.pad(value, (left, right, top, bottom))
    if tuple(value.shape[-2:]) != (target_height, target_width):
        raise RuntimeError("official bbox-frame alignment failed")
    return value


class WinnerForegroundPlusBBox05(torch.nn.Module):
    """R17 foreground loss plus a small exact evaluator bbox-cell term."""

    def __init__(self, annotation_contract):
        super().__init__()
        self.foreground = WinnerForegroundSSIML1SqrtArea()
        self.bbox_ssim = EvaluatorBBoxSSIM()
        self.weights = {
            int(key): float(value)
            for key, value in annotation_contract[
                "bbox_weight_by_acceleration"
            ].items()
        }

    def forward(
        self,
        output,
        target,
        data_range,
        foreground,
        boxes,
        box_count,
        acceleration,
    ):
        if output.shape[0] != 1:
            raise RuntimeError("bbox-aligned NAF_S requires batch size one")
        base_loss = self.foreground(output, target, data_range, foreground)
        bbox_output = center_crop_or_zero_pad(output)
        bbox_target = center_crop_or_zero_pad(target)
        bbox_losses = []
        count = int(box_count.reshape(-1)[0].item())
        for coordinates in boxes[0, :count].tolist():
            x, y, width, height = map(int, coordinates)
            x0, y0 = max(0, x), max(0, y)
            x1 = min(384, x + width)
            y1 = min(384, y + height)
            if x1 - x0 < 7 or y1 - y0 < 7:
                continue
            bbox_losses.append(
                1.0
                - self.bbox_ssim(
                    bbox_output[0, y0:y1, x0:x1],
                    bbox_target[0, y0:y1, x0:x1],
                    data_range.reshape(-1)[0],
                ).mean()
            )
        if not bbox_losses:
            return base_loss
        route = int(acceleration.reshape(-1)[0].item())
        bbox_loss = self.weights[route] * torch.stack(bbox_losses).sum()
        return base_loss + BBOX_LOSS_COEFFICIENT * bbox_loss


def validate_base(
    path: Path,
    expected_sha256: str,
) -> tuple[dict, str, int, int]:
    resolved = path.resolve()
    if not resolved.is_relative_to(VESSL_RESULT_ROOT):
        raise RuntimeError("base checkpoint is outside /root/result")
    observed_sha256 = sha256(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError("base checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    model_config = config.get("model") if isinstance(config, dict) else None
    rung = str(checkpoint.get("rung", ""))
    num_cascades = (
        int(model_config.get("num_cascades", -1))
        if isinstance(model_config, dict)
        else -1
    )
    n_history = (
        int(model_config.get("n_history", -1))
        if isinstance(model_config, dict)
        else -1
    )
    counts = REGISTERED_ARCHITECTURES.get(
        (rung, num_cascades, n_history)
    )
    if (
        checkpoint.get("format_version") != 2
        or checkpoint.get("model_family") != "promptmr-plus-reduced"
        or counts is None
        or checkpoint.get("scratch") is not True
        or checkpoint.get("external_learned_state") is not False
        or not isinstance(checkpoint.get("model"), dict)
        or not isinstance(config, dict)
        or config.get("scratch") is not True
        or config.get("external_learned_state") is not False
        or config.get("source_commit") != PINNED_SOURCE_COMMIT
        or config.get("train_acceleration") != "all"
        or int(config.get("parameter_count", -1)) != counts[0]
        or int(config.get("trainable_parameter_count", counts[1]))
        != counts[1]
    ):
        raise RuntimeError("base is not a registered VESSL-scratch generalist")
    return checkpoint, rung, num_cascades, n_history



def validate_routed_branch(
    path: Path,
    expected_sha256: str,
    *,
    acceleration: int,
    generalist_sha256: str,
) -> dict:
    resolved = path.resolve()
    if not resolved.is_relative_to(VESSL_RESULT_ROOT):
        raise RuntimeError("routed branch is outside /root/result")
    if sha256(path) != expected_sha256:
        raise RuntimeError("routed branch SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    expected_step = {4: 4_672, 8: 1_158}[int(acceleration)]
    expected_horizon = {4: 35_040, 8: 2_315}[int(acceleration)]
    expected_loss = {
        4: "exact_upstream_ssim",
        8: "r10_image_masked_ssim_valid_windows_mean",
    }[int(acceleration)]
    if (
        checkpoint.get("format_version") != 2
        or checkpoint.get("model_family") != "promptmr-plus-reduced"
        or checkpoint.get("scratch") is not True
        or checkpoint.get("external_learned_state") is not False
        or not isinstance(checkpoint.get("model"), dict)
        or not isinstance(config, dict)
        or config.get("train_acceleration") != f"acc{acceleration}"
        or config.get("vessl_parent_checkpoint_sha256")
        != generalist_sha256
        or config.get("lineage_origin") != "VESSL_SCRATCH_MODEL_ONLY"
        or config.get("specialist_loss_family") != expected_loss
        or config.get("lr_schedule") != "specialist_warmup1_cosine"
        or config.get("scheduler", {}).get("total_steps")
        != expected_horizon
        or int(config.get("stop_after_optimizer_steps", -1))
        != expected_step
        or not valid_specialist_training_data_contract(
            config.get("training_dataset")
        )
        or int(checkpoint.get("global_optimizer_step", -1))
        != expected_step
    ):
        raise RuntimeError(
            f"ACC{acceleration} routed branch contract mismatch"
        )
    return checkpoint


class RealFirstBalancedSampler(Sampler):
    """Fixed 50/50 acceleration budget with 80/20 real/virtual ACC8."""

    schema = "equal_acc_real_acc8_real80_virtual20_v1"

    def __init__(self, members, seed: int, samples_per_group: int):
        self.seed = int(seed)
        self.samples_per_group = int(samples_per_group)
        self.epoch = 0
        self.cursor = 0
        self.acc4_real = []
        self.acc8_real = []
        self.acc8_virtual = []
        offset = 0
        for member in members:
            variants = tuple(member._acc4_pair_variant)
            for local_index, (acceleration, variant) in enumerate(
                zip(member.accelerations, variants)
            ):
                global_index = offset + local_index
                if int(acceleration) == 4:
                    self.acc4_real.append(global_index)
                elif int(variant) < 0:
                    self.acc8_real.append(global_index)
                else:
                    self.acc8_virtual.append(global_index)
            offset += len(member)
        if not self.acc4_real or not self.acc8_real or not self.acc8_virtual:
            raise RuntimeError("R13 sampler pools are incomplete")
        self.acc8_real_count = round(self.samples_per_group * 0.80)
        self.acc8_virtual_count = (
            self.samples_per_group - self.acc8_real_count
        )

    def __len__(self):
        return 2 * self.samples_per_group

    @staticmethod
    def _draw(pool, count, generator):
        result = []
        while len(result) < count:
            order = torch.randperm(len(pool), generator=generator).tolist()
            result.extend(pool[index] for index in order)
        return result[:count]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        acc4 = self._draw(
            self.acc4_real, self.samples_per_group, generator
        )
        acc8 = self._draw(
            self.acc8_real, self.acc8_real_count, generator
        ) + self._draw(
            self.acc8_virtual, self.acc8_virtual_count, generator
        )
        order = torch.randperm(len(acc8), generator=generator).tolist()
        acc8 = [acc8[index] for index in order]
        result = []
        first_acc4 = self.epoch % 2 == 0
        for index in range(self.samples_per_group):
            pair = (acc4[index], acc8[index])
            result.extend(pair if first_acc4 else reversed(pair))
        for position in range(self.cursor, len(result)):
            self.cursor = position + 1
            yield result[position]
        self.epoch += 1
        self.cursor = 0

    def state_dict(self):
        return {
            "schema": self.schema,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "samples_per_group": self.samples_per_group,
            "acc4_real_pool": len(self.acc4_real),
            "acc8_real_pool": len(self.acc8_real),
            "acc8_virtual_pool": len(self.acc8_virtual),
            "acc8_real_count": self.acc8_real_count,
            "acc8_virtual_count": self.acc8_virtual_count,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in (
            "schema",
            "seed",
            "samples_per_group",
            "acc4_real_pool",
            "acc8_real_pool",
            "acc8_virtual_pool",
            "acc8_real_count",
            "acc8_virtual_count",
        ):
            if state.get(key) != expected[key]:
                raise RuntimeError(f"R13 sampler recovery mismatch: {key}")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        if not 0 <= self.cursor <= len(self):
            raise RuntimeError("R13 sampler cursor is invalid")

def scheduled_lr(
    completed_steps: int,
    *,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int | None = None,
) -> float:
    warmup_steps = (
        max(1, total_steps // 10)
        if warmup_steps is None
        else max(1, int(warmup_steps))
    )
    if completed_steps < warmup_steps:
        return peak_lr * float(completed_steps + 1) / warmup_steps
    denominator = max(1, total_steps - warmup_steps - 1)
    progress = min(
        1.0,
        float(completed_steps - warmup_steps) / denominator,
    )
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def recovery_payload(
    *,
    args: argparse.Namespace,
    refiner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: BalancedAccelerationSampler,
    optimizer_step: int,
    base_sha256: str,
    module_sha256: str,
    training_data_contract: dict[str, object],
    annotation_contract: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "schema": "vessl-post-refiner-recovery-v1",
        "state": "RECOVERABLE",
        "variant": args.variant,
        "mask_conditioned": args.mask_conditioned,
        "mask_conditioning": (
            MASK_CONDITIONING_CONTRACT if args.mask_conditioned else None
        ),
        "views": list(args.views),
        "epochs": args.epochs,
        "terminal_optimizer_steps": args.optimizer_steps,
        "lr_horizon_optimizer_steps": args.lr_horizon_optimizer_steps,
        "loss_family": args.loss_family,
        "bbox_loss_coefficient": (
            BBOX_LOSS_COEFFICIENT
            if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
            else None
        ),
        "organizer_annotations_used_for_training": (
            args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
        ),
        "annotation_contract": annotation_contract,
        "training_data_contract": training_data_contract,
        "inference_annotation_access": False,
        "training_data": (
            "organizer_train_plus_val_final"
            if args.extra_train_root is not None
            else "organizer_train_only"
        ),
        "optimizer_step": optimizer_step,
        "post_refiner_state": refiner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all(),
        "base_checkpoint_sha256": base_sha256,
        "training_base_route": "exact_acceleration_specialist_before_shared_naf_s_v1",
        "routed_branch_sha256": {
            "acc4": args.acc4_checkpoint_sha256,
            "acc8": args.acc8_checkpoint_sha256,
        },
        "sampler_policy": "equal_acc_real_acc8_real80_virtual20_v1",
        "post_refiner_module_sha256": module_sha256,
        "peak_lr": args.peak_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "leaderboard_data_read": False,
        "external_learned_state_imported": False,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce GTX 1080":
        raise RuntimeError("final post-refiner must train on the GTX 1080")
    checkpoint, rung, num_cascades, n_history = validate_base(
        args.base_checkpoint,
        args.base_checkpoint_sha256,
    )
    base_epoch = int(checkpoint.get("epoch", -1))
    if args.epochs == 10 and (
        args.variant != "NAF_S"
        or base_epoch != 51
        or args.optimizer_steps != 44_736
    ):
        raise RuntimeError(
            "the asymmetric terminal refiner must be NAF_S after sealed E51"
        )
    if args.epochs == 20 and (
        args.variant != "NAF_S"
        or base_epoch not in {50, 51}
        or args.optimizer_steps != 91_141
    ):
        raise RuntimeError("the legacy deadline refiner base is invalid")
    if args.epochs == 21 and (
        args.variant != "NAF_S"
        or base_epoch != 49
        or args.optimizer_steps != 91_231
        or args.lr_horizon_optimizer_steps != 93_567
    ):
        raise RuntimeError(
            "the R25 terminal refiner must be NAF_S after sealed E49"
        )
    if args.epochs == 15 and (
        args.variant != "NAF_S" or base_epoch != 40
    ):
        raise RuntimeError(
            "the E41-E55 final post-refiner must be NAF_S after a sealed E40 base"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module_path = Path(
        __import__(
            "utils.learning.promptmr_post_refiner",
            fromlist=["__file__"],
        ).__file__
    )
    module_sha256 = sha256(module_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    branch_states = {
        4: validate_routed_branch(
            args.acc4_checkpoint,
            args.acc4_checkpoint_sha256,
            acceleration=4,
            generalist_sha256=args.base_checkpoint_sha256,
        ),
        8: validate_routed_branch(
            args.acc8_checkpoint,
            args.acc8_checkpoint_sha256,
            acceleration=8,
            generalist_sha256=args.base_checkpoint_sha256,
        ),
    }
    del checkpoint
    refiner = make_refiner(
        args.variant,
        mask_conditioned=args.mask_conditioned,
        input_mode=args.input_mode,
    ).to(device).train()
    wrappers = {}
    for acceleration, branch_state in branch_states.items():
        base = PromptMRPlusAdapter(
            build_rung_model(rung, n_history, num_cascades)
        )
        base.load_state_dict(branch_state["model"], strict=True)
        del branch_state
        base.to(device).eval()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        wrappers[acceleration] = BaseOnceRefinerTTA(
            base,
            refiner,
            views=args.views,
        )
    del branch_states
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
    )
    train_dataset_raw = PromptMRProductionDataset(
        args.train_root,
        args.trusted_data_manifest,
        num_adj_slices=1,
        mraugment="off",
        legal_mask_family=True,
        legal_mask_seed=args.seed,
        acc4_to_acc8_pair_augmentation=True,
    )
    train_dataset = (
        AnnotationBoundDataset(train_dataset_raw)
        if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
        else train_dataset_raw
    )
    datasets = [train_dataset]
    if args.extra_train_root is not None:
        extra_dataset_raw = PromptMRProductionDataset(
            args.extra_train_root,
            args.extra_trusted_data_manifest,
            num_adj_slices=1,
            mraugment="off",
            legal_mask_family=True,
            legal_mask_seed=args.seed,
            acc4_to_acc8_pair_augmentation=True,
        )
        datasets.append(
            AnnotationBoundDataset(extra_dataset_raw)
            if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
            else extra_dataset_raw
        )
    dataset = (
        train_dataset
        if len(datasets) == 1
        else ConcatDataset(datasets)
    )
    for member in datasets:
        member.set_epoch(0)
    accelerations = tuple(
        acceleration
        for member in datasets
        for acceleration in member.accelerations
    )
    annotation_contract = (
        derive_bbox_annotation_contract(datasets)
        if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
        else None
    )
    training_data_contract = derive_training_data_contract(
        datasets,
        annotation_contract,
    )
    loss_module = (
        WinnerForegroundPlusBBox05(annotation_contract)
        if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
        else WinnerForegroundSSIML1SqrtArea()
        if args.loss_family == "winner_foreground_ssim_l1_sqrt_area_v1"
        else UpstreamSSIMObjective()
    ).to(device)
    sampler = RealFirstBalancedSampler(
        datasets,
        args.seed,
        samples_per_group=train_dataset.base_samples_per_group,
    )
    steps_per_epoch = len(sampler)
    total_steps = (
        int(args.optimizer_steps)
        if args.optimizer_steps is not None
        else steps_per_epoch * args.epochs
    )
    lr_horizon_steps = (
        int(args.lr_horizon_optimizer_steps)
        if args.lr_horizon_optimizer_steps is not None
        else total_steps
    )
    if lr_horizon_steps < total_steps:
        raise RuntimeError("post-refiner LR horizon precedes its stop step")
    warmup_steps = max(1, steps_per_epoch // 10)
    optimizer_step = 0
    if args.resume is not None:
        recovery = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=True,
        )
        if (
            recovery.get("schema") != "vessl-post-refiner-recovery-v1"
            or recovery.get("variant") != args.variant
            or bool(recovery.get("mask_conditioned", False))
            is not args.mask_conditioned
            or recovery.get("views") != list(args.views)
            or int(recovery.get("epochs", -1)) != args.epochs
            or recovery.get("terminal_optimizer_steps")
            != args.optimizer_steps
            or recovery.get("lr_horizon_optimizer_steps")
            != args.lr_horizon_optimizer_steps
            or recovery.get("loss_family") != args.loss_family
            or recovery.get("bbox_loss_coefficient") != (
                BBOX_LOSS_COEFFICIENT
                if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
                else None
            )
            or recovery.get("organizer_annotations_used_for_training")
            is not (args.loss_family == BBOX_ALIGNED_LOSS_FAMILY)
            or recovery.get("annotation_contract") != annotation_contract
            or recovery.get("training_data_contract")
            != training_data_contract
            or recovery.get("inference_annotation_access") is not False
            or recovery.get("training_data")
            != (
                "organizer_train_plus_val_final"
                if args.extra_train_root is not None
                else "organizer_train_only"
            )
            or recovery.get("base_checkpoint_sha256")
            != args.base_checkpoint_sha256
            or recovery.get("training_base_route")
            != "exact_acceleration_specialist_before_shared_naf_s_v1"
            or recovery.get("routed_branch_sha256") != {
                "acc4": args.acc4_checkpoint_sha256,
                "acc8": args.acc8_checkpoint_sha256,
            }
            or recovery.get("sampler_policy")
            != "equal_acc_real_acc8_real80_virtual20_v1"
            or recovery.get("post_refiner_module_sha256") != module_sha256
            or float(recovery.get("peak_lr", -1)) != args.peak_lr
            or float(recovery.get("weight_decay", -1))
            != args.weight_decay
            or int(recovery.get("seed", -1)) != args.seed
            or recovery.get("external_learned_state_imported") is not False
        ):
            raise RuntimeError("post-refiner recovery contract mismatch")
        refiner.load_state_dict(
            recovery["post_refiner_state"],
            strict=True,
        )
        optimizer.load_state_dict(recovery["optimizer"])
        sampler.load_state_dict(recovery["sampler"])
        optimizer_step = int(recovery["optimizer_step"])
        torch.set_rng_state(recovery["torch_cpu_rng"])
        torch.cuda.set_rng_state_all(recovery["torch_cuda_rng"])
    elif any(args.output_dir.iterdir()):
        raise RuntimeError("non-empty post-refiner output requires --resume")

    recovery_path = args.output_dir / "recovery.pt"
    if args.resume is None:
        atomic_torch(
            recovery_path,
            recovery_payload(
                args=args,
                refiner=refiner,
                optimizer=optimizer,
                sampler=sampler,
                optimizer_step=optimizer_step,
                base_sha256=args.base_checkpoint_sha256,
                module_sha256=module_sha256,
                training_data_contract=training_data_contract,
                annotation_contract=annotation_contract,
            ),
        )
    atomic_json(
        args.output_dir / "launch.json",
        {
            "schema": "vessl-post-refiner-launch-v1",
            "state": "RUNNING",
            "variant": args.variant,
            "mask_conditioned": args.mask_conditioned,
            "mask_conditioning": (
                MASK_CONDITIONING_CONTRACT if args.mask_conditioned else None
            ),
            "input_mode": args.input_mode,
            "zero_filled_definition": (
                ZERO_FILLED_DEFINITION
                if args.input_mode == INPUT_MODE_ZF_CONTEXT
                else None
            ),
            "normalization": "shared_detached_reconstruction_amax",
            "spatial_match": "center_crop_then_zero_pad",
            "views": list(args.views),
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "terminal_optimizer_steps": args.optimizer_steps,
            "lr_horizon_optimizer_steps": lr_horizon_steps,
            "loss_family": args.loss_family,
            "bbox_loss_coefficient": (
                BBOX_LOSS_COEFFICIENT
                if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
                else None
            ),
            "organizer_annotations_used_for_training": (
                args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
            ),
            "annotation_contract": annotation_contract,
            "training_data_contract": training_data_contract,
            "inference_annotation_access": False,
            "loss_lambda_l1": (
                0.1
                if args.loss_family
                in WINNER_LOSS_FAMILIES
                else None
            ),
            "sqrt_area_weighting": (
                args.loss_family
                in WINNER_LOSS_FAMILIES
            ),
            "training_dataset": {
                "roots": [str(member.root.resolve()) for member in datasets],
                "base_example_count": sum(
                    member.base_example_count for member in datasets
                ),
                "training_example_count": sum(
                    member.training_example_count for member in datasets
                ),
                "virtual_acc8_example_count": (
                    sum(
                        member.virtual_acc8_example_count
                        for member in datasets
                    )
                ),
                "base_acceleration_counts": {
                    str(acceleration): sum(
                        member.base_acceleration_counts.get(acceleration, 0)
                        for member in datasets
                    )
                    for acceleration in (4, 8)
                },
                "training_acceleration_counts": (
                    {
                        str(acceleration): sum(
                            member.training_acceleration_counts.get(
                                acceleration, 0
                            )
                            for member in datasets
                        )
                        for acceleration in (4, 8)
                    }
                ),
                "epoch_sample_budget_per_group": (
                    train_dataset.base_samples_per_group
                ),
                "sampler_uses_fixed_base_epoch_budget": True,
            },
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "base_checkpoint_sha256": args.base_checkpoint_sha256,
            "training_base_route": "exact_acceleration_specialist_before_shared_naf_s_v1",
            "routed_branch_checkpoint": {
                "acc4": str(args.acc4_checkpoint.resolve()),
                "acc8": str(args.acc8_checkpoint.resolve()),
            },
            "routed_branch_sha256": {
                "acc4": args.acc4_checkpoint_sha256,
                "acc8": args.acc8_checkpoint_sha256,
            },
            "sampler_policy": "equal_acc_real_acc8_real80_virtual20_v1",
            "sampler_acc8_real_fraction": 0.80,
            "post_refiner_module_sha256": module_sha256,
            "leaderboard_data_read": False,
            "training_data": (
                "organizer_train_plus_val_final"
                if args.extra_train_root is not None
                else "organizer_train_only"
            ),
            "validation_used_for_checkpoint_selection": False,
            "external_learned_state_imported": False,
            "started_unix": time.time(),
        },
    )
    losses: list[float] = []
    started = time.time()
    while optimizer_step < total_steps:
        sampler_epoch_before = int(sampler.epoch)
        for member in datasets:
            member.set_epoch(sampler_epoch_before)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )
        made_progress = False
        for batch in loader:
            made_progress = True
            optimizer.zero_grad(set_to_none=True)
            lr = scheduled_lr(
                optimizer_step,
                total_steps=lr_horizon_steps,
                peak_lr=args.peak_lr,
                warmup_steps=warmup_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            prepared = PromptMRInput(
                batch["masked_kspace"].to(device),
                batch["mask"].to(device),
                batch["num_low_frequencies"].to(device),
                int(batch["acceleration"][0]),
            )
            acceleration = int(batch["acceleration"][0])
            output = wrappers[acceleration](
                prepared,
                crop_size=tuple(map(int, batch["target"].shape[-2:])),
                use_checkpoint=False,
                compute_sens_per_coil=True,
            )
            target = batch["target"].to(device)
            maximum = batch["max_value"].to(device)
            foreground = batch["foreground"].to(device)
            loss = (
                loss_module(
                    output,
                    target,
                    maximum,
                    foreground,
                    batch["score_boxes"],
                    batch["score_box_count"],
                    batch["acceleration"],
                )
                if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
                else loss_module(output, target, maximum, foreground)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite post-refiner loss")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in refiner.parameters()
                if parameter.grad is not None
            ]
            if (
                not gradients
                or not all(
                    torch.isfinite(value).all() for value in gradients
                )
                or not any(
                    bool(torch.count_nonzero(value))
                    for value in gradients
                )
            ):
                raise RuntimeError("post-refiner gradient contract failed")
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
            optimizer_step += 1
            losses.append(float(loss.detach().cpu()))
            if optimizer_step % 50 == 0 or optimizer_step == total_steps:
                atomic_json(
                    args.output_dir / "status.json",
                    {
                        "schema": "vessl-post-refiner-status-v1",
                        "state": "TRAINING",
                        "epoch": min(
                            args.epochs,
                            (optimizer_step - 1)
                            // steps_per_epoch
                            + 1,
                        ),
                        "optimizer_step": optimizer_step,
                        "steps_per_epoch": steps_per_epoch,
                        "total_steps": total_steps,
                        "loss": losses[-1],
                        "mean_loss_since_resume": (
                            sum(losses) / len(losses)
                        ),
                        "lr": lr,
                        "updated_unix": time.time(),
                    },
                )
            if optimizer_step % 100 == 0:
                atomic_torch(
                    recovery_path,
                    recovery_payload(
                        args=args,
                        refiner=refiner,
                        optimizer=optimizer,
                        sampler=sampler,
                        optimizer_step=optimizer_step,
                        base_sha256=args.base_checkpoint_sha256,
                        module_sha256=module_sha256,
                        training_data_contract=training_data_contract,
                        annotation_contract=annotation_contract,
                    ),
                )
            if optimizer_step >= total_steps:
                break
        if (
            not made_progress
            and int(sampler.epoch) == sampler_epoch_before
            and optimizer_step < total_steps
        ):
            raise RuntimeError("post-refiner sampler made no progress")

    if optimizer_step != total_steps:
        raise RuntimeError(
            "post-refiner horizon ended at an unexpected step"
        )
    final_checkpoint = args.output_dir / (
        f"checkpoint-step-{optimizer_step:09d}.pt"
        if args.optimizer_steps is not None
        else f"epoch-{args.epochs:02d}.pt"
    )
    atomic_torch(
        final_checkpoint,
        {
            "schema": "vessl-base-once-post-refiner-v1",
            "state": "COMPLETE",
            "variant": args.variant,
            "role": "main_output_post_refiner",
            "views": list(args.views),
            "views_batched": True,
            "parameter_count": registered_parameter_count(
                args.variant,
                args.mask_conditioned,
            ),
            "mask_conditioned": args.mask_conditioned,
            "mask_conditioning": (
                MASK_CONDITIONING_CONTRACT if args.mask_conditioned else None
            ),
            "input_mode": args.input_mode,
            "zero_filled_definition": (
                ZERO_FILLED_DEFINITION
                if args.input_mode == INPUT_MODE_ZF_CONTEXT
                else None
            ),
            "normalization": "shared_detached_reconstruction_amax",
            "spatial_match": "center_crop_then_zero_pad",
            "epoch": args.epochs,
            "parent_epoch": base_epoch,
            "late_branch_epochs": (
                [50, 70]
                if args.epochs == 21 and base_epoch == 49
                else
                [72, 81]
                if args.epochs == 10
                else [52, 71]
                if args.epochs == 20 and base_epoch == 51
                else [51, 70]
                if args.epochs == 20
                else [41, 55]
                if args.epochs == 15
                else None
            ),
            "epoch_budget": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "partial_terminal_epoch": total_steps % steps_per_epoch != 0,
            "completed_epoch_equivalent": total_steps / steps_per_epoch,
            "trainable_parameter_scope": (
                (
                    "naf_s_plus_mask_conditioner"
                    if args.mask_conditioned
                    else "naf_s_only"
                )
                if args.epochs in {10, 15, 20, 21}
                else None
            ),
            "frozen_parameter_scope": (
                "main_c10_e49_all_parameters"
                if args.epochs == 21 and base_epoch == 49
                else "main_c10_e51_all_parameters"
                if args.epochs == 10
                or (args.epochs == 20 and base_epoch == 51)
                else "main_c10_e50_all_parameters"
                if args.epochs == 20
                else "main_c10_e40_all_parameters"
                if args.epochs == 15
                else None
            ),
            "main_parameters_updated": False,
            "optimizer_step": optimizer_step,
            "lr_horizon_optimizer_steps": lr_horizon_steps,
            "post_refiner_state": refiner.state_dict(),
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "base_checkpoint_sha256": args.base_checkpoint_sha256,
            "training_base_route": "exact_acceleration_specialist_before_shared_naf_s_v1",
            "routed_branch_checkpoint": {
                "acc4": str(args.acc4_checkpoint.resolve()),
                "acc8": str(args.acc8_checkpoint.resolve()),
            },
            "routed_branch_sha256": {
                "acc4": args.acc4_checkpoint_sha256,
                "acc8": args.acc8_checkpoint_sha256,
            },
            "sampler_policy": "equal_acc_real_acc8_real80_virtual20_v1",
            "sampler_acc8_real_fraction": 0.80,
            "post_refiner_module_sha256": module_sha256,
            "maximum_residual_fraction": 0.05,
            "peak_lr": args.peak_lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "loss_family": args.loss_family,
            "bbox_loss_coefficient": (
                BBOX_LOSS_COEFFICIENT
                if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
                else None
            ),
            "organizer_annotations_used_for_training": (
                args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
            ),
            "annotation_contract": annotation_contract,
            "training_data_contract": training_data_contract,
            "inference_annotation_access": False,
            "loss_lambda_l1": (
                0.1
                if args.loss_family
                in WINNER_LOSS_FAMILIES
                else None
            ),
            "sqrt_area_weighting": (
                args.loss_family
                in WINNER_LOSS_FAMILIES
            ),
            "training_data": (
                "organizer_train_plus_val_final"
                if args.extra_train_root is not None
                else "organizer_train_only"
            ),
            "validation_used_for_checkpoint_selection": False,
            "leaderboard_data_read": False,
            "checkpoint_selection_influence": False,
            "trained_on_vessl": True,
            "external_learned_state_imported": False,
        },
    )
    final_sha256 = sha256(final_checkpoint)
    atomic_json(
        args.output_dir / "receipt.json",
        {
            "schema": "vessl-post-refiner-training-receipt-v1",
            "state": "PASS",
            "variant": args.variant,
            "role": "main_output_post_refiner",
            "views": list(args.views),
            "views_batched": True,
            "parameter_count": registered_parameter_count(
                args.variant,
                args.mask_conditioned,
            ),
            "mask_conditioned": args.mask_conditioned,
            "mask_conditioning": (
                MASK_CONDITIONING_CONTRACT if args.mask_conditioned else None
            ),
            "input_mode": args.input_mode,
            "zero_filled_definition": (
                ZERO_FILLED_DEFINITION
                if args.input_mode == INPUT_MODE_ZF_CONTEXT
                else None
            ),
            "normalization": "shared_detached_reconstruction_amax",
            "spatial_match": "center_crop_then_zero_pad",
            "epochs": args.epochs,
            "parent_epoch": base_epoch,
            "late_branch_epochs": (
                [50, 70]
                if args.epochs == 21 and base_epoch == 49
                else
                [72, 81]
                if args.epochs == 10
                else [52, 71]
                if args.epochs == 20 and base_epoch == 51
                else [51, 70]
                if args.epochs == 20
                else [41, 55]
                if args.epochs == 15
                else None
            ),
            "steps_per_epoch": steps_per_epoch,
            "partial_terminal_epoch": total_steps % steps_per_epoch != 0,
            "completed_epoch_equivalent": total_steps / steps_per_epoch,
            "optimizer_steps": optimizer_step,
            "lr_horizon_optimizer_steps": lr_horizon_steps,
            "loss_family": args.loss_family,
            "bbox_loss_coefficient": (
                BBOX_LOSS_COEFFICIENT
                if args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
                else None
            ),
            "organizer_annotations_used_for_training": (
                args.loss_family == BBOX_ALIGNED_LOSS_FAMILY
            ),
            "annotation_contract": annotation_contract,
            "training_data_contract": training_data_contract,
            "inference_annotation_access": False,
            "loss_lambda_l1": (
                0.1
                if args.loss_family
                in WINNER_LOSS_FAMILIES
                else None
            ),
            "sqrt_area_weighting": (
                args.loss_family
                in WINNER_LOSS_FAMILIES
            ),
            "training_data": (
                "organizer_train_plus_val_final"
                if args.extra_train_root is not None
                else "organizer_train_only"
            ),
            "validation_used_for_checkpoint_selection": False,
            "mean_loss": sum(losses) / len(losses),
            "minimum_loss": min(losses),
            "maximum_loss": max(losses),
            "seconds": time.time() - started,
            "checkpoint": str(final_checkpoint),
            "checkpoint_sha256": final_sha256,
            "base_checkpoint_sha256": args.base_checkpoint_sha256,
            "training_base_route": "exact_acceleration_specialist_before_shared_naf_s_v1",
            "routed_branch_sha256": {
                "acc4": args.acc4_checkpoint_sha256,
                "acc8": args.acc8_checkpoint_sha256,
            },
            "sampler_policy": "equal_acc_real_acc8_real80_virtual20_v1",
            "sampler_acc8_real_fraction": 0.80,
            "post_refiner_module_sha256": module_sha256,
            "leaderboard_data_read": False,
            "external_learned_state_imported": False,
            "completed_unix": time.time(),
        },
    )
    recovery_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
