#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the R30 neighbor-ZF amendment."""

from __future__ import annotations

import ast
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "R30_CPU_PREFLIGHT.json"
INPUT_MODE = "recon_zero_filled_residual_neighbor_zf"
ZERO_FILLED_DEFINITION = (
    "rss(fftshift(ifft2(ifftshift(masked_kspace),norm=ortho)))"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def embedded_contract(schema: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": schema,
        "root": "/root/Data/train",
        "leaderboard_data_read": False,
    }
    value["contract_sha256"] = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return value


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def install_router_imports():
    utils = sys.modules.setdefault("utils", ModuleType("utils"))
    learning = sys.modules.setdefault("utils.learning", ModuleType("utils.learning"))
    utils.learning = learning
    conditioning = ModuleType("utils.learning.promptmr_mask_conditioning")
    conditioning.validate_profile = lambda value: value
    sys.modules[conditioning.__name__] = conditioning
    legal = load(
        "utils.learning.promptmr_legal_mask",
        ROOT / "promptmr_legal_mask.py",
    )
    mask_router = load(
        "utils.learning.promptmr_mask_router",
        ROOT / "promptmr_mask_router.py",
    )
    router = load(
        "r30_promptmr_router",
        ROOT / "promptmr_router.py",
    )
    return legal, mask_router, router


def component(source: str) -> dict[str, object]:
    return {
        "rung": "R2",
        "num_cascades": 10,
        "n_history": 0,
        "scratch": True,
        "external_learned_state": False,
        "trained_on_vessl": True,
        "source_checkpoint_sha256": source,
        "model": {},
    }


def routed_checkpoint() -> dict[str, object]:
    training_contract = embedded_contract(
        "vessl-organizer-training-data-contract-v1"
    )
    return {
        "schema": "vessl-acceleration-routed-promptmr-v2",
        "all_components_trained_on_vessl": True,
        "tta_views": ["identity"],
        "tta_views_by_acceleration": {
            "acc4": ["identity"],
            "acc8": ["identity", "flip_lr"],
        },
        "routing_contract": {
            "schema": "promptmr-legal-mask-routing-contract-v2",
            "generalist_component": "generalist",
            "unknown_or_mismatch": "generalist",
            "public_frequency_weighting": False,
        },
        "components": {
            "generalist": component("a" * 64),
            "acc4": component("b" * 64),
            "acc8": component("c" * 64),
        },
        "post_refiner": {
            "enabled": True,
            "role": "main_output_post_refiner",
            "variant": "NAF_S",
            "views": ["identity", "flip_lr"],
            "views_batched": True,
            "base_forward_count_per_slice": 1,
            "refiner_forward_count_per_slice": 1,
            "mask_conditioned": False,
            "epoch": 21,
            "parent_epoch": 49,
            "late_branch_epochs": [50, 70],
            "trainable_parameter_scope": "naf_s_plus_adjacent_zf_stem",
            "frozen_parameter_scope": "main_c10_e49_all_parameters",
            "main_parameters_updated": False,
            "optimizer_steps": 88_895,
            "lr_horizon_optimizer_steps": 93_567,
            "steps_per_epoch": 4_672,
            "partial_terminal_epoch": True,
            "training_data": "organizer_train_plus_val_final",
            "loss_family": (
                "winner_foreground_ssim_l1_sqrt_area_plus_"
                "official384_bbox05_v2"
            ),
            "loss_lambda_l1": 0.1,
            "sqrt_area_weighting": True,
            "bbox_loss_coefficient": 0.5,
            "organizer_annotations_used_for_training": True,
            "annotation_contract": {
                "schema": "organizer-train-val-official384-bbox-cell-weighting-v2",
                "source_coordinate_frame": [384, 384],
                "training_tensor_alignment": "test_part_center_crop_then_zero_pad_v1",
            },
            "inference_annotation_access": False,
            "validation_used_for_checkpoint_selection": False,
            "trained_on_vessl": True,
            "external_learned_state_imported": False,
            "training_data_contract": training_contract,
            "post_refiner_state": {},
            "parameter_count": 73_489,
            "input_mode": INPUT_MODE,
            "zero_filled_definition": ZERO_FILLED_DEFINITION,
            "normalization": "shared_detached_reconstruction_amax",
            "spatial_match": "center_crop_then_zero_pad",
            "adjacent_slice_context": {
                "count": 3,
                "positions": ["previous", "current", "next"],
                "boundary_policy": "replicate_nearest_slice",
                "source": "same_volume_masked_kspace_only",
            },
        },
    }


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise RuntimeError(f"function not found: {path}:{name}")


def compile_function(path: Path, name: str, namespace: dict[str, object]):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(selected) != 1:
        raise RuntimeError(f"exactly one function required: {path}:{name}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


if torch.cuda.is_initialized():
    raise RuntimeError("CUDA was initialized before the CPU-only preflight")

post = load("r30_promptmr_post_refiner", ROOT / "promptmr_post_refiner.py")
torch.manual_seed(430)
directional = post.NAFResidualImageRefiner(
    variant="NAF_S",
    input_mode=post.INPUT_MODE_DIRECTIONAL,
)
torch.manual_seed(430)
zf_context = post.NAFResidualImageRefiner(
    variant="NAF_S",
    input_mode=post.INPUT_MODE_ZF_CONTEXT,
)
torch.manual_seed(430)
neighbor_zf = post.NAFResidualImageRefiner(
    variant="NAF_S",
    input_mode=post.INPUT_MODE_NEIGHBOR_ZF,
)
directional_count = sum(parameter.numel() for parameter in directional.parameters())
zf_count = sum(parameter.numel() for parameter in zf_context.parameters())
neighbor_zf_count = sum(parameter.numel() for parameter in neighbor_zf.parameters())
if directional_count != 72_625 or zf_count != 72_625:
    raise RuntimeError("NAF_S parameter-count contract drifted")
if state_sha256(directional) != state_sha256(zf_context):
    raise RuntimeError("R30 changed NAF_S initialization or state shape")
if neighbor_zf_count != 73_489:
    raise RuntimeError("neighbor-ZF NAF_S parameter-count contract drifted")
if bool(torch.count_nonzero(neighbor_zf.head.weight)) or bool(
    torch.count_nonzero(neighbor_zf.head.bias)
):
    raise RuntimeError("neighbor-ZF NAF_S output is not zero initialized")

generator = torch.Generator(device="cpu").manual_seed(430)
paired = torch.randn(1, 5, 24, 20, 2, generator=generator)
observed_zf = post.zero_filled_rss(paired)
complex_kspace = torch.view_as_complex(paired.contiguous())
oracle_image = torch.fft.fftshift(
    torch.fft.ifft2(
        torch.fft.ifftshift(complex_kspace, dim=(-2, -1)),
        dim=(-2, -1),
        norm="ortho",
    ),
    dim=(-2, -1),
)
oracle_zf = torch.sqrt(
    (oracle_image.real.float().square() + oracle_image.imag.float().square())
    .sum(dim=1)
    .clamp_min(0.0)
)
zf_oracle_max_abs = float((observed_zf - oracle_zf).abs().max().item())
if zf_oracle_max_abs > 1e-6:
    raise RuntimeError("ZF FFT/RSS oracle mismatch")
if post.center_crop_or_zero_pad(observed_zf, (18, 24)).shape[-2:] != (18, 24):
    raise RuntimeError("ZF crop/pad contract failed")


class DummyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.last_route = 4

    def forward(self, prepared, **kwargs):
        self.calls += 1
        return torch.ones(1, 18, 24, dtype=torch.float32)


base = DummyBase()
wrapper = post.BaseOnceRefinerTTA(
    base,
    neighbor_zf,
    views=("identity", "flip_lr"),
)
output = wrapper(
    SimpleNamespace(kspace=paired, acceleration=4),
    neighbor_masked_kspace=(paired * 0.75, paired * 1.25),
)
if base.calls != 1 or output.shape != (1, 18, 24):
    raise RuntimeError("base-once neighbor-ZF wrapper contract failed")
if not bool(torch.isfinite(output).all().item()):
    raise RuntimeError("neighbor-ZF wrapper output is nonfinite")
if not torch.equal(output, torch.ones_like(output)):
    raise RuntimeError("zero-initialized neighbor-ZF wrapper changed base output")

legal, mask_router, router = install_router_imports()
for acceleration in (4, 8):
    key, receipt = mask_router.select_component(
        legal.legal_cartesian_mask(368, acceleration, 1),
        generalist_component="generalist",
    )
    if key != f"acc{acceleration}" or receipt.get("specialist_activated") is not True:
        raise RuntimeError(f"exact ACC{acceleration} dispatch failed")
unknown = legal.legal_cartesian_mask(368, 8, 1).copy()
unknown[3] = ~unknown[3]
unknown_key, unknown_receipt = mask_router.select_component(
    unknown,
    generalist_component="generalist",
)
if unknown_key != "generalist" or unknown_receipt.get("specialist_activated") is not False:
    raise RuntimeError("unknown-mask generalist dispatch failed")

valid = routed_checkpoint()
contract = router.RoutedCheckpointContract.validate(valid)
if contract.post_refiner_enabled is not True:
    raise RuntimeError("actual shipped-router R30 contract did not validate")
for field, bad_value in (
    ("optimizer_steps", 93_567),
    ("lr_horizon_optimizer_steps", 88_895),
    ("input_mode", "recon_horizontal_vertical"),
):
    mutated = json.loads(json.dumps(valid))
    mutated["post_refiner"][field] = bad_value
    try:
        router.RoutedCheckpointContract.validate(mutated)
    except RuntimeError:
        pass
    else:
        raise RuntimeError(f"router accepted invalid post-refiner {field}")

trainer_source = (ROOT / "vessl_train_post_refiner.py").read_text(encoding="utf-8")
for required in (
    "expected_step = {4: 7_008, 8: 1_158}",
    "expected_horizon = {4: 35_040, 8: 2_315}",
    "args.input_mode != INPUT_MODE_NEIGHBOR_ZF",
):
    if required not in trainer_source:
        raise RuntimeError(f"trainer contract repair is absent: {required}")
parse_post = compile_function(
    ROOT / "vessl_train_post_refiner.py",
    "parse_args",
    {
        "argparse": argparse,
        "Path": Path,
        "REGISTERED_REFINERS": {
            "NAF_S": 72_625,
            "PLAIN_168K": 168_049,
            "NAF_M": 248_641,
            "NAF_L": 815_713,
        },
        "ALLOWED_VIEWS": ("identity", "flip_lr", "flip_ud", "rot180"),
        "INPUT_MODE_DIRECTIONAL": "recon_horizontal_vertical",
        "INPUT_MODE_ZF_CONTEXT": "recon_zero_filled_residual",
        "INPUT_MODE_NEIGHBOR_ZF": INPUT_MODE,
        "BBOX_ALIGNED_LOSS_FAMILY": (
            "winner_foreground_ssim_l1_sqrt_area_plus_"
            "official384_bbox05_v2"
        ),
    },
)
saved_argv = sys.argv
try:
    sys.argv = [
        "vessl_train_post_refiner.py",
        "--base-checkpoint", "/root/result/e49.pt",
        "--base-checkpoint-sha256", "a" * 64,
        "--acc4-checkpoint", "/root/result/acc4.pt",
        "--acc4-checkpoint-sha256", "b" * 64,
        "--acc8-checkpoint", "/root/result/acc8.pt",
        "--acc8-checkpoint-sha256", "c" * 64,
        "--variant", "NAF_S",
        "--input-mode", INPUT_MODE,
        "--views", "identity", "flip_lr",
        "--epochs", "21",
        "--optimizer-steps", "88895",
        "--lr-horizon-optimizer-steps", "93567",
        "--output-dir", "/root/result/naf-r30",
        "--train-root", "/root/Data/train",
        "--trusted-data-manifest", "/root/result/provenance.json",
        "--extra-train-root", "/root/Data/val",
        "--extra-trusted-data-manifest", "/root/result/provenance.json",
        "--loss-family", (
            "winner_foreground_ssim_l1_sqrt_area_plus_"
            "official384_bbox05_v2"
        ),
        "--peak-lr", "0.0001",
        "--weight-decay", "0.0001",
        "--seed", "430",
    ]
    parsed_post = parse_post()
finally:
    sys.argv = saved_argv
if (
    parsed_post.input_mode != INPUT_MODE
    or parsed_post.optimizer_steps != 88_895
    or parsed_post.lr_horizon_optimizer_steps != 93_567
):
    raise RuntimeError("R30 post-refiner command/parser contract failed")

controller_train_source = function_source(ROOT / "controller.py", "train_post_refiner")
if '"--input-mode"' not in controller_train_source:
    raise RuntimeError("controller does not pass the sealed R30 input mode")

test_part_path = ROOT / "test_part.py"
prep_source = function_source(test_part_path, "prep_volume")
recon_source = function_source(test_part_path, "recon_slice")
if any(token in prep_source for token in ("_ifft2c(", "_fft2c(", "select_component(")):
    raise RuntimeError("prep_volume performs timed reconstruction or routing work")
for required in (
    "select_component(",
    "_views_for_route(model, route)",
    "infer_acceleration_from_mask(ctx[\"mask\"])",
    "neighbor_masked_kspace",
    "previous_index",
    "next_index",
):
    if required not in recon_source:
        raise RuntimeError(f"timed exact-route invariant is absent: {required}")
views_source = function_source(test_part_path, "_views_for_route")
if 'if route == "generalist":' not in views_source or 'return ("identity",)' not in views_source:
    raise RuntimeError("unknown/generalist outer-TTA identity contract is absent")

timed_recon_slice = compile_function(
    test_part_path,
    "recon_slice",
    {
        "torch": torch,
        "NAFResidualImageRefiner": post.NAFResidualImageRefiner,
        "INPUT_MODE_NEIGHBOR_ZF": post.INPUT_MODE_NEIGHBOR_ZF,
        "select_component": lambda *args, **kwargs: ("acc4", {}),
        "infer_acceleration_from_mask": lambda mask: 4,
        "_views_for_route": lambda model, route: ("identity",),
        "_transform_tta_input": lambda kspace, mask, view: (kspace, mask),
        "PromptMRInput": lambda **kwargs: SimpleNamespace(**kwargs),
        "_restore_tta_output": lambda output, view: output,
        "_center_crop_or_zero_pad": lambda output, shape: output,
    },
)


class DummyRoutedBase:
    generalist_component = "generalist"

    def __init__(self) -> None:
        self.released = False

    def release_active(self) -> None:
        self.released = True


class DummyTimedModel:
    def __init__(self) -> None:
        self.base_model = DummyRoutedBase()
        self.refiner = neighbor_zf
        self.neighbor_calls = []

    def __call__(self, prepared, **kwargs):
        self.neighbor_calls.append(kwargs.get("neighbor_masked_kspace"))
        return torch.ones(1, 18, 24, dtype=torch.float32)


timed_model = DummyTimedModel()
timed_volume = torch.randn(3, 5, 18, 24, 2, generator=generator)
timed_context = {
    "volume": timed_volume,
    "mask": torch.ones(1, 1, 1, 24, 1, dtype=torch.bool),
    "num_low_frequencies": torch.tensor([-1], dtype=torch.int64),
    "crop_size": (18, 24),
    "num_slices": 3,
}
for slice_index in (1, 0, 2):
    result = timed_recon_slice(timed_model, timed_context, slice_index)
    if result.shape != (18, 24) or not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("timed neighbor-ZF recon_slice output contract failed")
center_neighbors, first_neighbors, last_neighbors = timed_model.neighbor_calls
if (
    not torch.equal(center_neighbors[0], timed_volume[0].unsqueeze(0))
    or not torch.equal(center_neighbors[1], timed_volume[2].unsqueeze(0))
    or not torch.equal(first_neighbors[0], timed_volume[0].unsqueeze(0))
    or not torch.equal(first_neighbors[1], timed_volume[1].unsqueeze(0))
    or not torch.equal(last_neighbors[0], timed_volume[1].unsqueeze(0))
    or not torch.equal(last_neighbors[1], timed_volume[2].unsqueeze(0))
    or timed_model.base_model.released is not True
):
    raise RuntimeError("timed neighbor-ZF boundary/dispatch contract failed")

if torch.cuda.is_initialized():
    raise RuntimeError("CPU-only R30 preflight initialized CUDA")

receipt = {
    "schema": "vessl-r30-neighbor-zf-cpu-preflight-v1",
    "state": "PASS",
    "cpu_only": True,
    "cuda_initialized": False,
    "candidate_count": 1,
    "fallback_registered": False,
    "post_e49_optimizer_steps": 7_008 + 1_158 + 88_895,
    "naf_s_parameter_count": neighbor_zf_count,
    "zero_initialized_output_identity": True,
    "adjacent_slice_count": 3,
    "adjacent_boundary_policy": "replicate_nearest_slice",
    "zf_oracle_max_abs": zf_oracle_max_abs,
    "exact_acc4_dispatch": True,
    "exact_acc8_dispatch": True,
    "unknown_mask_generalist_dispatch": True,
    "unknown_mask_outer_tta": ["identity"],
    "actual_shipped_router_validation": True,
    "timed_neighbor_recon_slice_executed": True,
    "post_refiner_command_parser": True,
    "contract_mutations_rejected": [
        "optimizer_steps",
        "lr_horizon_optimizer_steps",
        "input_mode",
    ],
    "active_generalist_process_touched": False,
    "source_hashes": {
        path.name: sha256(path)
        for path in (
            ROOT / "controller.py",
            ROOT / "train.py",
            ROOT / "promptmr_production.py",
            ROOT / "vessl_train_post_refiner.py",
            ROOT / "vessl_build_routed_promptmr_checkpoint.py",
            ROOT / "promptmr_post_refiner.py",
            ROOT / "promptmr_router.py",
            ROOT / "promptmr_mask_router.py",
            ROOT / "promptmr_legal_mask.py",
            ROOT / "test_part.py",
        )
    },
}
OUTPUT.write_bytes(canonical_bytes(receipt))
print(json.dumps(receipt, sort_keys=True))
