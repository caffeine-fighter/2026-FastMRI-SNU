"""Final acceleration-routed PromptMR+ adapter for fixed recon_eval.py."""

from __future__ import annotations

import h5py
import math
import numpy as np
import torch
import torch.nn.functional as F

from utils.learning.promptmr_production import build_rung_model
try:
    from utils.learning.promptmr_mask_conditioning import (
        condition_component_adapter,
    )
except ModuleNotFoundError:
    from ops.vessl_promptmr_mask_conditioning import (
        condition_component_adapter,
    )
from utils.learning.promptmr_post_refiner import (
    BaseOnceRefinerTTA,
    INPUT_MODE_DIRECTIONAL,
    NAFResidualImageRefiner,
    ResidualImageRefiner,
)
from utils.learning.promptmr_mask_router import select_component
from utils.learning.promptmr_router import (
    AccelerationRoutedPromptMR,
    RoutedCheckpointContract,
    infer_acceleration_from_mask,
)
from utils.model.promptmr_plus_adapter import PromptMRInput, PromptMRPlusAdapter


INPUT_KIND = "kspace"
ORGANIZER_RECONSTRUCTION_SHAPE = (384, 384)


def _ifft2c(value: torch.Tensor) -> torch.Tensor:
    value = torch.fft.ifftshift(value, dim=(-2, -1))
    value = torch.fft.ifftn(value, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(value, dim=(-2, -1))


def _fft2c(value: torch.Tensor) -> torch.Tensor:
    value = torch.fft.ifftshift(value, dim=(-2, -1))
    value = torch.fft.fftn(value, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(value, dim=(-2, -1))


def _even_spatial_flip_kspace(
    complex_kspace: torch.Tensor,
    *,
    dimension: int,
) -> torch.Tensor:
    """Apply an exact even-length spatial flip directly in shifted k-space."""
    dimension = dimension % complex_kspace.ndim
    size = int(complex_kspace.shape[dimension])
    if size % 2:
        raise ValueError("fast spatial flip requires an even dimension")
    destination = torch.arange(size, device=complex_kspace.device)
    source = (-destination).remainder(size)
    unshifted_frequency = (destination + size // 2).remainder(size)
    phase = torch.exp(
        (2j * math.pi / size) * unshifted_frequency
    ).to(dtype=complex_kspace.dtype)
    phase_shape = [1] * complex_kspace.ndim
    phase_shape[dimension] = size
    return complex_kspace.index_select(dimension, source) * phase.reshape(
        phase_shape
    )


def _flip_lr_input(
    paired_kspace: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    complex_kspace = torch.view_as_complex(paired_kspace.contiguous())
    support = mask[..., 0].to(dtype=torch.bool)
    width = int(complex_kspace.shape[-1])
    if width % 2 == 0:
        # For an even-width, fft-shifted DFT, spatial ``torch.flip`` is
        # exactly K'[j] = exp(2πi*k/N) K[-j], where
        # k=(j+N/2) mod N.  This avoids a full IFFT+FFT pair per coil/slice
        # while matching the original physics-safe transform to floating
        # point roundoff (official widths are 368/372).
        transformed = _even_spatial_flip_kspace(
            complex_kspace, dimension=-1
        )
        destination = torch.arange(width, device=complex_kspace.device)
        source = (-destination).remainder(width)
        transformed_support = support.index_select(-1, source)
    else:
        # Preserve the general definition for unexpected odd widths.
        transformed = _fft2c(
            torch.flip(_ifft2c(complex_kspace), dims=(-1,))
        )
        expanded_support = support.expand_as(complex_kspace)
        indicator = expanded_support.to(dtype=complex_kspace.dtype)
        transformed_indicator = _fft2c(
            torch.flip(_ifft2c(indicator), dims=(-1,))
        ).abs()
        tolerance = (
            torch.finfo(transformed_indicator.dtype).eps
            * max(support.shape[-2:])
            * 16
        )
        transformed_support = transformed_indicator > tolerance
        first_row = transformed_support[:, :, :1, :]
        if not torch.equal(
            transformed_support,
            first_row.expand_as(transformed_support),
        ):
            raise RuntimeError(
                "flip_lr TTA no longer has a column-broadcast mask"
            )
        transformed_support = first_row
    transformed = transformed.masked_fill(~transformed_support, 0)
    transformed_mask = transformed_support.unsqueeze(-1)
    return torch.view_as_real(transformed), transformed_mask


def _flip_ud_input(
    paired_kspace: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    complex_kspace = torch.view_as_complex(paired_kspace.contiguous())
    support = mask[..., 0].to(dtype=torch.bool)
    height = int(complex_kspace.shape[-2])
    if height % 2 == 0:
        transformed = _even_spatial_flip_kspace(
            complex_kspace, dimension=-2
        )
    else:
        transformed = _fft2c(
            torch.flip(_ifft2c(complex_kspace), dims=(-2,))
        )
    # A vertical spatial flip only permutes the readout-frequency axis.
    # Cartesian column support therefore remains unchanged.
    transformed = transformed.masked_fill(~support, 0)
    return torch.view_as_real(transformed), mask.to(dtype=torch.bool)


def _transform_tta_input(
    paired_kspace: torch.Tensor,
    mask: torch.Tensor,
    view: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if view == "identity":
        return paired_kspace, mask
    if view == "flip_lr":
        return _flip_lr_input(paired_kspace, mask)
    if view == "flip_ud":
        return _flip_ud_input(paired_kspace, mask)
    if view == "rot180":
        lr_kspace, lr_mask = _flip_lr_input(paired_kspace, mask)
        return _flip_ud_input(lr_kspace, lr_mask)
    raise RuntimeError(f"unsupported routed TTA view: {view}")


def _restore_tta_output(output: torch.Tensor, view: str) -> torch.Tensor:
    if view == "identity":
        return output
    if view == "flip_lr":
        return torch.flip(output, dims=(-1,))
    if view == "flip_ud":
        return torch.flip(output, dims=(-2,))
    if view == "rot180":
        return torch.flip(output, dims=(-2, -1))
    raise RuntimeError(f"unsupported routed TTA view: {view}")


def _center_crop_or_zero_pad(
    output: torch.Tensor, target_shape: tuple[int, int]
) -> torch.Tensor:
    if output.ndim != 2:
        raise RuntimeError(
            f"expected 2-D reconstruction, got {tuple(output.shape)}"
        )
    target_height, target_width = map(int, target_shape)
    height, width = map(int, output.shape)
    if height > target_height:
        top = (height - target_height) // 2
        output = output[top : top + target_height]
        height = target_height
    if width > target_width:
        left = (width - target_width) // 2
        output = output[:, left : left + target_width]
        width = target_width
    pad_height = target_height - height
    pad_width = target_width - width
    if pad_height or pad_width:
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        output = F.pad(output, (left, right, top, bottom))
    if tuple(output.shape) != (target_height, target_width):
        raise RuntimeError("reconstruction alignment failed")
    return output


def load_model(args, device):
    checkpoint_path = args.exp_dir / "best_model.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    contract = RoutedCheckpointContract.validate(checkpoint)
    tta_views = tuple(checkpoint.get("tta_views", ["identity"]))
    branches = {}
    for component_key in contract.component_keys:
        component = checkpoint["components"][component_key]
        if (
            component["rung"] != contract.rung
            or int(component["num_cascades"]) != contract.num_cascades
            or int(component["n_history"]) != contract.n_history
        ):
            raise RuntimeError("routed architecture changed after validation")
        core = build_rung_model(
            contract.rung,
            contract.n_history,
            contract.num_cascades,
        )
        model = condition_component_adapter(
            PromptMRPlusAdapter(core),
            component,
        )
        model.load_state_dict(
            component["model"], strict=True
        )
        branches[component_key] = model
    model = AccelerationRoutedPromptMR(
        branches["acc4"],
        branches["acc8"],
        generalist_component=contract.generalist_component,
        generalist_model=branches.get("generalist"),
    )
    post_refiner = checkpoint.get("post_refiner", {"enabled": False})
    if contract.post_refiner_enabled:
        variant = str(post_refiner["variant"])
        refiner = (
            ResidualImageRefiner()
            if variant == "PLAIN_168K"
            else NAFResidualImageRefiner(
                variant=variant,
                mask_conditioned=bool(
                    post_refiner.get("mask_conditioned", False)
                ),
                input_mode=str(
                    post_refiner.get(
                        "input_mode",
                        INPUT_MODE_DIRECTIONAL,
                    )
                ),
            )
        )
        refiner.load_state_dict(
            post_refiner["post_refiner_state"],
            strict=True,
        )
        model = BaseOnceRefinerTTA(
            model,
            refiner,
            views=post_refiner["views"],
        )
    model.full_model_tta_views = tta_views
    model.tta_views_by_acceleration = (
        dict(contract.tta_views_by_acceleration)
        if contract.tta_views_by_acceleration is not None
        else None
    )
    del checkpoint
    # The router intentionally leaves both experts on CPU here and moves only
    # the mask-selected expert to ``device`` on the first recon_slice call.
    model.to(device=device).eval()
    return model


def _views_for_route(model, route: str) -> tuple[str, ...]:
    routed = getattr(model, "tta_views_by_acceleration", None)
    if routed is None:
        return tuple(model.full_model_tta_views)
    if route == "generalist":
        return ("identity",)
    acceleration = {"acc4": 4, "acc8": 8}.get(route)
    if acceleration is None:
        raise RuntimeError(f"unsupported exact mask route: {route!r}")
    try:
        return tuple(routed[acceleration])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"missing routed TTA contract for acceleration {acceleration!r}"
        ) from error


def prep_volume(image_path, kspace_path, device):
    """Only organizer-approved deterministic k-space input preparation.

    ``image_path`` is intentionally ignored.  Challenge image-file fields are
    evaluation metadata and must never influence reconstruction.
    """
    with h5py.File(kspace_path, "r") as handle:
        kspace = np.asarray(handle["kspace"], dtype=np.complex64)
        mask = np.asarray(handle["mask"]).astype(bool, copy=False)
    if kspace.ndim != 4 or mask.shape != (kspace.shape[-1],):
        raise RuntimeError("invalid leaderboard k-space/mask contract")

    volume = torch.from_numpy(
        np.stack((kspace.real, kspace.imag), axis=-1).astype(
            np.float32, copy=False
        )
    ).to(device=device)
    mask_tensor = torch.from_numpy(
        mask.reshape(1, 1, 1, -1, 1).copy()
    ).to(device=device)
    volume.mul_(mask_tensor)
    return {
        "volume": volume,
        "mask": mask_tensor,
        "num_low_frequencies": torch.full(
            (1,), -1, dtype=torch.int64, device=device
        ),
        "crop_size": ORGANIZER_RECONSTRUCTION_SHAPE,
        "num_slices": int(volume.shape[0]),
    }


@torch.inference_mode()
def recon_slice(model, ctx, s):
    base_kspace = ctx["volume"][s].unsqueeze(0)
    routed_base = getattr(model, "base_model", model)
    route, _ = select_component(
        ctx["mask"],
        declared_acceleration=None,
        generalist_component=routed_base.generalist_component,
    )
    declared_acceleration = (
        4
        if route == "acc4"
        else 8
        if route == "acc8"
        else infer_acceleration_from_mask(ctx["mask"])
    )
    outputs = []
    for view in _views_for_route(model, route):
        kspace, mask = _transform_tta_input(
            base_kspace, ctx["mask"], view
        )
        prepared = PromptMRInput(
            kspace=kspace,
            mask=mask,
            num_low_frequencies=ctx["num_low_frequencies"],
            acceleration=declared_acceleration,
        )
        output = model(
            prepared,
            crop_size=None,
            use_checkpoint=False,
            compute_sens_per_coil=True,
        )[0]
        output = _restore_tta_output(output, view)
        outputs.append(output.to(dtype=torch.float32))
    averaged = torch.stack(outputs, dim=0).mean(dim=0)
    result = _center_crop_or_zero_pad(averaged, ctx["crop_size"])
    if int(s) + 1 == int(ctx["num_slices"]):
        routed_base.release_active()
    return result
