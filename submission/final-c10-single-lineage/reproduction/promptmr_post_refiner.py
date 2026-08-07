"""Small image-domain NAF refiner used after the routed main reconstruction.

The main PromptMR reconstruction is evaluated once.  R30 derives
physics-correct zero-filled RSS images from the current and immediately
adjacent masked k-space slices and applies one shared, batched residual
refiner to the registered views.  Optional conditioning consumes only the
mask-derived route exposed by the already-validated acceleration router.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


ALLOWED_VIEWS = ("identity", "flip_lr", "flip_ud", "rot180")
DEFAULT_VIEWS = ALLOWED_VIEWS
EXPECTED_REFINER_PARAMETERS = 168_049
NAF_REFINER_VARIANTS = {
    "NAF_S": {"channels": 48, "blocks": 4, "parameter_count": 72_625},
    "NAF_M": {"channels": 64, "blocks": 8, "parameter_count": 248_641},
    "NAF_L": {"channels": 96, "blocks": 12, "parameter_count": 815_713},
}
MASK_CONDITION_ROUTES = {"unknown": 0, "acc4": 4, "acc8": 8}
MASK_CONDITION_ROUTE_COUNT = 3
MASK_CONDITION_STRENGTH = 0.1
INPUT_MODE_DIRECTIONAL = "recon_horizontal_vertical"
INPUT_MODE_ZF_CONTEXT = "recon_zero_filled_residual"
INPUT_MODE_NEIGHBOR_ZF = "recon_zero_filled_residual_neighbor_zf"
NAF_INPUT_MODES = (
    INPUT_MODE_DIRECTIONAL,
    INPUT_MODE_ZF_CONTEXT,
    INPUT_MODE_NEIGHBOR_ZF,
)
ZERO_FILLED_DEFINITION = (
    "rss(fftshift(ifft2(ifftshift(masked_kspace),norm=ortho)))"
)


def _validate_views(views: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(views)
    if (
        not normalized
        or normalized[0] != "identity"
        or len(set(normalized)) != len(normalized)
        or any(view not in ALLOWED_VIEWS for view in normalized)
    ):
        raise ValueError("post-refiner views must be unique and identity-first")
    return normalized


def _view(image: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return image
    if name == "flip_lr":
        return torch.flip(image, dims=(-1,))
    if name == "flip_ud":
        return torch.flip(image, dims=(-2,))
    if name == "rot180":
        return torch.flip(image, dims=(-2, -1))
    raise ValueError(f"unsupported post-refiner view: {name}")


def _directional_features(image: torch.Tensor) -> torch.Tensor:
    horizontal = F.pad(
        image[..., 1:] - image[..., :-1], (0, 1, 0, 0), mode="replicate"
    )
    vertical = F.pad(
        image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1), mode="replicate"
    )
    return torch.cat((image, horizontal, vertical), dim=1)


def zero_filled_rss(masked_kspace: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal IFFT followed by coil RSS in FP32."""
    if (
        not torch.is_tensor(masked_kspace)
        or masked_kspace.ndim != 5
        or masked_kspace.shape[-1] != 2
        or not masked_kspace.is_floating_point()
    ):
        raise ValueError("masked k-space must be floating [B,C,H,W,2]")
    complex_kspace = torch.view_as_complex(masked_kspace.contiguous())
    image = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(complex_kspace, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
    squared = image.real.float().square() + image.imag.float().square()
    result = torch.sqrt(squared.sum(dim=1).clamp_min(0.0))
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("zero-filled reconstruction is nonfinite")
    return result


def center_crop_or_zero_pad(
    value: torch.Tensor,
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """Match the frozen reconstruction frame without target information."""
    target_height, target_width = map(int, target_shape)
    height, width = map(int, value.shape[-2:])
    if height > target_height:
        top = (height - target_height) // 2
        value = value[..., top : top + target_height, :]
        height = target_height
    if width > target_width:
        left = (width - target_width) // 2
        value = value[..., :, left : left + target_width]
        width = target_width
    pad_height = target_height - height
    pad_width = target_width - width
    if pad_height < 0 or pad_width < 0:
        raise RuntimeError("zero-filled spatial matching failed")
    if pad_height or pad_width:
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        value = F.pad(value, (left, right, top, bottom))
    if tuple(value.shape[-2:]) != (target_height, target_width):
        raise RuntimeError("zero-filled context has the wrong shape")
    return value


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.second = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.second(F.silu(self.first(value)))
        return value + 0.2 * residual


class ResidualImageRefiner(nn.Module):
    """Registered 168k-parameter residual head retained for old packages."""

    def __init__(
        self,
        *,
        channels: int = 48,
        blocks: int = 4,
        maximum_residual_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        if channels != 48 or blocks != 4 or maximum_residual_fraction != 0.05:
            raise ValueError("the registered residual architecture is byte-bound")
        self.maximum_residual_fraction = float(maximum_residual_fraction)
        self.stem = nn.Conv2d(
            3, channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.blocks = nn.ModuleList(_ResidualBlock(channels) for _ in range(blocks))
        self.head = nn.Conv2d(
            channels, 1, kernel_size=3, padding=1, padding_mode="reflect"
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        observed = sum(parameter.numel() for parameter in self.parameters())
        if observed != EXPECTED_REFINER_PARAMETERS:
            raise RuntimeError(f"post-refiner parameter drift: {observed}")

    def forward(self, normalized_image: torch.Tensor) -> torch.Tensor:
        _validate_input(normalized_image)
        value = F.silu(self.stem(_directional_features(normalized_image)))
        for block in self.blocks:
            value = block(value)
        return self.maximum_residual_fraction * torch.tanh(self.head(value))


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.permute(0, 2, 3, 1)
        value = F.layer_norm(value, (value.shape[-1],), self.weight, self.bias, 1e-6)
        return value.permute(0, 3, 1, 2)


class _SimpleGate(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second


class _NAFBlock(nn.Module):
    """Compact NAFNet-style block with zero-initialized residual scales."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.norm1 = _LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, expanded, kernel_size=1)
        self.depthwise = nn.Conv2d(
            expanded,
            expanded,
            kernel_size=3,
            padding=1,
            groups=expanded,
            padding_mode="reflect",
        )
        self.gate1 = _SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, kernel_size=1)
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm2 = _LayerNorm2d(channels)
        self.ffn_expand = nn.Conv2d(channels, expanded, kernel_size=1)
        self.gate2 = _SimpleGate()
        self.ffn_project = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branch = self.expand(self.norm1(value))
        branch = self.gate1(self.depthwise(branch))
        branch = branch * self.channel_attention(branch)
        first = value + self.beta * self.project(branch)
        branch = self.gate2(self.ffn_expand(self.norm2(first)))
        return first + self.gamma * self.ffn_project(branch)


def _validate_input(value: torch.Tensor) -> None:
    if (
        not torch.is_tensor(value)
        or value.ndim != 4
        or value.shape[1] != 1
        or not value.is_floating_point()
    ):
        raise ValueError("refiner input must be floating BCHW with C=1")


class NAFResidualImageRefiner(nn.Module):
    """Small bounded NAFNet-style residual corrector."""

    def __init__(
        self,
        *,
        variant: str = "NAF_M",
        maximum_residual_fraction: float = 0.05,
        mask_conditioned: bool = False,
        input_mode: str = INPUT_MODE_DIRECTIONAL,
    ) -> None:
        super().__init__()
        if variant not in NAF_REFINER_VARIANTS:
            raise ValueError(f"unsupported NAF refiner variant: {variant}")
        if maximum_residual_fraction != 0.05:
            raise ValueError("the screened residual bound is byte-bound")
        if input_mode not in NAF_INPUT_MODES:
            raise ValueError(f"unsupported NAF input mode: {input_mode}")
        spec = NAF_REFINER_VARIANTS[variant]
        channels = int(spec["channels"])
        blocks = int(spec["blocks"])
        self.variant = variant
        self.mask_conditioned = bool(mask_conditioned)
        self.input_mode = str(input_mode)
        self.channels = channels
        self.maximum_residual_fraction = float(maximum_residual_fraction)
        input_channels = 5 if self.input_mode == INPUT_MODE_NEIGHBOR_ZF else 3
        self.input_channels = input_channels
        self.stem = nn.Conv2d(
            input_channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.blocks = nn.ModuleList(_NAFBlock(channels) for _ in range(blocks))
        self.mask_conditioner = (
            nn.Embedding(
                MASK_CONDITION_ROUTE_COUNT,
                (blocks + 1) * 2 * channels,
            )
            if self.mask_conditioned
            else None
        )
        if self.mask_conditioner is not None:
            nn.init.zeros_(self.mask_conditioner.weight)
        self.head = nn.Conv2d(
            channels, 1, kernel_size=3, padding=1, padding_mode="reflect"
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        observed = sum(parameter.numel() for parameter in self.parameters())
        expected = (
            int(spec["parameter_count"])
            + (input_channels - 3) * channels * 3 * 3
            + (
                MASK_CONDITION_ROUTE_COUNT * (blocks + 1) * 2 * channels
                if self.mask_conditioned
                else 0
            )
        )
        if observed != expected:
            raise RuntimeError(f"{variant} parameter drift: {observed} != {expected}")

    def _film_parameters(
        self,
        acceleration_condition: int | torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.mask_conditioned:
            if acceleration_condition is not None:
                raise ValueError(
                    "unconditioned NAF refiner cannot receive a route condition"
                )
            return None
        if acceleration_condition is None:
            acceleration = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )
        elif torch.is_tensor(acceleration_condition):
            acceleration = acceleration_condition.to(
                device=device,
                dtype=torch.long,
            ).reshape(-1)
            if acceleration.numel() == 1:
                acceleration = acceleration.expand(batch_size)
            elif acceleration.numel() != batch_size:
                raise ValueError("route-condition batch size mismatch")
        else:
            acceleration = torch.full(
                (batch_size,),
                int(acceleration_condition),
                dtype=torch.long,
                device=device,
            )
        valid = (acceleration == 0) | (acceleration == 4) | (acceleration == 8)
        if not bool(valid.all().item()):
            raise ValueError("route condition must be unknown(0), acc4, or acc8")
        route_ids = torch.where(
            acceleration == 4,
            torch.ones_like(acceleration),
            torch.where(
                acceleration == 8,
                torch.full_like(acceleration, 2),
                torch.zeros_like(acceleration),
            ),
        )
        return self.mask_conditioner(route_ids).reshape(
            batch_size,
            len(self.blocks) + 1,
            2,
            self.channels,
            1,
            1,
        )

    @staticmethod
    def _apply_film(value: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        scale = parameters[:, 0]
        shift = parameters[:, 1]
        return value * (
            1.0 + MASK_CONDITION_STRENGTH * torch.tanh(scale)
        ) + MASK_CONDITION_STRENGTH * torch.tanh(shift)

    def forward(
        self,
        normalized_image: torch.Tensor,
        normalized_zero_filled: torch.Tensor | None = None,
        normalized_neighbor_zero_filled: torch.Tensor | None = None,
        acceleration_condition: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_input(normalized_image)
        if self.input_mode in (INPUT_MODE_ZF_CONTEXT, INPUT_MODE_NEIGHBOR_ZF):
            if normalized_zero_filled is None:
                raise ValueError("ZF-context NAF requires zero-filled input")
            _validate_input(normalized_zero_filled)
            if normalized_zero_filled.shape != normalized_image.shape:
                raise ValueError("reconstruction and zero-filled shapes differ")
            features = [
                normalized_image,
                normalized_zero_filled,
                normalized_image - normalized_zero_filled,
            ]
            if self.input_mode == INPUT_MODE_NEIGHBOR_ZF:
                if (
                    not torch.is_tensor(normalized_neighbor_zero_filled)
                    or normalized_neighbor_zero_filled.ndim != 4
                    or normalized_neighbor_zero_filled.shape[0]
                    != normalized_image.shape[0]
                    or normalized_neighbor_zero_filled.shape[1] != 2
                    or normalized_neighbor_zero_filled.shape[-2:]
                    != normalized_image.shape[-2:]
                    or not normalized_neighbor_zero_filled.is_floating_point()
                    or not bool(
                        torch.isfinite(normalized_neighbor_zero_filled).all().item()
                    )
                ):
                    raise ValueError(
                        "neighbor ZF context must be finite floating [B,2,H,W]"
                    )
                features.append(normalized_neighbor_zero_filled)
            elif normalized_neighbor_zero_filled is not None:
                raise ValueError("single-slice ZF NAF cannot receive neighbors")
            features = torch.cat(features, dim=1)
        else:
            if (
                normalized_zero_filled is not None
                or normalized_neighbor_zero_filled is not None
            ):
                raise ValueError("directional NAF cannot receive zero-filled input")
            features = _directional_features(normalized_image)
        film = self._film_parameters(
            acceleration_condition,
            batch_size=normalized_image.shape[0],
            device=normalized_image.device,
        )
        value = self.stem(features)
        if film is not None:
            value = self._apply_film(value, film[:, 0])
        for index, block in enumerate(self.blocks, start=1):
            if film is not None:
                value = self._apply_film(value, film[:, index])
            value = block(value)
        return self.maximum_residual_fraction * torch.tanh(self.head(value))


class BaseOnceRefinerTTA(nn.Module):
    """One main reconstruction followed by one batched shared refiner call."""

    def __init__(
        self,
        base_model: nn.Module,
        refiner: nn.Module,
        *,
        views: Sequence[str] = DEFAULT_VIEWS,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be torch.nn.Module")
        if not isinstance(refiner, (ResidualImageRefiner, NAFResidualImageRefiner)):
            raise TypeError("refiner must be a registered image refiner")
        self.base_model = base_model
        self.refiner = refiner
        self.views = _validate_views(views)
        self.base_forward_count_per_slice = 1
        self.refiner_forward_count_per_slice = 1
        self.refiner_view_count_per_slice = len(self.views)
        self.num_adj_slices = getattr(base_model, "num_adj_slices", 1)

    def release_active(self) -> None:
        release = getattr(self.base_model, "release_active", None)
        if release is not None:
            release()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        neighbor_masked_kspace = kwargs.pop("neighbor_masked_kspace", None)
        base = self.base_model(*args, **kwargs)
        if (
            not torch.is_tensor(base)
            or base.ndim != 3
            or base.shape[0] != 1
            or not base.is_floating_point()
            or not bool(torch.isfinite(base).all().item())
        ):
            raise ValueError("base reconstruction must be finite (1,H,W)")
        base_fp32 = base.to(dtype=torch.float32)
        scale = base_fp32.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        )
        normalized = (base_fp32 / scale).unsqueeze(1)
        viewed = torch.cat([_view(normalized, name) for name in self.views], dim=0)
        viewed_zero_filled = None
        viewed_neighbor_zero_filled = None
        if (
            isinstance(self.refiner, NAFResidualImageRefiner)
            and self.refiner.input_mode
            in (INPUT_MODE_ZF_CONTEXT, INPUT_MODE_NEIGHBOR_ZF)
        ):
            prepared = args[0] if args else None
            masked_kspace = getattr(prepared, "kspace", None)
            zero_filled = center_crop_or_zero_pad(
                zero_filled_rss(masked_kspace),
                tuple(base_fp32.shape[-2:]),
            )
            normalized_zero_filled = (zero_filled / scale).unsqueeze(1)
            viewed_zero_filled = torch.cat(
                [
                    _view(normalized_zero_filled, name)
                    for name in self.views
                ],
                dim=0,
            )
            if self.refiner.input_mode == INPUT_MODE_NEIGHBOR_ZF:
                if (
                    not isinstance(neighbor_masked_kspace, (tuple, list))
                    or len(neighbor_masked_kspace) != 2
                ):
                    raise ValueError(
                        "neighbor-ZF NAF requires previous and next masked k-space"
                    )
                neighbor_images = []
                for neighbor in neighbor_masked_kspace:
                    neighbor_images.append(
                        center_crop_or_zero_pad(
                            zero_filled_rss(neighbor),
                            tuple(base_fp32.shape[-2:]),
                        )
                    )
                normalized_neighbors = (
                    torch.stack(neighbor_images, dim=1) / scale.unsqueeze(1)
                )
                viewed_neighbor_zero_filled = torch.cat(
                    [
                        _view(normalized_neighbors, name)
                        for name in self.views
                    ],
                    dim=0,
                )
            elif neighbor_masked_kspace is not None:
                raise ValueError(
                    "single-slice ZF NAF cannot receive neighbor k-space"
                )
        if (
            isinstance(self.refiner, NAFResidualImageRefiner)
            and self.refiner.mask_conditioned
        ):
            route = getattr(self.base_model, "last_route", None)
            if route == "generalist":
                acceleration_condition = 0
            elif route in (4, 8):
                acceleration_condition = int(route)
            else:
                prepared = args[0] if args else None
                declared = getattr(prepared, "acceleration", 0)
                acceleration_condition = (
                    int(declared) if declared in (4, 8) else 0
                )
            residuals = self.refiner(
                viewed,
                normalized_zero_filled=viewed_zero_filled,
                normalized_neighbor_zero_filled=viewed_neighbor_zero_filled,
                acceleration_condition=acceleration_condition,
            )
        elif (
            isinstance(self.refiner, NAFResidualImageRefiner)
            and self.refiner.input_mode
            in (INPUT_MODE_ZF_CONTEXT, INPUT_MODE_NEIGHBOR_ZF)
        ):
            residuals = self.refiner(
                viewed,
                normalized_zero_filled=viewed_zero_filled,
                normalized_neighbor_zero_filled=viewed_neighbor_zero_filled,
            )
        else:
            residuals = self.refiner(viewed)
        if residuals.shape != viewed.shape or not bool(torch.isfinite(residuals).all().item()):
            raise ValueError("post-refiner output contract failed")
        restored = [
            _view(residuals[index : index + 1], name)
            for index, name in enumerate(self.views)
        ]
        averaged_residual = torch.stack(restored, dim=0).mean(dim=0)
        return (base_fp32 + averaged_residual.squeeze(1) * scale).clamp_min(0.0)


def build_contract(*, views: Sequence[str] = DEFAULT_VIEWS) -> dict:
    normalized = _validate_views(views)
    return {
        "schema": "base-once-post-refiner-tta-v1",
        "base_forward_count_per_slice": 1,
        "refiner_forward_count_per_slice": 1,
        "refiner_views_batched": True,
        "views": list(normalized),
        "aggregation": "inverse_view_fp32_mean",
        "refiner": {
            "channels": 48,
            "blocks": 4,
            "parameter_count": EXPECTED_REFINER_PARAMETERS,
            "maximum_residual_fraction": 0.05,
            "zero_initialized_output": True,
        },
        "official_reconstruction_path_required": True,
        "gtx1080_8192mib_admission_required": True,
        "vessl_scratch_lineage_required": True,
    }


def build_naf_contract(
    *,
    variant: str,
    views: Sequence[str] = DEFAULT_VIEWS,
    mask_conditioned: bool = False,
    input_mode: str = INPUT_MODE_DIRECTIONAL,
) -> dict:
    normalized = _validate_views(views)
    if variant not in NAF_REFINER_VARIANTS:
        raise ValueError(f"unsupported NAF refiner variant: {variant}")
    if input_mode not in NAF_INPUT_MODES:
        raise ValueError(f"unsupported NAF input mode: {input_mode}")
    spec = NAF_REFINER_VARIANTS[variant]
    conditioned_parameters = (
        MASK_CONDITION_ROUTE_COUNT
        * (int(spec["blocks"]) + 1)
        * 2
        * int(spec["channels"])
        if mask_conditioned
        else 0
    )
    context_parameters = (
        2 * int(spec["channels"]) * 3 * 3
        if input_mode == INPUT_MODE_NEIGHBOR_ZF
        else 0
    )
    return {
        "schema": "base-once-naf-refiner-tta-v1",
        "base_forward_count_per_slice": 1,
        "refiner_forward_count_per_slice": 1,
        "refiner_views_batched": True,
        "views": list(normalized),
        "aggregation": "inverse_view_fp32_mean",
        "refiner": {
            "family": "NAFNet_style",
            "variant": variant,
            **spec,
            "parameter_count": int(spec["parameter_count"])
            + context_parameters
            + conditioned_parameters,
            "input_channels": 5 if input_mode == INPUT_MODE_NEIGHBOR_ZF else 3,
            "maximum_residual_fraction": 0.05,
            "zero_initialized_output": True,
            "input_mode": input_mode,
            "zero_filled_definition": (
                ZERO_FILLED_DEFINITION
                if input_mode in (INPUT_MODE_ZF_CONTEXT, INPUT_MODE_NEIGHBOR_ZF)
                else None
            ),
            "neighbor_context": {
                "enabled": input_mode == INPUT_MODE_NEIGHBOR_ZF,
                "offsets": [-1, 1],
                "boundary": "replicate_nearest_slice",
                "source": "same_volume_masked_kspace_only",
            },
            "normalization": "shared_detached_reconstruction_amax",
            "spatial_match": "center_crop_then_zero_pad",
            "mask_conditioning": {
                "enabled": bool(mask_conditioned),
                "source": "input_kspace_mask_exact_route",
                "routes": MASK_CONDITION_ROUTES,
                "parameter_count": conditioned_parameters,
                "zero_initialized": True,
                "application": "stem_and_each_naf_block_film",
                "maximum_scale_shift": MASK_CONDITION_STRENGTH,
            },
        },
        "official_reconstruction_path_required": True,
        "gtx1080_8192mib_admission_required": True,
        "vessl_scratch_lineage_required": True,
    }
