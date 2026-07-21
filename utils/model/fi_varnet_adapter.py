"""Thin, integrity-checked import adapter for the pinned upstream FI-VarNet."""

import functools
import hashlib
import importlib.util
from numbers import Integral
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as torch_functional
from torch.utils.checkpoint import checkpoint


PINNED_UPSTREAM_ROOT = Path("/root/upstream-fastMRI-91f2df47")
PINNED_UPSTREAM_COMMIT = "91f2df4711adbb6d643df1810f234e4abcf5881b"
PINNED_FEATURE_VARNET_SHA256 = (
    "810bf9c18b6e81b38bfc7b3732a26e2b87dc146c907a9b8bbc2d63428ea45d99"
)
PINNED_LOSSES_SHA256 = (
    "73ebfe3bc2d9c72b04250cc5a8dc35f31b283496a9411ab92fb422eca59f57ad"
)
PINNED_TRANSFORMS_SHA256 = (
    "0eedd9b6762ea720bd8014a8cd0365a022e1e16de96293b609e45fad96fb65c2"
)
PINNED_LICENSE_SHA256 = (
    "52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85"
)
_MODULE_NAME = "_pinned_fastmri_91f2df47_feature_varnet"
_LOSS_MODULE_NAME = "_pinned_fastmri_91f2df47_losses"
_PINNED_FEATURE_SOURCE = (
    PINNED_UPSTREAM_ROOT / "fastmri_examples" / "feature_varnet" / "feature_varnet.py"
).resolve()
FI_DETERMINISTIC_REFLECT_PAD_CONTRACT = {
    "schema": "fi-varnet-reflect-padding-adapter-v2",
    "implementation": "utils.model.fi_varnet_adapter.deterministic_reflect_pad2d",
    "version": "1.0.0",
    "native_forward_exact": True,
    "state_dict_unchanged": True,
    "strict_deterministic_algorithms": True,
    "scope": "process-global-pinned-feature-varnet-module-only",
    "pinned_module_name": _MODULE_NAME,
    "pinned_module_origin": str(_PINNED_FEATURE_SOURCE),
    "pinned_feature_varnet_sha256": PINNED_FEATURE_VARNET_SHA256,
}


def deterministic_reflect_pad2d(tensor, pad):
    """Reflect-pad the final two dimensions without native reflection backward.

    This implementation is deliberately closed to the four-value, nonnegative
    2D padding contract used by the pinned FI-VarNet.  Width is reflected first,
    then height, exactly matching ``torch.nn.functional.pad(..., 'reflect')``.
    Its autograd graph contains only slices, flips, and concatenations.
    """
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        raise ValueError("Deterministic reflect_pad2d requires one 4D tensor")
    if not isinstance(pad, (tuple, list)) or len(pad) != 4:
        raise ValueError("Deterministic reflect_pad2d pad must contain four integers")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in pad):
        raise ValueError("Deterministic reflect_pad2d pad must contain four integers")
    left, right, top, bottom = (int(value) for value in pad)
    if min(left, right, top, bottom) < 0:
        raise ValueError("Deterministic reflect_pad2d padding must be nonnegative")
    height, width = tensor.shape[-2:]
    if left >= width or right >= width or top >= height or bottom >= height:
        raise ValueError("Deterministic reflect_pad2d padding must be smaller than input")

    horizontal = torch.cat(
        (
            tensor[..., 1 : left + 1].flip(-1),
            tensor,
            tensor[..., width - right - 1 : width - 1].flip(-1),
        ),
        dim=-1,
    )
    return torch.cat(
        (
            horizontal[..., 1 : top + 1, :].flip(-2),
            horizontal,
            horizontal[..., height - bottom - 1 : height - 1, :].flip(-2),
        ),
        dim=-2,
    )


class _PinnedFunctionalProxy:
    """Delegate F except for reflect pad in the one pinned upstream module."""

    __slots__ = ("_functional",)

    def __init__(self, functional):
        self._functional = functional

    def __getattr__(self, name):
        return getattr(self._functional, name)

    def pad(self, input, pad, mode="constant", value=None):
        if mode == "reflect":
            if value is not None:
                raise ValueError("Deterministic reflect_pad2d forbids a value argument")
            return deterministic_reflect_pad2d(input, pad)
        return self._functional.pad(input, pad, mode=mode, value=value)


def validate_deterministic_reflect_pad_receipt(receipt):
    """Validate the closed production receipt for adapter schema v2."""
    if (
        not isinstance(receipt, dict)
        or set(receipt) != set(FI_DETERMINISTIC_REFLECT_PAD_CONTRACT)
        or any(
            type(receipt[key]) is not type(expected) or receipt[key] != expected
            for key, expected in FI_DETERMINISTIC_REFLECT_PAD_CONTRACT.items()
        )
    ):
        raise ValueError("Invalid deterministic reflect-padding adapter receipt")
    return receipt


def install_deterministic_reflect_pad_adapter(model):
    """Install the adapter only into the cached pinned FI module.

    The assignment to that module's ``F`` binding is process-global for callers
    of the exact pinned module, but it does not mutate ``torch.nn.functional`` or
    any unrelated module. The adapter is idempotent and carries no model state.
    """
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise RuntimeError(
            "Deterministic reflect-padding adapter requires strict deterministic algorithms"
        )
    fi_class = load_pinned_fi_varnet_class()
    module = sys.modules.get(_MODULE_NAME)
    if (
        module is None
        or fi_class.__module__ != _MODULE_NAME
        or model.__class__.__module__ != _MODULE_NAME
        or not isinstance(model, fi_class)
        or Path(module.__file__).resolve() != _PINNED_FEATURE_SOURCE
        or _sha256(_PINNED_FEATURE_SOURCE) != PINNED_FEATURE_VARNET_SHA256
    ):
        raise RuntimeError("Deterministic reflect-padding adapter requires exact pinned FI model")

    state_before = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    current = module.F
    if current is torch_functional:
        module.F = _PinnedFunctionalProxy(torch_functional)
    elif not (
        isinstance(current, _PinnedFunctionalProxy)
        and current._functional is torch_functional
    ):
        raise RuntimeError("Pinned FI functional binding was unexpectedly modified")

    state_after = model.state_dict()
    if list(state_after) != list(state_before) or any(
        not torch.equal(state_after[key].detach().cpu(), expected)
        for key, expected in state_before.items()
    ):
        raise RuntimeError("Deterministic reflect-padding adapter changed model state_dict")
    return dict(FI_DETERMINISTIC_REFLECT_PAD_CONTRACT)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_fi_varnet_class(upstream_root=PINNED_UPSTREAM_ROOT):
    root = Path(upstream_root).resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not verify pinned upstream commit") from exc
    if revision != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected pinned upstream commit {PINNED_UPSTREAM_COMMIT}, found {revision}"
        )

    source = root / "fastmri_examples" / "feature_varnet" / "feature_varnet.py"
    if not source.is_file() or _sha256(source) != PINNED_FEATURE_VARNET_SHA256:
        raise RuntimeError("Pinned upstream FI-VarNet source SHA-256 mismatch")

    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        cached_source = Path(cached.__file__).resolve()
        if cached_source != source.resolve():
            raise RuntimeError("Pinned FI-VarNet module cache points to another source")
        return cached.FIVarNet

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pinned FI-VarNet from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module.FIVarNet


def build_pinned_fi_varnet(**kwargs):
    return load_pinned_fi_varnet_class()(**kwargs)


def _checkpoint_module_forward(module):
    if getattr(module, '_fi_activation_checkpoint_enabled', False):
        return
    original_forward = module.forward

    @functools.wraps(original_forward)
    def checkpointed_forward(*args, **kwargs):
        if not module.training or not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        return checkpoint(
            original_forward,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
            **kwargs,
        )

    module.forward = checkpointed_forward
    module._fi_activation_checkpoint_enabled = True


def enable_fi_activation_checkpointing(model):
    """Checkpoint every FI feature and image cascade without changing state keys."""
    if not hasattr(model, 'cascades') or not hasattr(model, 'image_cascades'):
        raise TypeError('FI activation checkpointing requires both cascade collections')
    feature_cascades = list(model.cascades)
    image_cascades = list(model.image_cascades)
    state_before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    for module in (*feature_cascades, *image_cascades):
        _checkpoint_module_forward(module)
    state_after = model.state_dict()
    state_unchanged = list(state_after) == list(state_before) and all(
        torch.equal(state_after[key], value) for key, value in state_before.items()
    )
    if not state_unchanged:
        raise RuntimeError('Activation checkpointing changed the FI model state_dict')
    return {
        'enabled': True,
        'implementation': 'torch.utils.checkpoint.checkpoint',
        'use_reentrant': False,
        'preserve_rng_state': True,
        'feature_cascades': len(feature_cascades),
        'image_cascades': len(image_cascades),
        'state_dict_unchanged': True,
    }


def verify_pinned_upstream_sources(upstream_root=PINNED_UPSTREAM_ROOT):
    """Attest the complete executable/licensing source closure used by the smoke."""
    root = Path(upstream_root).resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not verify pinned upstream commit") from exc
    if revision != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected pinned upstream commit {PINNED_UPSTREAM_COMMIT}, found {revision}"
        )
    paths = {
        'feature_varnet_sha256': (
            root / 'fastmri_examples' / 'feature_varnet' / 'feature_varnet.py',
            PINNED_FEATURE_VARNET_SHA256,
        ),
        'losses_sha256': (root / 'fastmri' / 'losses.py', PINNED_LOSSES_SHA256),
        'transforms_sha256': (
            root / 'fastmri' / 'data' / 'transforms.py',
            PINNED_TRANSFORMS_SHA256,
        ),
        'license_sha256': (root / 'LICENSE.md', PINNED_LICENSE_SHA256),
    }
    for description, (path, expected) in paths.items():
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise RuntimeError(f"Pinned upstream {description} mismatch")
    return {
        'upstream_commit': revision,
        **{description: expected for description, (_, expected) in paths.items()},
    }


def load_pinned_ssim_loss_class(upstream_root=PINNED_UPSTREAM_ROOT):
    """Load SSIMLoss only from the exact upstream source and MIT license bytes."""
    root = Path(upstream_root).resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not verify pinned upstream commit") from exc
    if revision != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected pinned upstream commit {PINNED_UPSTREAM_COMMIT}, found {revision}"
        )

    source = root / "fastmri" / "losses.py"
    license_path = root / "LICENSE.md"
    if not source.is_file() or _sha256(source) != PINNED_LOSSES_SHA256:
        raise RuntimeError("Pinned upstream SSIMLoss source SHA-256 mismatch")
    if not license_path.is_file() or _sha256(license_path) != PINNED_LICENSE_SHA256:
        raise RuntimeError("Pinned upstream MIT license SHA-256 mismatch")

    cached = sys.modules.get(_LOSS_MODULE_NAME)
    if cached is not None:
        if Path(cached.__file__).resolve() != source.resolve():
            raise RuntimeError("Pinned SSIMLoss module cache points to another source")
        return cached.SSIMLoss

    spec = importlib.util.spec_from_file_location(_LOSS_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pinned SSIMLoss from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LOSS_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_LOSS_MODULE_NAME, None)
        raise
    return module.SSIMLoss


def build_pinned_ssim_loss(**kwargs):
    return load_pinned_ssim_loss_class()(**kwargs)
