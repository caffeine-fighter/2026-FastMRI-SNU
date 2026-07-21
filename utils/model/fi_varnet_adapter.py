"""Thin, integrity-checked import adapter for the pinned upstream FI-VarNet."""

import functools
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import torch
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
