"""PromptMR+ training contracts routed through the repository's train.py stack."""

from contextlib import contextmanager
from copy import deepcopy
import ctypes
import errno
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import uuid

import torch
import yaml

from utils.learning.resume import (
    _publish_fd_without_overwrite,
    build_training_state,
    validate_training_checkpoint,
)
from utils.model.promptmr_plus_adapter import (
    PROMPTMR_PLUS_MANIFEST_SHA256,
    PROMPTMR_PLUS_ROOT,
    PromptMRInput,
    PromptMRPlusAdapter,
    build_promptmr_plus,
    import_promptmr_plus_module,
    verify_promptmr_plus_source,
)


_UPSTREAM_REPOSITORY = "https://github.com/hellopipu/PromptMR-plus"
_UPSTREAM_COMMIT = "934eeda6d4d18cd39e406fa1eee9e1f70603cb5e"
_TRAINING_CONFIG = "configs/train/pmr-plus/fm-knee.yaml"
_PINNED_RECIPE = {
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "scheduler": {"name": "StepLR", "step_size": 35, "gamma": 0.1},
    "loss": {"name": "SSIMLoss", "window": 7, "k1": 0.01, "k2": 0.03},
    "gradient_clip_norm": 0.01,
    "uniform_resolution": [384, 384],
    "use_checkpoint": True,
    "compute_sens_per_coil": True,
    "precision": "fp32",
    "batch_size": 1,
}


def load_promptmr_training_recipe():
    """Verify pinned source/config bytes and return the immutable local recipe."""
    manifest = verify_promptmr_plus_source()
    if manifest.get("repository") != _UPSTREAM_REPOSITORY:
        raise ValueError("PromptMR+ repository identity mismatch")
    if manifest.get("commit") != _UPSTREAM_COMMIT:
        raise ValueError("PromptMR+ upstream commit mismatch")

    config_path = PROMPTMR_PLUS_ROOT / _TRAINING_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_args = config["model"]["init_args"]
    data_args = config["data"]["init_args"]
    observed = {
        "learning_rate": model_args["lr"],
        "scheduler_step_size": model_args["lr_step_size"],
        "uniform_resolution": data_args["train_transform"]["init_args"][
            "uniform_resolution"
        ],
        "batch_size": data_args["batch_size"],
        "use_checkpoint": model_args["use_checkpoint"],
        "compute_sens_per_coil": model_args["compute_sens_per_coil"],
    }
    expected = {
        "learning_rate": _PINNED_RECIPE["learning_rate"],
        "scheduler_step_size": _PINNED_RECIPE["scheduler"]["step_size"],
        "uniform_resolution": _PINNED_RECIPE["uniform_resolution"],
        "batch_size": _PINNED_RECIPE["batch_size"],
        "use_checkpoint": _PINNED_RECIPE["use_checkpoint"],
        "compute_sens_per_coil": _PINNED_RECIPE["compute_sens_per_coil"],
    }
    if observed != expected:
        raise ValueError(
            f"PromptMR+ pinned training config mismatch: expected={expected}, observed={observed}"
        )

    recipe = deepcopy(_PINNED_RECIPE)
    recipe["upstream_repository"] = _UPSTREAM_REPOSITORY
    recipe["upstream_commit"] = _UPSTREAM_COMMIT
    recipe["source_manifest_sha256"] = PROMPTMR_PLUS_MANIFEST_SHA256
    recipe["training_config"] = _TRAINING_CONFIG
    recipe["training_config_sha256"] = manifest["files"][_TRAINING_CONFIG]
    return recipe


@dataclass(frozen=True)
class PromptMRTrainingComponents:
    model: torch.nn.Module
    loss: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: None
    provenance: dict


def build_promptmr_training_components(args, device):
    """Build the exact upstream model and immutable FP32 learning recipe."""
    recipe = load_promptmr_training_recipe()
    expected_args = {
        "model_family": "promptmr-plus",
        "batch_size": recipe["batch_size"],
        "lr": recipe["learning_rate"],
        "precision": recipe["precision"],
        "seed": 430,
    }
    observed_args = {
        key: getattr(args, key, None) for key in expected_args
    }
    if observed_args != expected_args:
        raise ValueError(
            f"PromptMR+ runtime recipe mismatch: expected={expected_args}, observed={observed_args}"
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = PromptMRPlusAdapter(build_promptmr_plus()).to(device=device)
    loss_class = import_promptmr_plus_module("mri_utils.losses").SSIMLoss
    loss = loss_class(
        win_size=recipe["loss"]["window"],
        k1=recipe["loss"]["k1"],
        k2=recipe["loss"]["k2"],
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=recipe["weight_decay"],
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=recipe["scheduler"]["step_size"],
        gamma=recipe["scheduler"]["gamma"],
    )
    provenance = {
        "model_family": "promptmr-plus",
        "precision": "fp32",
        "tf32_allowed": False,
        "seed": 430,
        "upstream_repository": recipe["upstream_repository"],
        "upstream_commit": recipe["upstream_commit"],
        "source_manifest_sha256": recipe["source_manifest_sha256"],
        "training_config": recipe["training_config"],
        "training_config_sha256": recipe["training_config_sha256"],
        "train_split": str(args.data_path_train),
        "validation_split": str(args.data_path_val),
        "recipe": deepcopy(_PINNED_RECIPE),
    }
    return PromptMRTrainingComponents(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        provenance=provenance,
    )


def promptmr_train_step(args, components, batch):
    """Run one exact FP32 PromptMR+ optimizer step with finite-gradient gates."""
    model = components.model
    optimizer = components.optimizer
    model.train()
    device = next(model.parameters()).device
    if batch.masked_kspace.shape[0] != 1:
        raise ValueError("PromptMR+ training requires batch size 1")
    acceleration = int(batch.acceleration.reshape(-1)[0].item())
    if acceleration not in (4, 8):
        raise ValueError("PromptMR+ acceleration must be 4 or 8")
    prepared = PromptMRInput(
        kspace=batch.masked_kspace.to(device=device),
        mask=batch.mask.to(device=device),
        num_low_frequencies=batch.num_low_frequencies.to(device=device),
        acceleration=acceleration,
    )
    target = batch.target.to(device=device)
    maximum = batch.max_value.to(device=device)

    optimizer.zero_grad(set_to_none=True)
    try:
        output = model(
            prepared,
            crop_size=args.promptmr_uniform_resolution,
            use_checkpoint=args.promptmr_use_checkpoint,
            compute_sens_per_coil=args.promptmr_compute_sens_per_coil,
        )
        loss = components.loss(
            output.unsqueeze(1), target.unsqueeze(1), maximum
        )
        if loss.numel() != 1 or not torch.isfinite(loss):
            raise FloatingPointError("PromptMR+ training loss must be one finite scalar")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("PromptMR+ participating gradients must be finite")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.promptmr_gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("PromptMR+ gradient norm must be finite")
        optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "optimizer_steps": 1,
        }
    finally:
        optimizer.zero_grad(set_to_none=True)


def create_promptmr_data_loader(data_path, args, shuffle):
    """Create the fixed batch-1 paired loader without changing VarNet loading."""
    from torch.utils.data import DataLoader
    from utils.data.promptmr_plus import PromptMRPlusSliceData

    dataset = PromptMRPlusSliceData(
        data_path,
        input_key=args.input_key,
        target_key=args.target_key,
        max_key=args.max_key,
    )
    return DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle)


@contextmanager
def _open_staged_file(directory_fd, mode):
    name = f".promptmr-stage-{uuid.uuid4().hex}"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    identity = os.fstat(fd)
    try:
        with os.fdopen(fd, mode, closefd=False) as handle:
            yield handle, name, (identity.st_dev, identity.st_ino)
    finally:
        os.close(fd)
        try:
            observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (observed.st_dev, observed.st_ino) == (
                identity.st_dev,
                identity.st_ino,
            ):
                os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _publish_staged(source, directory_fd, destination_name):
    _publish_fd_without_overwrite(source, directory_fd, destination_name)
    published = os.stat(
        destination_name, dir_fd=directory_fd, follow_symlinks=False
    )
    source_identity = os.fstat(source.fileno())
    if (published.st_dev, published.st_ino) != (
        source_identity.st_dev,
        source_identity.st_ino,
    ):
        raise RuntimeError("PromptMR+ published inode differs from validated inode")
    os.fsync(directory_fd)


def _cpu_state_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state_snapshot(item) for item in value)
    return deepcopy(value)


def _states_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _states_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _states_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _write_json_exclusive(directory_fd, name, payload):
    with _open_staged_file(directory_fd, mode="w+") as (
        handle,
        _staging_name,
        _identity,
    ):
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        if json.load(handle) != payload:
            raise ValueError(f"PromptMR+ JSON roundtrip mismatch: {name}")
        _publish_staged(handle, directory_fd, name)


def _save_checkpoint_exclusive(directory_fd, name, checkpoint):
    expected_cpu = _cpu_state_snapshot(checkpoint)
    with _open_staged_file(directory_fd, mode="w+b") as (
        handle,
        _staging_name,
        _identity,
    ):
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        checkpoint_bytes = handle.read()
        digest = hashlib.sha256(checkpoint_bytes).hexdigest()
        handle.seek(0)
        roundtrip = torch.load(handle, map_location="cpu", weights_only=True)
        validate_training_checkpoint(roundtrip)
        required = {
            "model_family",
            "global_optimizer_step",
            "scheduler",
            "scaler",
            "provenance",
            "history",
        }
        if not required.issubset(roundtrip):
            raise ValueError(
                f"PromptMR+ smoke checkpoint missing fields: "
                f"{sorted(required - set(roundtrip))}"
            )
        if (
            roundtrip["scaler"] is not None
            or roundtrip["model_family"] != "promptmr-plus"
            or roundtrip["global_optimizer_step"] != 1
            or not isinstance(roundtrip["history"], list)
            or len(roundtrip["history"]) != 1
            or not _states_equal(roundtrip, expected_cpu)
        ):
            raise ValueError("PromptMR+ smoke checkpoint roundtrip mismatch")
        _publish_staged(handle, directory_fd, name)
    return roundtrip, digest


def _collect_smoke_telemetry(device, preflight=None):
    telemetry = {
        "precision": "fp32",
        "device": str(device),
        "preflight": deepcopy(preflight),
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "process_gpu_memory_mib": None,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    if device.type == "cuda":
        from scripts.probe_promptmr_plus_8gb import environment, process_gpu_memory_mib

        torch.cuda.synchronize(device)
        runtime_environment = environment()
        runtime_environment["probe_platform_label"] = runtime_environment.get("platform")
        runtime_environment["platform"] = (
            "VESSL"
            if any(name.startswith("VESSL_") for name in os.environ)
            else "local"
        )
        telemetry.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_index": int(device.index),
                "memory_total_mib": torch.cuda.get_device_properties(device).total_memory
                / (1024.0**2),
                "runtime_environment": runtime_environment,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                / (1024.0**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device)
                / (1024.0**2),
                "process_gpu_memory_mib": process_gpu_memory_mib(),
            }
        )
    return telemetry


def validate_promptmr_cuda_capacity(device):
    """Require a 24 GiB-class device before any PromptMR+ output or model access."""
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    if not 23 * 1024**3 <= total_memory < 25 * 1024**3:
        raise RuntimeError(
            "PromptMR+ one-step smoke requires an RTX 3090-class 24 GiB device"
        )
    return total_memory


def preflight_promptmr_gpu():
    """Fail closed unless the smoke has exclusive access to an idle GPU."""
    from scripts.probe_promptmr_plus_8gb import (
        assert_exclusive_gpu_process,
        assert_gpu_idle_before_probe,
    )

    assert_exclusive_gpu_process()
    return assert_gpu_idle_before_probe()


def _rename_no_replace(directory_fd, source_name, destination_name):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            directory_fd,
            os.fsencode(source_name),
            directory_fd,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number, os.strerror(error_number), destination_name
            )
        raise OSError(error_number, os.strerror(error_number), destination_name)


@contextmanager
def _open_smoke_directories(run_path, exp_path):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_fd = os.open(run_path.parent, flags)
    staging_name = f".promptmr-run-stage-{uuid.uuid4().hex}"
    run_fd = None
    exp_fd = None
    published = False
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        run_fd = os.open(staging_name, flags, dir_fd=parent_fd)
        _write_json_exclusive(
            run_fd,
            "RUN_INCOMPLETE.json",
            {"schema": "promptmr-plus-one-step-smoke/incomplete-v1"},
        )
        os.mkdir("checkpoints", mode=0o700, dir_fd=run_fd)
        os.fsync(run_fd)
        exp_fd = os.open("checkpoints", flags, dir_fd=run_fd)
        _rename_no_replace(parent_fd, staging_name, run_path.name)
        published = True
        os.fsync(parent_fd)
        yield run_fd, exp_fd
    finally:
        if exp_fd is not None:
            os.close(exp_fd)
        if run_fd is not None and not published:
            for name in ("RUN_INCOMPLETE.json",):
                try:
                    os.unlink(name, dir_fd=run_fd)
                except FileNotFoundError:
                    pass
            try:
                os.rmdir("checkpoints", dir_fd=run_fd)
            except FileNotFoundError:
                pass
            os.fsync(run_fd)
        if run_fd is not None:
            os.close(run_fd)
        if not published:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _run_promptmr_in_open_directories(args, device, run_fd, exp_fd):
    components = build_promptmr_training_components(args, device)
    loader = create_promptmr_data_loader(args.data_path_train, args, shuffle=False)
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("PromptMR+ training split is empty") from exc
    inventory_sha256 = getattr(loader.dataset, "inventory_sha256", None)
    if (
        not isinstance(inventory_sha256, str)
        or len(inventory_sha256) != 64
        or any(character not in "0123456789abcdef" for character in inventory_sha256)
    ):
        raise ValueError("PromptMR+ training inventory digest is invalid")
    components.provenance["train_inventory_sha256"] = inventory_sha256
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = promptmr_train_step(args, components, batch)
    history = [
        {
            "optimizer_step": 1,
            "loss": metrics["loss"],
            "gradient_norm": metrics["gradient_norm"],
            "fname": str(batch.fname[0]),
            "slice_num": int(batch.slice_num.reshape(-1)[0].item()),
            "acceleration": int(batch.acceleration.reshape(-1)[0].item()),
        }
    ]
    checkpoint = build_training_state(
        epoch=0,
        model=components.model,
        optimizer=components.optimizer,
        best_val_loss=metrics["loss"],
    )
    checkpoint.update(
        {
            "model_family": "promptmr-plus",
            "global_optimizer_step": 1,
            "scheduler": components.scheduler.state_dict(),
            "scaler": None,
            "provenance": deepcopy(components.provenance),
            "history": history,
        }
    )
    validate_training_checkpoint(checkpoint)
    roundtrip, checkpoint_sha256 = _save_checkpoint_exclusive(
        exp_fd, "smoke_model.pt", checkpoint
    )
    telemetry = _collect_smoke_telemetry(
        device, getattr(args, "promptmr_gpu_preflight", None)
    )
    roundtrip_history = roundtrip["history"]
    roundtrip_provenance = roundtrip["provenance"]
    _write_json_exclusive(run_fd, "smoke_history.json", roundtrip_history)
    report = {
        "schema": "promptmr-plus-one-step-smoke/v1",
        "status": "PASS",
        "optimizer_steps": roundtrip["global_optimizer_step"],
        "loss": roundtrip_history[0]["loss"],
        "gradient_norm": roundtrip_history[0]["gradient_norm"],
        "checkpoint_roundtrip": "PASS",
        "checkpoint_sha256": checkpoint_sha256,
        "telemetry": telemetry,
        "provenance": roundtrip_provenance,
    }
    _write_json_exclusive(run_fd, "smoke_report.json", report)
    _write_json_exclusive(
        run_fd,
        "RUN_COMPLETE.json",
        {"schema": "promptmr-plus-one-step-smoke/complete-v1"},
    )
    os.unlink("RUN_INCOMPLETE.json", dir_fd=run_fd)
    os.fsync(run_fd)
    return report


def run_promptmr_one_step_smoke(args, device):
    """Run and durably publish one bounded real-batch PromptMR+ optimizer step."""
    load_promptmr_training_recipe()
    if device.type == "cuda":
        validate_promptmr_cuda_capacity(device)
    run_path = Path(args.run_dir)
    exp_path = Path(args.exp_dir)
    if (
        re.fullmatch(r"FEATURE_PROMPTMR_PLUS_[A-Z0-9_]{1,96}", run_path.name)
        is None
        or exp_path != run_path / "checkpoints"
        or not run_path.parent.is_dir()
        or run_path.parent.is_symlink()
    ):
        raise ValueError("Unsafe PromptMR+ smoke output path")
    try:
        with _open_smoke_directories(run_path, exp_path) as (run_fd, exp_fd):
            return _run_promptmr_in_open_directories(
                args, device, run_fd, exp_fd
            )
    except BaseException:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise
