import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

import train
import utils.learning.fi_acc8_full_training as full_training_module
from utils.learning.fi_acc8_training import inspect_acc8_training_data
from utils.learning.fi_acc8_full_training import (
    FI_ACC8_FULL_RECIPE,
    FI_ACC8_FULL_NAMESPACE,
    FullSamplerCursor,
    VerifiedAcc8FileTransaction,
    atomic_write_status,
    build_full_checkpoint,
    deterministic_file_order,
    deterministic_slice_order,
    load_full_checkpoint,
    open_verified_acc8_file,
    publish_full_checkpoint,
    run_boundary_engine,
    run_full_finite_optimizer_step,
    snapshot_bytes,
)


def _make_pair(root, name, slices=2, offset=0):
    (root / "kspace").mkdir(parents=True, exist_ok=True)
    (root / "image").mkdir(parents=True, exist_ok=True)
    shape = (2, 8, 6)
    mask = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float32)
    values = np.arange(slices * np.prod(shape), dtype=np.float32).reshape(
        (slices, *shape)
    ) + offset
    kspace = (values + 1j).astype(np.complex64)
    with h5py.File(root / "kspace" / name, "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        hf.create_dataset("mask", data=mask)
    with h5py.File(root / "image" / name, "w") as hf:
        hf.create_dataset(
            "image_label", data=np.full((slices, 8, 6), 2 + offset, np.float32)
        )
        hf.attrs["max"] = np.float32(10 + offset)


def _manifest(root, files=2, slices_each=2):
    for index in range(files):
        _make_pair(
            root,
            f"brain{index}_acc8_sample.h5",
            slices=slices_each,
            offset=index * 100,
        )
    _make_pair(root, "ignored_acc4_sample.h5", slices=1, offset=500)
    return inspect_acc8_training_data(
        root,
        input_key="kspace",
        target_key="image_label",
        max_key="max",
        expected_files=files,
        expected_slices=files * slices_each,
        maximum_input_shape=(2, 8, 6),
        expected_total_files=files + 1,
        expected_ignored_acc4_files=1,
    )


def _tiny_state():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FI_ACC8_FULL_RECIPE.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / FI_ACC8_FULL_RECIPE.ramp_steps, 1.0)
    )
    return model, optimizer, scheduler


def _reflect_receipt():
    return dict(full_training_module.FI_DETERMINISTIC_REFLECT_PAD_CONTRACT)


def _bindings():
    return {
        "source_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
        "gpu_uuid": "GPU-exact",
        "reflect_padding_adapter": _reflect_receipt(),
    }


def _advance_tiny_state(model, optimizer, scheduler, steps):
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([[float(index + 1)]])).square().sum().backward()
        optimizer.step()
        scheduler.step()


def _exact_transactions(records, cursor):
    remaining = (cursor.epoch - 1) * len(records) + cursor.file_cursor
    transactions = []
    for epoch in range(1, cursor.epoch + 1):
        mapping = {record.name: record for record in records}
        for name in deterministic_file_order(records, epoch):
            if not remaining:
                return transactions
            record = mapping[name]
            transactions.append(
                {
                    "name": name,
                    "accepted": True,
                    "epoch": epoch,
                    "slices": record.slices,
                    "slice_order": deterministic_slice_order(record, epoch),
                }
            )
            remaining -= 1
    return transactions


def _write_resume_generation(root, name, sha256, sampler):
    name = f"generation-{hashlib.sha256(name.encode('utf-8')).hexdigest()[:32]}"
    generation_dir = root / "checkpoint-generations" / name
    generation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = generation_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(name.encode("ascii"))
    metadata = {
        "schema": "fi-varnet-acc8-checkpoint-generation-v1",
        "generation": name,
        "checkpoint_sha256": sha256,
        "sampler": sampler,
        "epoch_end": False,
        "metrics": {"loss_sum": 0.0, "loss_count": 0},
    }
    (generation_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return {
        "generation": name,
        "checkpoint_sha256": sha256,
        "sampler": sampler,
    }, checkpoint_path


def _pointer_entry(index, sampler=None):
    if sampler is None:
        sampler = FullSamplerCursor(1, 0, 0).as_dict()
    return {
        "generation": f"generation-{index:032x}",
        "checkpoint_sha256": f"{index:064x}",
        "sampler": sampler,
    }


def _canonical_pointer():
    epoch_generations = [
        _pointer_entry(
            epoch,
            FullSamplerCursor(
                epoch, 0, (epoch - 1) * FI_ACC8_FULL_RECIPE.slices_per_epoch
            ).as_dict(),
        )
        for epoch in range(2, FI_ACC8_FULL_RECIPE.base_epochs + 2)
    ]
    return {
        "format_version": 1,
        "latest": _pointer_entry(100),
        "previous": _pointer_entry(101),
        "epoch_generations": epoch_generations,
    }


def _resume_args(checkpoint_path, sha256):
    return SimpleNamespace(
        fi_acc8_full_training=True,
        data_path_train=Path("/root/Data/train"),
        resume_checkpoint=checkpoint_path,
        resume_checkpoint_sha256=sha256,
        input_key="kspace",
        target_key="image_label",
        max_key="max",
        GPU_NUM=0,
        expected_gpu_uuid="GPU-exact",
    )


def test_full_recipe_and_cli_are_separate_frozen_epoch_30_lane():
    assert FI_ACC8_FULL_NAMESPACE == "EXP_FI_ACC8_CKPT_BASE_E30_R1"
    assert FI_ACC8_FULL_RECIPE.as_dict() == {
        "schema": "fi-varnet-acc8-checkpointed-full-training-v2",
        "model_family": "fi-varnet-acc8",
        "namespace": "EXP_FI_ACC8_CKPT_BASE_E30_R1",
        "scope": "FULL_TRAINING_ONLY",
        "scratch": True,
        "external_learned_state": False,
        "seed": 431,
        "batch_size": 1,
        "precision": "fp32",
        "autocast": False,
        "optimizer": "AdamW",
        "lr": 0.0003,
        "weight_decay": 0.0,
        "loss": "upstream-fastmri-SSIMLoss",
        "gradient_clipping": False,
        "num_cascades": 12,
        "chans": 18,
        "pools": 4,
        "sens_chans": 8,
        "sens_pools": 4,
        "acceleration": 8,
        "train_files": 85,
        "slices_per_epoch": 2315,
        "base_epochs": 30,
        "base_max_steps": 69450,
        "scheduler_horizon_epochs": 40,
        "scheduler_max_steps": 92600,
        "ramp_steps": 3704,
        "cosine_decay_start": 46300,
        "checkpoint_file_cadence": 1,
        "status_interval_seconds": 300,
        "activation_checkpoint_feature_cascades": 12,
        "activation_checkpoint_image_cascades": 12,
        "reflect_padding_adapter_schema": "fi-varnet-reflect-padding-adapter-v2",
        "reflect_padding_adapter_implementation": "utils.model.fi_varnet_adapter.deterministic_reflect_pad2d",
        "reflect_padding_adapter_version": "1.0.0",
        "reflect_padding_native_forward_exact": True,
        "reflect_padding_state_dict_unchanged": True,
        "reflect_padding_strict_deterministic_algorithms": True,
    }
    argv = [
        "train.py",
        "--model-family",
        "fi-varnet-acc8",
        "--fi-acc8-full-training",
        "--expected-gpu-uuid",
        "GPU-exact",
        "--data-path-train",
        "/root/Data/train",
    ]
    with patch("sys.argv", argv):
        args = train.parse()
    assert args.fi_acc8_full_training is True
    assert args.fi_acc8_one_step_smoke is False
    assert args.net_name == Path(FI_ACC8_FULL_NAMESPACE)
    assert args.num_epochs == 30
    with patch("sys.argv", argv + ["--fi-acc8-one-step-smoke"]):
        with pytest.raises(SystemExit):
            train.parse()
    with patch("sys.argv", argv + ["--num-epochs", "40"]):
        with pytest.raises(SystemExit):
            train.parse()


def test_determinism_contract_is_exact_and_fail_closed_before_cuda(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with patch.object(torch, "use_deterministic_algorithms") as enable, patch.object(
        torch, "are_deterministic_algorithms_enabled", return_value=True
    ):
        contract = full_training_module.configure_determinism_pre_cuda()
    enable.assert_called_once_with(True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert contract == {
        "schema": "fi-acc8-determinism-v2",
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "implementation": "utils.model.fi_varnet_adapter.deterministic_reflect_pad2d",
        "version": "1.0.0",
        "native_forward_exact": True,
        "state_dict_unchanged": True,
        "strict_deterministic_algorithms": True,
    }
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with patch.object(torch, "use_deterministic_algorithms") as enable:
        with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
            full_training_module.configure_determinism_pre_cuda()
    enable.assert_not_called()

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with patch.object(
        torch,
        "use_deterministic_algorithms",
        side_effect=RuntimeError("nondeterministic operation is unsupported"),
    ):
        with pytest.raises(RuntimeError, match="nondeterministic operation"):
            full_training_module.configure_determinism_pre_cuda()


def test_production_model_build_fails_closed_without_valid_adapter_receipt():
    model = torch.nn.Linear(2, 2)
    with patch.object(full_training_module, "build_model", return_value=model), patch.object(
        full_training_module,
        "install_deterministic_reflect_pad_adapter",
        return_value=None,
    ), patch.object(full_training_module, "enable_fi_activation_checkpointing") as activation:
        with pytest.raises(ValueError, match="reflect-padding adapter receipt"):
            full_training_module._build_full_training_model_with_adapters(SimpleNamespace())
    activation.assert_not_called()


def test_production_model_build_returns_valid_adapter_and_activation_receipts():
    model = torch.nn.Linear(2, 2)
    reflect_receipt = dict(full_training_module.FI_DETERMINISTIC_REFLECT_PAD_CONTRACT)
    activation_receipt = dict(full_training_module.FI_ACTIVATION_CHECKPOINT_CONTRACT)
    with patch.object(full_training_module, "build_model", return_value=model), patch.object(
        full_training_module,
        "install_deterministic_reflect_pad_adapter",
        return_value=reflect_receipt,
    ), patch.object(
        full_training_module,
        "enable_fi_activation_checkpointing",
        return_value=activation_receipt,
    ):
        actual = full_training_module._build_full_training_model_with_adapters(
            SimpleNamespace()
        )
    assert actual == (model, reflect_receipt, activation_receipt)


def test_source_and_recipe_digests_are_closed_over_exact_adapter_receipt():
    source = {"commit": "exact", "feature_varnet_sha256": "a" * 64}
    receipt = _reflect_receipt()
    source_digest = full_training_module.source_binding_sha256(source, receipt)
    recipe_digest = full_training_module.recipe_sha256(receipt)
    assert source_digest == full_training_module.source_binding_sha256(source, receipt)
    assert recipe_digest == full_training_module.recipe_sha256(receipt)

    malformed = dict(receipt, native_forward_exact=False)
    with pytest.raises(ValueError, match="reflect-padding adapter receipt"):
        full_training_module.source_binding_sha256(source, malformed)
    with pytest.raises(ValueError, match="reflect-padding adapter receipt"):
        full_training_module.recipe_sha256(malformed)


def test_resource_preflight_uses_checkpoint_and_max_volume_and_fails_closed(tmp_path):
    manifest = SimpleNamespace(
        records=(
            SimpleNamespace(kspace_size=40_000_000, image_size=10_000_000),
            SimpleNamespace(kspace_size=25_000_000, image_size=5_000_000),
        )
    )
    result = full_training_module.preflight_full_training_resources(
        manifest,
        tmp_path / "not-created",
        available_ram_bytes=8_000_000_000,
        free_disk_bytes=100_000_000_000,
    )
    assert result["checkpoint_bytes"] == 1_479_000_000
    assert result["max_volume_bytes"] == 50_000_000
    assert result["max_pair_bytes"] == 50_000_000
    assert result["retained_checkpoint_limit"] == 32
    assert result["cpu_model_bytes"] == result["checkpoint_bytes"]
    assert result["cpu_optimizer_bytes"] == result["checkpoint_bytes"]
    assert result["checkpoint_snapshot_bytes"] == result["checkpoint_bytes"]
    assert result["checkpoint_serialization_bytes"] == result["checkpoint_bytes"]
    assert result["required_ram_bytes"] == (
        result["max_pair_bytes"]
        + result["cpu_model_bytes"]
        + result["cpu_optimizer_bytes"]
        + result["checkpoint_snapshot_bytes"]
        + result["checkpoint_serialization_bytes"]
        + result["ram_margin_bytes"]
    )
    assert result["retained_checkpoint_bytes"] == 32 * result["checkpoint_bytes"]
    assert result["staging_checkpoint_bytes"] == result["checkpoint_bytes"]
    assert result["required_disk_bytes"] == (
        result["retained_checkpoint_bytes"]
        + result["staging_checkpoint_bytes"]
        + result["disk_margin_bytes"]
    )
    assert not (tmp_path / "not-created").exists()

    with pytest.raises(RuntimeError, match="RAM"):
        full_training_module.preflight_full_training_resources(
            manifest,
            tmp_path / "not-created",
            available_ram_bytes=result["required_ram_bytes"] - 1,
            free_disk_bytes=result["required_disk_bytes"],
        )
    with pytest.raises(RuntimeError, match="disk"):
        full_training_module.preflight_full_training_resources(
            manifest,
            tmp_path / "not-created",
            available_ram_bytes=result["required_ram_bytes"],
            free_disk_bytes=result["required_disk_bytes"] - 1,
        )
    assert not (tmp_path / "not-created").exists()


def test_resource_preflight_measures_output_filesystem_not_data_filesystem(tmp_path):
    manifest = SimpleNamespace(
        root=Path("/different/data/filesystem"),
        records=(SimpleNamespace(kspace_size=10, image_size=20),),
    )
    output = tmp_path / "result" / "run"
    measured = []

    def disk_usage(path):
        measured.append(Path(path))
        return SimpleNamespace(free=100_000_000_000)

    with patch.object(full_training_module.shutil, "disk_usage", side_effect=disk_usage):
        full_training_module.preflight_full_training_resources(
            manifest, output, available_ram_bytes=100_000_000_000
        )
    assert measured == [tmp_path]


def test_seeded_file_and_slice_permutations_are_total_and_epoch_specific(tmp_path):
    manifest = _manifest(tmp_path)
    first = deterministic_file_order(manifest, 1)
    assert first == deterministic_file_order(manifest, 1)
    assert set(first) == {record.name for record in manifest.records}
    assert first != deterministic_file_order(manifest, 2)
    for record in manifest.records:
        order = deterministic_slice_order(record, 1)
        assert order == deterministic_slice_order(record, 1)
        assert sorted(order) == list(range(record.slices))


def test_verified_file_transaction_uses_all_slices_and_records_same_fd_hashes(tmp_path):
    manifest = _manifest(tmp_path)
    record = manifest.records[0]
    order = deterministic_slice_order(record, 1)
    with open_verified_acc8_file(manifest, record, order) as transaction:
        samples = list(transaction)
    assert [sample["slice"] for sample in samples] == order
    assert all(sample["fname"] == record.name for sample in samples)
    assert transaction.receipt["kspace_sha256"] == record.kspace_sha256
    assert transaction.receipt["image_sha256"] == record.image_sha256
    assert transaction.receipt["slice_order"] == order
    assert transaction.receipt["accepted"] is True
    assert all(sample["mask"].dtype == torch.float32 for sample in samples)


def test_verified_file_transaction_has_one_directly_usable_implementation(tmp_path):
    manifest = _manifest(tmp_path)
    record = manifest.records[0]
    order = deterministic_slice_order(record, 1)

    with VerifiedAcc8FileTransaction(manifest, record, order) as transaction:
        assert len(list(transaction)) == record.slices

    assert transaction.receipt["accepted"] is True
    assert not hasattr(full_training_module, "_VerifiedAcc8FileTransactionFixed")


def test_verified_file_transaction_rejects_partial_consumption_and_mutation(tmp_path):
    manifest = _manifest(tmp_path)
    record = manifest.records[0]
    order = deterministic_slice_order(record, 1)
    with pytest.raises(ValueError, match="all slices"):
        with open_verified_acc8_file(manifest, record, order) as transaction:
            next(iter(transaction))

    leaf = tmp_path / "kspace" / record.name
    with h5py.File(leaf, "r+") as hf:
        hf["kspace"][0, 0, 0, 0] += np.complex64(1)
    with pytest.raises(ValueError, match="bytes changed"):
        with open_verified_acc8_file(manifest, record, order):
            pass


def test_boundary_engine_interruption_resume_has_no_duplicate_or_skip_and_exact_state_bytes():
    records = (
        SimpleNamespace(name="a_acc8.h5", slices=3),
        SimpleNamespace(name="b_acc8.h5", slices=2),
    )

    def execute(model, optimizer, scheduler, cursor, stop=None):
        ordered = []
        checkpoints = []

        def step(record, slice_index):
            value = torch.tensor([[float((slice_index + 1) * (1 if record.name[0] == "a" else 7))]])
            optimizer.zero_grad(set_to_none=True)
            loss = model(value).square().sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ordered.append((cursor.epoch, record.name, slice_index))

        def boundary(next_cursor, receipt):
            checkpoints.append(
                (
                    copy.deepcopy(next_cursor),
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(optimizer.state_dict()),
                    copy.deepcopy(scheduler.state_dict()),
                    tuple(ordered),
                    receipt,
                )
            )

        try:
            run_boundary_engine(
                records,
                cursor,
                step,
                boundary,
                epoch_limit=2,
                stop_after_boundaries=stop,
            )
        except InterruptedError as error:
            error.checkpoints = checkpoints
            raise
        return ordered, checkpoints

    torch.manual_seed(9)
    full_model, full_optimizer, full_scheduler = _tiny_state()
    full_order, _ = execute(
        full_model, full_optimizer, full_scheduler, FullSamplerCursor(1, 0, 0), None
    )
    full_bytes = snapshot_bytes(
        {
            "model": full_model.state_dict(),
            "optimizer": full_optimizer.state_dict(),
            "scheduler": full_scheduler.state_dict(),
            "cursor": FullSamplerCursor(3, 0, 10).as_dict(),
        }
    )

    torch.manual_seed(9)
    first_model, first_optimizer, first_scheduler = _tiny_state()
    with pytest.raises(InterruptedError) as interruption:
        execute(
            first_model,
            first_optimizer,
            first_scheduler,
            FullSamplerCursor(1, 0, 0),
            1,
        )
    checkpoints = interruption.value.checkpoints
    cursor, model_state, optimizer_state, scheduler_state, prefix, _ = checkpoints[-1]
    resumed_model, resumed_optimizer, resumed_scheduler = _tiny_state()
    resumed_model.load_state_dict(model_state)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    suffix, _ = execute(
        resumed_model, resumed_optimizer, resumed_scheduler, cursor, None
    )
    resumed_bytes = snapshot_bytes(
        {
            "model": resumed_model.state_dict(),
            "optimizer": resumed_optimizer.state_dict(),
            "scheduler": resumed_scheduler.state_dict(),
            "cursor": FullSamplerCursor(3, 0, 10).as_dict(),
        }
    )
    assert list(prefix) + suffix == full_order
    assert len(full_order) == len(set(full_order)) == 10
    assert resumed_bytes == full_bytes


def test_full_optimizer_step_uses_nominal_scheduler_without_smoke_lr_priming():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, kspace, mask, crop_size=None):
            del mask, crop_size
            return kspace[:, 0, :, :, 0] * self.scale

    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FI_ACC8_FULL_RECIPE.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / FI_ACC8_FULL_RECIPE.ramp_steps, 1.0)
    )
    sample = {
        "kspace": torch.ones(1, 1, 8, 8, 2),
        "mask": torch.ones(1, 1, 1, 8, 1, dtype=torch.float32),
        "target": torch.ones(1, 8, 8),
        "maximum": torch.tensor([1.0]),
        "fname": "tiny_acc8.h5",
        "slice": 0,
    }
    result = run_full_finite_optimizer_step(
        model,
        sample,
        lambda output, target, maximum: ((output - target) ** 2).mean(),
        optimizer,
        scheduler,
        torch.device("cpu"),
        global_step=0,
    )
    assert result["applied_lr"] == 0.0
    assert result["post_step_lr"] == pytest.approx(
        FI_ACC8_FULL_RECIPE.lr / FI_ACC8_FULL_RECIPE.ramp_steps
    )
    assert result["global_step"] == 1
    assert result["finite_loss"] is True
    assert result["finite_gradients"] is True
    assert result["finite_parameters"] is True


def test_checkpoint_invariants_cover_steps_lr_transactions_and_finite_tensors():
    records = (
        SimpleNamespace(name="a_acc8.h5", slices=2),
        SimpleNamespace(name="b_acc8.h5", slices=2),
    )
    model, optimizer, scheduler = _tiny_state()
    _advance_tiny_state(model, optimizer, scheduler, 2)
    cursor = FullSamplerCursor(1, 1, 2)
    bindings = {
        "source_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
        "gpu_uuid": "GPU-exact",
        "reflect_padding_adapter": _reflect_receipt(),
    }
    checkpoint = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=cursor,
        records=records,
        bindings=bindings,
        transactions=_exact_transactions(records, cursor),
        metrics={"loss_sum": 1.0, "loss_count": 2},
        reflect_padding_adapter=_reflect_receipt(),
    )
    full_training_module.validate_full_checkpoint(checkpoint, records=records)

    mutations = []
    bad = copy.deepcopy(checkpoint)
    bad["metrics"]["loss_count"] = 1
    mutations.append((bad, "loss_count"))
    bad = copy.deepcopy(checkpoint)
    bad["scheduler"]["last_epoch"] = 1
    mutations.append((bad, "last_epoch"))
    bad = copy.deepcopy(checkpoint)
    bad["scheduler"]["_step_count"] = 2
    mutations.append((bad, "_step_count"))
    bad = copy.deepcopy(checkpoint)
    first_state = next(iter(bad["optimizer"]["state"].values()))
    first_state["step"] = torch.tensor(1.0)
    mutations.append((bad, "AdamW"))
    bad = copy.deepcopy(checkpoint)
    bad["optimizer"]["param_groups"][0]["lr"] = 0.25
    mutations.append((bad, "nominal"))
    bad = copy.deepcopy(checkpoint)
    bad["transactions"][0]["name"] = "wrong_acc8.h5"
    mutations.append((bad, "transaction"))
    bad = copy.deepcopy(checkpoint)
    next(iter(bad["model"].values())).view(-1)[0] = float("nan")
    mutations.append((bad, "nonfinite"))
    bad = copy.deepcopy(checkpoint)
    next(iter(bad["optimizer"]["state"].values()))["exp_avg"].view(-1)[0] = float("inf")
    mutations.append((bad, "nonfinite"))

    for bad, message in mutations:
        with pytest.raises(ValueError, match=message):
            full_training_module.validate_full_checkpoint(bad, records=records)


def test_checkpoint_reflect_adapter_receipt_is_required_closed_and_provenance_bound():
    model, optimizer, scheduler = _tiny_state()
    receipt = dict(full_training_module.FI_DETERMINISTIC_REFLECT_PAD_CONTRACT)
    checkpoint = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=FullSamplerCursor(1, 0, 0),
        records=(SimpleNamespace(name="a_acc8.h5", slices=1),),
        bindings=_bindings(),
        transactions=[],
        metrics={"loss_sum": 0.0, "loss_count": 0},
        reflect_padding_adapter=receipt,
    )
    assert checkpoint["reflect_padding_adapter"] == receipt
    assert checkpoint["bindings"]["reflect_padding_adapter"] == receipt
    assert checkpoint["provenance"]["reflect_padding_adapter"] == receipt

    missing = copy.deepcopy(checkpoint)
    missing.pop("reflect_padding_adapter")
    with pytest.raises(ValueError, match="top-level schema"):
        full_training_module.validate_full_checkpoint(missing)

    malformed = copy.deepcopy(checkpoint)
    malformed["reflect_padding_adapter"]["native_forward_exact"] = False
    with pytest.raises(ValueError, match="reflect-padding adapter receipt"):
        full_training_module.validate_full_checkpoint(malformed)

    malformed_binding = copy.deepcopy(checkpoint)
    malformed_binding["bindings"]["reflect_padding_adapter"]["version"] = "other"
    with pytest.raises(ValueError, match="reflect-padding adapter receipt"):
        full_training_module.validate_full_checkpoint(malformed_binding)

    disagreeing = copy.deepcopy(checkpoint)
    disagreeing["provenance"]["reflect_padding_adapter"]["version"] = "other"
    with pytest.raises(ValueError, match="provenance"):
        full_training_module.validate_full_checkpoint(disagreeing)


def test_full_checkpoint_requires_exact_sha_bindings_and_valid_next_cursor(tmp_path):
    model, optimizer, scheduler = _tiny_state()
    _advance_tiny_state(model, optimizer, scheduler, 2)
    bindings = {
        "source_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
        "gpu_uuid": "GPU-exact",
        "reflect_padding_adapter": _reflect_receipt(),
    }
    cursor = FullSamplerCursor(1, 1, 2)
    records = (
        SimpleNamespace(name="a_acc8.h5", slices=2),
        SimpleNamespace(name="b_acc8.h5", slices=2),
    )
    checkpoint_provenance = {
        "source": {"upstream_commit": "exact"},
        "data": {"manifest_sha256": bindings["data_manifest_sha256"]},
        "recipe": FI_ACC8_FULL_RECIPE.as_dict(),
        "gpu": {"uuid": "GPU-exact"},
        "reflect_padding_adapter": _reflect_receipt(),
    }
    checkpoint = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=cursor,
        records=records,
        bindings=bindings,
        provenance=checkpoint_provenance,
        transactions=_exact_transactions(records, cursor),
        metrics={"loss_sum": 1.0, "loss_count": 2},
        reflect_padding_adapter=_reflect_receipt(),
    )
    publication = publish_full_checkpoint(tmp_path, checkpoint, epoch_end=False)
    resumed_model, resumed_optimizer, resumed_scheduler = _tiny_state()
    loaded = load_full_checkpoint(
        publication["checkpoint_path"],
        expected_sha256=publication["sha256"],
        expected_bindings=bindings,
        records=records,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device=torch.device("cpu"),
        epochs=2,
    )
    assert loaded["sampler"] == cursor.as_dict()
    assert loaded["provenance"] == checkpoint_provenance
    bad_sha_model, bad_sha_optimizer, bad_sha_scheduler = _tiny_state()
    with pytest.raises(ValueError, match="SHA-256"):
        load_full_checkpoint(
            publication["checkpoint_path"],
            expected_sha256="0" * 64,
            expected_bindings=bindings,
            records=records,
            model=bad_sha_model,
            optimizer=bad_sha_optimizer,
            scheduler=bad_sha_scheduler,
            device=torch.device("cpu"),
            epochs=2,
        )
    wrong = dict(bindings, gpu_uuid="GPU-other")
    wrong_model, wrong_optimizer, wrong_scheduler = _tiny_state()
    with pytest.raises(ValueError, match="binding"):
        load_full_checkpoint(
            publication["checkpoint_path"],
            expected_sha256=publication["sha256"],
            expected_bindings=wrong,
            records=records,
            model=wrong_model,
            optimizer=wrong_optimizer,
            scheduler=wrong_scheduler,
            device=torch.device("cpu"),
            epochs=2,
        )


def test_resume_is_fresh_cpu_staged_then_moves_model_and_all_optimizer_tensors(tmp_path):
    records = (
        SimpleNamespace(name="a_acc8.h5", slices=2),
        SimpleNamespace(name="b_acc8.h5", slices=2),
    )
    source_model, source_optimizer, source_scheduler = _tiny_state()
    _advance_tiny_state(source_model, source_optimizer, source_scheduler, 2)
    cursor = FullSamplerCursor(1, 1, 2)
    bindings = {
        "source_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
        "gpu_uuid": "GPU-exact",
        "reflect_padding_adapter": _reflect_receipt(),
    }
    checkpoint = build_full_checkpoint(
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        cursor=cursor,
        records=records,
        bindings=bindings,
        transactions=_exact_transactions(records, cursor),
        metrics={"loss_sum": 1.0, "loss_count": 2},
        reflect_padding_adapter=_reflect_receipt(),
    )
    publication = publish_full_checkpoint(tmp_path, checkpoint, epoch_end=False)

    model, optimizer, scheduler = _tiny_state()
    real_deepcopy = copy.deepcopy

    def forbid_live_training_copy(value, *args, **kwargs):
        if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)) or (
            isinstance(value, tuple)
            and any(isinstance(item, (torch.nn.Module, torch.optim.Optimizer)) for item in value)
        ):
            raise AssertionError("resume must not deepcopy live training objects")
        return real_deepcopy(value, *args, **kwargs)

    with patch.object(
        full_training_module.copy,
        "deepcopy",
        side_effect=forbid_live_training_copy,
    ):
        load_full_checkpoint(
            publication["checkpoint_path"],
            expected_sha256=publication["sha256"],
            expected_bindings=bindings,
            records=records,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("meta"),
            epochs=2,
        )
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    optimizer_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    ]
    assert optimizer_tensors
    assert all(value.device.type == "meta" for value in optimizer_tensors)

    dirty_model, dirty_optimizer, dirty_scheduler = _tiny_state()
    _advance_tiny_state(dirty_model, dirty_optimizer, dirty_scheduler, 1)
    before = snapshot_bytes(
        {
            "model": dirty_model.state_dict(),
            "optimizer": dirty_optimizer.state_dict(),
            "scheduler": dirty_scheduler.state_dict(),
        }
    )
    with pytest.raises(ValueError, match="fresh"):
        load_full_checkpoint(
            publication["checkpoint_path"],
            expected_sha256=publication["sha256"],
            expected_bindings=bindings,
            records=records,
            model=dirty_model,
            optimizer=dirty_optimizer,
            scheduler=dirty_scheduler,
            device=torch.device("cpu"),
            epochs=2,
        )
    assert snapshot_bytes(
        {
            "model": dirty_model.state_dict(),
            "optimizer": dirty_optimizer.state_dict(),
            "scheduler": dirty_scheduler.state_dict(),
        }
    ) == before


def test_checkpoint_publication_rejects_duplicate_epoch_cursor_without_mutation(tmp_path):
    model, optimizer, scheduler = _tiny_state()
    _advance_tiny_state(model, optimizer, scheduler, 1)
    records = (SimpleNamespace(name="a_acc8.h5", slices=1),)
    cursor = FullSamplerCursor(2, 0, 1)
    checkpoint = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=cursor,
        records=records,
        bindings={
            "source_sha256": "1" * 64,
            "data_manifest_sha256": "2" * 64,
            "recipe_sha256": "3" * 64,
            "gpu_uuid": "GPU-exact",
            "reflect_padding_adapter": _reflect_receipt(),
        },
        transactions=_exact_transactions(records, cursor),
        metrics={"loss_sum": 1.0, "loss_count": 1},
        reflect_padding_adapter=_reflect_receipt(),
    )
    checkpoint["sampler"]["global_step"] = FI_ACC8_FULL_RECIPE.slices_per_epoch
    publish_full_checkpoint(tmp_path, checkpoint, epoch_end=True)
    pointer_before = (tmp_path / "checkpoint-current.json").read_bytes()
    generations_before = {
        path.name for path in (tmp_path / "checkpoint-generations").iterdir()
    }

    with pytest.raises(ValueError, match="already retained"):
        publish_full_checkpoint(tmp_path, checkpoint, epoch_end=True)

    assert (tmp_path / "checkpoint-current.json").read_bytes() == pointer_before
    assert {
        path.name for path in (tmp_path / "checkpoint-generations").iterdir()
    } == generations_before


def test_checkpoint_publication_retains_latest_previous_and_epoch_generations(tmp_path):
    model, optimizer, scheduler = _tiny_state()
    bindings = {
        "source_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
        "gpu_uuid": "GPU-exact",
        "reflect_padding_adapter": _reflect_receipt(),
    }
    records = tuple(
        SimpleNamespace(name=f"{name}_acc8.h5", slices=1) for name in ("a", "b", "c")
    )
    publications = (
        (FullSamplerCursor(1, 1, 1), False),
        (FullSamplerCursor(1, 2, 2), False),
        (FullSamplerCursor(2, 0, 3), True),
        (FullSamplerCursor(2, 1, 4), False),
    )
    for index, (cursor, epoch_end) in enumerate(publications, 1):
        _advance_tiny_state(model, optimizer, scheduler, 1)
        checkpoint = build_full_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=cursor,
            records=records,
            bindings=bindings,
            transactions=_exact_transactions(records, cursor),
            metrics={"loss_sum": float(index), "loss_count": index},
            reflect_padding_adapter=_reflect_receipt(),
        )
        if epoch_end:
            checkpoint["sampler"]["global_step"] = FI_ACC8_FULL_RECIPE.slices_per_epoch
        publish_full_checkpoint(tmp_path, checkpoint, epoch_end=epoch_end)
    pointer = json.loads((tmp_path / "checkpoint-current.json").read_text())
    generations = sorted((tmp_path / "checkpoint-generations").iterdir())
    retained = {pointer["latest"]["generation"], pointer["previous"]["generation"]}
    retained.add(pointer["epoch_generations"][0]["generation"])
    assert {path.name for path in generations} == retained
    assert all((path / "checkpoint.pt").stat().st_mode & 0o222 == 0 for path in generations)


def test_repeated_latest_previous_resume_publications_stay_within_32_generations(tmp_path):
    model, optimizer, scheduler = _tiny_state()
    records = (SimpleNamespace(name="a_acc8.h5", slices=1),)
    bindings = {
        "source_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
        "gpu_uuid": "GPU-exact",
        "reflect_padding_adapter": _reflect_receipt(),
    }

    for completed_epoch in range(1, FI_ACC8_FULL_RECIPE.base_epochs + 1):
        _advance_tiny_state(model, optimizer, scheduler, 1)
        cursor = FullSamplerCursor(completed_epoch + 1, 0, completed_epoch)
        checkpoint = build_full_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=cursor,
            records=records,
            bindings=bindings,
            transactions=_exact_transactions(records, cursor),
            metrics={
                "loss_sum": float(completed_epoch),
                "loss_count": completed_epoch,
            },
            reflect_padding_adapter=_reflect_receipt(),
        )
        checkpoint["sampler"]["global_step"] = (
            completed_epoch * FI_ACC8_FULL_RECIPE.slices_per_epoch
        )
        publish_full_checkpoint(tmp_path, checkpoint, epoch_end=True)

        pointer = json.loads((tmp_path / "checkpoint-current.json").read_text())
        latest = pointer["latest"]
        latest_path = (
            tmp_path
            / "checkpoint-generations"
            / latest["generation"]
            / "checkpoint.pt"
        )
        assert full_training_module._check_resume_checkpoint_reference(
            tmp_path, latest_path, latest["checkpoint_sha256"]
        ) == latest_path.absolute()
        publish_full_checkpoint(tmp_path, checkpoint, epoch_end=False)

        pointer = json.loads((tmp_path / "checkpoint-current.json").read_text())
        for entry in (pointer["latest"], pointer["previous"]):
            path = (
                tmp_path
                / "checkpoint-generations"
                / entry["generation"]
                / "checkpoint.pt"
            )
            assert full_training_module._check_resume_checkpoint_reference(
                tmp_path, path, entry["checkpoint_sha256"]
            ) == path.absolute()
        publish_full_checkpoint(tmp_path, checkpoint, epoch_end=False)

        pointer = json.loads((tmp_path / "checkpoint-current.json").read_text())
        epoch_entries = pointer["epoch_generations"]
        epoch_cursors = [entry["sampler"]["epoch"] for entry in epoch_entries]
        retained = {
            pointer["latest"]["generation"],
            pointer["previous"]["generation"],
            *(entry["generation"] for entry in epoch_entries),
        }
        physical = {
            path.name for path in (tmp_path / "checkpoint-generations").iterdir()
        }
        assert len(epoch_entries) == completed_epoch
        assert len(epoch_cursors) == len(set(epoch_cursors)) <= 30
        assert physical == retained
        assert len(retained) <= full_training_module.FI_ACC8_RETAINED_CHECKPOINT_LIMIT

    final_pointer = json.loads((tmp_path / "checkpoint-current.json").read_text())
    final_retained = {
        final_pointer["latest"]["generation"],
        final_pointer["previous"]["generation"],
        *(entry["generation"] for entry in final_pointer["epoch_generations"]),
    }
    assert len(final_pointer["epoch_generations"]) == 30
    assert len(final_retained) == 32


def test_checkpoint_pointer_crash_preserves_old_authority_and_next_commit_prunes_orphan(tmp_path):
    model, optimizer, scheduler = _tiny_state()
    bindings = {
        "source_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "recipe_sha256": "3" * 64,
        "gpu_uuid": "GPU-exact",
        "reflect_padding_adapter": _reflect_receipt(),
    }

    records = (SimpleNamespace(name="a_acc8.h5", slices=1),)

    def checkpoint(cursor):
        return build_full_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler,
            cursor=cursor, records=records, bindings=bindings, transactions=[],
            metrics={"loss_sum": 0.0, "loss_count": 0},
            reflect_padding_adapter=_reflect_receipt(),
        )

    first = publish_full_checkpoint(
        tmp_path, checkpoint(FullSamplerCursor(1, 0, 0)), epoch_end=False
    )
    pointer_before = (tmp_path / "checkpoint-current.json").read_bytes()
    real_replace = full_training_module._atomic_replace_bytes

    def fail_pointer(path, payload):
        if Path(path).name == "checkpoint-current.json":
            raise OSError("simulated pointer crash")
        return real_replace(path, payload)

    with patch.object(full_training_module, "_atomic_replace_bytes", side_effect=fail_pointer):
        with pytest.raises(OSError, match="pointer crash"):
            publish_full_checkpoint(
                tmp_path, checkpoint(FullSamplerCursor(1, 0, 0)), epoch_end=False
            )
    assert (tmp_path / "checkpoint-current.json").read_bytes() == pointer_before
    assert first["checkpoint_path"].is_file()
    assert len(list((tmp_path / "checkpoint-generations").iterdir())) == 2

    second = publish_full_checkpoint(
        tmp_path, checkpoint(FullSamplerCursor(1, 0, 0)), epoch_end=False
    )
    assert second["checkpoint_path"].is_file()
    assert len(list((tmp_path / "checkpoint-generations").iterdir())) == 2


def test_fresh_output_collision_fails_before_inventory_gpu_or_cuda(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    existing = output / "keep"
    existing.write_bytes(b"unchanged")
    args = SimpleNamespace(
        fi_acc8_full_training=True,
        data_path_train=Path("/root/Data/train"),
        resume_checkpoint=None,
        resume_checkpoint_sha256=None,
        input_key="kspace", target_key="image_label", max_key="max",
        GPU_NUM=0, expected_gpu_uuid="GPU-exact",
    )
    with patch.object(
        full_training_module, "verify_pinned_upstream_sources", return_value={"commit": "x"}
    ), patch.object(full_training_module, "inspect_acc8_training_data") as inventory, patch.object(
        full_training_module, "preflight_smoke_gpu"
    ) as gpu, patch.object(full_training_module, "_select_smoke_device") as cuda:
        with pytest.raises(FileExistsError, match="already exists"):
            full_training_module.run_fi_acc8_full_training(args, output)
    inventory.assert_not_called()
    gpu.assert_not_called()
    cuda.assert_not_called()
    assert existing.read_bytes() == b"unchanged"


def test_pre_cuda_gates_finish_before_output_reservation_and_device_selection(tmp_path):
    output = tmp_path / "fresh"
    args = SimpleNamespace(
        fi_acc8_full_training=True,
        data_path_train=Path("/root/Data/train"),
        resume_checkpoint=None,
        resume_checkpoint_sha256=None,
        input_key="kspace", target_key="image_label", max_key="max",
        GPU_NUM=0, expected_gpu_uuid="GPU-exact",
    )
    manifest = SimpleNamespace(manifest_sha256="2" * 64, records=())
    events = []

    def event(name, value):
        def invoke(*args, **kwargs):
            del args, kwargs
            events.append(name)
            return value
        return invoke

    with patch.object(
        full_training_module, "verify_pinned_upstream_sources",
        side_effect=event("source", {"commit": "x"}),
    ), patch.object(
        full_training_module, "inspect_acc8_training_data",
        side_effect=event("inventory", manifest),
    ), patch.object(
        full_training_module, "preflight_smoke_gpu",
        side_effect=event("gpu-preflight", {"uuid": "GPU-exact"}),
    ), patch.object(
        full_training_module, "preflight_full_training_resources",
        side_effect=event("resources", {"schema": "resources"}),
    ), patch.object(
        full_training_module, "configure_determinism_pre_cuda",
        side_effect=event("determinism", dict(full_training_module.FI_ACC8_DETERMINISM_CONTRACT)),
    ), patch.object(
        full_training_module, "_build_full_training_model_with_adapters",
        side_effect=event(
            "adapters",
            (None, _reflect_receipt(), dict(full_training_module.FI_ACTIVATION_CHECKPOINT_CONTRACT)),
        ),
    ), patch.object(
        full_training_module, "_prepare_full_run_root",
        side_effect=event("reserve", output),
    ), patch.object(
        full_training_module, "_manifest_provenance", return_value={"manifest": "exact"}
    ), patch.object(
        full_training_module, "_write_file_fsync", return_value=None
    ), patch.object(
        full_training_module, "_select_smoke_device",
        side_effect=RuntimeError("stop after ordering proof"),
    ):
        with pytest.raises(RuntimeError, match="ordering proof"):
            full_training_module.run_fi_acc8_full_training(args, output)

    assert events == [
        "source", "inventory", "gpu-preflight", "resources", "determinism",
        "adapters", "reserve",
    ]


def test_resource_gate_failure_strands_no_zero_checkpoint_run_root(tmp_path):
    output = tmp_path / "fresh"
    args = SimpleNamespace(
        fi_acc8_full_training=True,
        data_path_train=Path("/root/Data/train"),
        resume_checkpoint=None,
        resume_checkpoint_sha256=None,
        input_key="kspace", target_key="image_label", max_key="max",
        GPU_NUM=0, expected_gpu_uuid="GPU-exact",
    )
    manifest = SimpleNamespace(manifest_sha256="2" * 64, records=())
    with patch.object(
        full_training_module, "verify_pinned_upstream_sources", return_value={"commit": "x"}
    ), patch.object(
        full_training_module, "inspect_acc8_training_data", return_value=manifest
    ), patch.object(
        full_training_module, "preflight_smoke_gpu", return_value={"uuid": "GPU-exact"}
    ), patch.object(
        full_training_module, "preflight_full_training_resources",
        side_effect=RuntimeError("Insufficient available RAM"),
    ), patch.object(full_training_module, "_select_smoke_device") as select:
        with pytest.raises(RuntimeError, match="RAM"):
            full_training_module.run_fi_acc8_full_training(args, output)
    select.assert_not_called()
    assert not output.exists()


def test_status_update_is_atomic_closed_and_five_minute_friendly(tmp_path):
    status_path = tmp_path / "status.json"
    status = {
        "schema": "fi-varnet-acc8-full-training-status-v2",
        "authoritative": False,
        "phase": "checkpointed",
        "pid": os.getpid(),
        "gpu_uuid": "GPU-exact",
        "gpu_name": "GeForce GTX 1080",
        "gpu_index": 0,
        "vram_allocated_bytes": 10,
        "vram_reserved_bytes": 20,
        "vram_peak_bytes": 30,
        "epoch": 1,
        "file_cursor": 1,
        "file": "a_acc8.h5",
        "slice": 1,
        "global_step": 1,
        "nominal_lr": 1e-8,
        "moving_loss": 0.5,
        "throughput_steps_per_second": 0.25,
        "eta_seconds": 12.5,
        "last_checkpoint_path": str(
            (tmp_path / "generation-a" / "checkpoint.pt").absolute()
        ),
        "last_checkpoint_sha256": "a" * 64,
        "command_argv": ["python", "train.py", "--fi-acc8-full-training"],
        "deterministic_contract": dict(
            full_training_module.FI_ACC8_DETERMINISM_CONTRACT
        ),
        "reflect_padding_adapter": _reflect_receipt(),
        "updated_unix_seconds": 1.0,
    }
    atomic_write_status(status_path, status)
    assert json.loads(status_path.read_text()) == status
    previous = status_path.read_bytes()
    invalid = dict(status, global_step=True)
    with pytest.raises(ValueError, match="status"):
        atomic_write_status(status_path, invalid)
    malformed_receipt = copy.deepcopy(status)
    malformed_receipt["reflect_padding_adapter"]["version"] = "other"
    with pytest.raises(ValueError, match="reflect-padding adapter receipt"):
        atomic_write_status(status_path, malformed_receipt)
    assert status_path.read_bytes() == previous
    assert not list(tmp_path.glob(".status-unpublished-*"))


def test_status_payload_uses_explicit_selected_device_runtime_without_cuda_probe(tmp_path):
    cursor = FullSamplerCursor(2, 3, 100)
    gpu = {"uuid": "GPU-exact", "name": "GeForce GTX 1080", "index": 0}
    runtime = {"allocated_bytes": 11, "reserved_bytes": 22, "peak_bytes": 33}
    checkpoint = (
        tmp_path / "checkpoint-generations" / "generation-a" / "checkpoint.pt"
    ).absolute()
    with patch.object(
        torch.cuda, "is_available", side_effect=AssertionError("forbidden status probe")
    ):
        status = full_training_module._status_payload(
            cursor,
            phase="training",
            gpu=gpu,
            device_runtime=runtime,
            file_name="brain_acc8.h5",
            slice_index=7,
            started=10.0,
            now_monotonic=20.0,
            starting_global_step=90,
            nominal_lr=2e-8,
            moving_loss=0.125,
            checkpoint_path=checkpoint,
            checkpoint_sha256="b" * 64,
            command_argv=["python", "train.py"],
            reflect_padding_adapter=_reflect_receipt(),
            updated_unix_seconds=30.0,
        )
    assert set(status) == full_training_module._STATUS_KEYS
    assert status["authoritative"] is False
    assert status["gpu_uuid"] == "GPU-exact"
    assert status["gpu_name"] == "GeForce GTX 1080"
    assert status["gpu_index"] == 0
    assert status["vram_allocated_bytes"] == 11
    assert status["vram_reserved_bytes"] == 22
    assert status["vram_peak_bytes"] == 33
    assert status["file_cursor"] == 3
    assert status["throughput_steps_per_second"] == 1.0
    assert status["eta_seconds"] == 69350.0
    assert status["last_checkpoint_path"] == str(checkpoint)
    assert status["last_checkpoint_sha256"] == "b" * 64
    assert (
        status["deterministic_contract"]
        == full_training_module.FI_ACC8_DETERMINISM_CONTRACT
    )
    assert status["reflect_padding_adapter"] == _reflect_receipt()


def test_run_provenance_and_summary_require_exact_reflect_adapter_receipt(tmp_path):
    receipt = _reflect_receipt()
    provenance = {
        "schema": "fi-varnet-acc8-full-training-provenance-v2",
        "source": {"commit": "exact"},
        "data": {"manifest_sha256": "2" * 64},
        "recipe": FI_ACC8_FULL_RECIPE.as_dict(),
        "gpu_preflight": {"uuid": "GPU-exact"},
        "scope": {"training": True, "evaluation": False, "submission": False},
        "reflect_padding_adapter": receipt,
    }
    assert full_training_module.validate_run_provenance(provenance) is provenance

    summary = {
        "schema": "fi-varnet-acc8-full-training-summary-v2",
        "namespace": FI_ACC8_FULL_NAMESPACE,
        "scope": "FULL_TRAINING_ONLY",
        "training_complete": True,
        "evaluation_authorized": False,
        "submission_authorized": False,
        "completed_epoch": 30,
        "global_step": 69450,
        "optimizer_steps": 69450,
        "scheduler_steps": 69450,
        "file_transactions": 2550,
        "loss_sum": 1.0,
        "loss_count": 69450,
        "mean_loss": 1.0 / 69450,
        "last_checkpoint": str((tmp_path / "checkpoint.pt").absolute()),
        "last_checkpoint_sha256": "a" * 64,
        "bindings": _bindings(),
        "reflect_padding_adapter": receipt,
        "pid": os.getpid(),
        "gpu_uuid": "GPU-exact",
        "peak_vram_bytes": 1,
        "elapsed_seconds": 1.0,
    }
    assert full_training_module.validate_full_training_summary(summary) is summary

    for artifact, validator in (
        (provenance, full_training_module.validate_run_provenance),
        (summary, full_training_module.validate_full_training_summary),
    ):
        malformed = copy.deepcopy(artifact)
        malformed["reflect_padding_adapter"]["version"] = "other"
        with pytest.raises(ValueError, match="reflect-padding adapter receipt"):
            validator(malformed)


@pytest.mark.parametrize("bad_entry", ["unknown", "symlink", "directory"])
def test_retired_generation_rejects_everything_except_two_regular_artifacts(
    tmp_path, bad_entry
):
    retired = tmp_path / ".retired-test"
    retired.mkdir()
    (retired / "checkpoint.pt").write_bytes(b"checkpoint")
    (retired / "metadata.json").write_bytes(b"metadata")
    if bad_entry == "unknown":
        (retired / "extra").write_bytes(b"extra")
    elif bad_entry == "symlink":
        (retired / "checkpoint.pt").unlink()
        (retired / "checkpoint.pt").symlink_to("metadata.json")
    else:
        (retired / "checkpoint.pt").unlink()
        (retired / "checkpoint.pt").mkdir()

    directory_fd = os.open(retired, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="generation"):
            full_training_module._purge_directory_fd(directory_fd)
    finally:
        os.close(directory_fd)
    assert (retired / "metadata.json").read_bytes() == b"metadata"


def test_retired_generation_cleanup_never_unlinks_a_racing_replacement(tmp_path):
    retired = tmp_path / ".retired-test"
    retired.mkdir()
    (retired / "checkpoint.pt").write_bytes(b"owned-checkpoint")
    (retired / "metadata.json").write_bytes(b"metadata")
    directory_fd = os.open(retired, os.O_RDONLY | os.O_DIRECTORY)
    real_rename = os.rename
    injected = False

    def race_rename(src, dst, *args, **kwargs):
        nonlocal injected
        if src == "checkpoint.pt" and str(dst).startswith(".delete-") and not injected:
            injected = True
            real_rename(
                "checkpoint.pt",
                "owned-checkpoint-preserved",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            replacement_fd = os.open(
                "checkpoint.pt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(replacement_fd, "wb") as replacement:
                replacement.write(b"attacker-replacement")
        return real_rename(src, dst, *args, **kwargs)

    try:
        with patch.object(full_training_module.os, "rename", side_effect=race_rename):
            with pytest.raises(RuntimeError, match="identity"):
                full_training_module._purge_directory_fd(directory_fd)
    finally:
        os.close(directory_fd)
    assert (retired / "owned-checkpoint-preserved").read_bytes() == b"owned-checkpoint"
    private_aliases = list(retired.glob(".delete-*"))
    assert len(private_aliases) == 1
    assert private_aliases[0].read_bytes() == b"attacker-replacement"


def test_atomic_replace_failure_cleanup_preserves_precheck_racing_replacement(tmp_path):
    destination = tmp_path / "status.json"
    staged_original = tmp_path / "staged-original-preserved"

    def fail_after_replacement(src, dst, *, src_dir_fd, dst_dir_fd):
        assert dst == destination.name
        os.rename(
            src,
            staged_original.name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        replacement_fd = os.open(
            src,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=src_dir_fd,
        )
        with os.fdopen(replacement_fd, "wb") as replacement:
            replacement.write(b"racing-replacement")
        raise OSError("injected replace failure")

    with patch.object(
        full_training_module.os, "replace", side_effect=fail_after_replacement
    ):
        with pytest.raises(OSError, match="injected replace failure") as error:
            full_training_module._atomic_replace_bytes(destination, b"owned-payload")
    assert staged_original.read_bytes() == b"owned-payload"
    unpublished = list(tmp_path.glob(".status.json-unpublished-*"))
    assert len(unpublished) == 1
    assert unpublished[0].read_bytes() == b"racing-replacement"
    assert any("preserved" in note for note in getattr(error.value, "__notes__", []))
    assert not destination.exists()


def test_atomic_replace_failure_cleanup_preserves_postcheck_racing_alias(tmp_path):
    destination = tmp_path / "status.json"
    real_rename = os.rename
    injected = False

    def fail_replace(*args, **kwargs):
        raise OSError("injected replace failure")

    def race_cleanup_rename(src, dst, *args, **kwargs):
        nonlocal injected
        if str(dst).startswith(".cleanup-") and not injected:
            injected = True
            directory_fd = kwargs["src_dir_fd"]
            real_rename(
                src,
                "owned-stage-preserved",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            replacement_fd = os.open(
                src,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(replacement_fd, "wb") as replacement:
                replacement.write(b"postcheck-replacement")
        return real_rename(src, dst, *args, **kwargs)

    with patch.object(
        full_training_module.os, "replace", side_effect=fail_replace
    ), patch.object(full_training_module.os, "rename", side_effect=race_cleanup_rename):
        with pytest.raises(OSError, match="injected replace failure") as error:
            full_training_module._atomic_replace_bytes(destination, b"owned-payload")
    assert (tmp_path / "owned-stage-preserved").read_bytes() == b"owned-payload"
    aliases = list(tmp_path.glob(".cleanup-*"))
    assert len(aliases) == 1
    assert aliases[0].read_bytes() == b"postcheck-replacement"
    assert any("preserved" in note for note in getattr(error.value, "__notes__", []))


def test_resume_reference_accepts_only_matching_latest_and_previous(tmp_path):
    sampler = FullSamplerCursor(2, 1, 4).as_dict()
    latest, latest_path = _write_resume_generation(
        tmp_path, "generation-latest", "a" * 64, sampler
    )
    previous, previous_path = _write_resume_generation(
        tmp_path, "generation-previous", "b" * 64, sampler
    )
    pointer = {
        "format_version": 1,
        "latest": latest,
        "previous": previous,
        "epoch_generations": [],
    }
    (tmp_path / "checkpoint-current.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )

    assert full_training_module._check_resume_checkpoint_reference(
        tmp_path, latest_path, latest["checkpoint_sha256"]
    ) == latest_path.absolute()
    assert full_training_module._check_resume_checkpoint_reference(
        tmp_path, previous_path, previous["checkpoint_sha256"]
    ) == previous_path.absolute()


def test_checkpoint_pointer_accepts_canonical_30_plus_latest_previous(tmp_path):
    pointer = _canonical_pointer()
    full_training_module._validate_checkpoint_pointer(pointer)
    (tmp_path / "checkpoint-current.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )

    loaded = full_training_module._load_pointer(tmp_path)
    assert loaded is not None
    retained = {
        loaded["latest"]["generation"],
        loaded["previous"]["generation"],
        *(entry["generation"] for entry in loaded["epoch_generations"]),
    }
    assert loaded == pointer
    assert len(loaded["epoch_generations"]) == 30
    assert len(retained) == full_training_module.FI_ACC8_RETAINED_CHECKPOINT_LIMIT == 32


def test_checkpoint_pointer_allows_current_entries_to_reference_epoch_generations():
    pointer = _canonical_pointer()
    pointer["latest"] = copy.deepcopy(pointer["epoch_generations"][-1])
    pointer["previous"] = copy.deepcopy(pointer["epoch_generations"][-2])

    assert full_training_module._validate_checkpoint_pointer(pointer) is pointer


@pytest.mark.parametrize(
    "malformation",
    [
        "top-level-extra",
        "version-bool",
        "latest-none",
        "entry-extra",
        "generation-shape",
        "sha-shape",
        "sampler-missing-key",
        "sampler-bool",
        "sampler-boundary",
    ],
)
def test_checkpoint_pointer_rejects_malformed_schema_entry_and_sampler(malformation):
    pointer = _canonical_pointer()
    if malformation == "top-level-extra":
        pointer["extra"] = None
    elif malformation == "version-bool":
        pointer["format_version"] = True
    elif malformation == "latest-none":
        pointer["latest"] = None
    elif malformation == "entry-extra":
        pointer["latest"]["extra"] = None
    elif malformation == "generation-shape":
        pointer["latest"]["generation"] = "generation-" + "A" * 32
    elif malformation == "sha-shape":
        pointer["latest"]["checkpoint_sha256"] = "A" * 64
    elif malformation == "sampler-missing-key":
        pointer["latest"]["sampler"].pop("boundary")
    elif malformation == "sampler-bool":
        pointer["latest"]["sampler"]["global_step"] = True
    else:
        pointer["latest"]["sampler"]["boundary"] = "slice"

    with pytest.raises(ValueError, match="checkpoint pointer"):
        full_training_module._validate_checkpoint_pointer(pointer)


@pytest.mark.parametrize(
    "malformation",
    [
        "too-many",
        "file-cursor",
        "epoch-low",
        "epoch-high",
        "global-step",
        "out-of-order",
        "duplicate-epoch",
        "duplicate-generation",
        "same-current-generation",
        "conflicting-epoch-reference",
    ],
)
def test_checkpoint_pointer_rejects_invalid_epoch_lineage(malformation):
    pointer = _canonical_pointer()
    entries = pointer["epoch_generations"]
    if malformation == "too-many":
        entries.append(
            _pointer_entry(200, FullSamplerCursor(31, 0, 30 * 2315).as_dict())
        )
    elif malformation == "file-cursor":
        entries[0]["sampler"]["file_cursor"] = 1
    elif malformation == "epoch-low":
        entries[0]["sampler"]["epoch"] = 1
    elif malformation == "epoch-high":
        entries[-1]["sampler"]["epoch"] = 32
    elif malformation == "global-step":
        entries[0]["sampler"]["global_step"] += 1
    elif malformation == "out-of-order":
        entries[0], entries[1] = entries[1], entries[0]
    elif malformation == "duplicate-epoch":
        entries[1]["sampler"] = copy.deepcopy(entries[0]["sampler"])
    elif malformation == "duplicate-generation":
        entries[1]["generation"] = entries[0]["generation"]
    elif malformation == "same-current-generation":
        pointer["previous"]["generation"] = pointer["latest"]["generation"]
    else:
        pointer["latest"] = copy.deepcopy(entries[-1])
        pointer["latest"]["checkpoint_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="checkpoint pointer"):
        full_training_module._validate_checkpoint_pointer(pointer)


def test_resume_rejects_preexisting_pointer_with_31_epoch_entries(tmp_path):
    latest, latest_path = _write_resume_generation(
        tmp_path,
        "latest-valid",
        "a" * 64,
        FullSamplerCursor(1, 0, 0).as_dict(),
    )
    pointer = _canonical_pointer()
    pointer["latest"] = latest
    pointer["epoch_generations"].append(
        _pointer_entry(200, FullSamplerCursor(31, 0, 30 * 2315).as_dict())
    )
    (tmp_path / "checkpoint-current.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(ValueError, match="checkpoint pointer"):
        full_training_module._check_resume_checkpoint_reference(
            tmp_path, latest_path, latest["checkpoint_sha256"]
        )


def test_publication_rejects_preexisting_31_epoch_pointer_before_generation_creation(
    tmp_path,
):
    pointer = _canonical_pointer()
    pointer["epoch_generations"].append(
        _pointer_entry(200, FullSamplerCursor(31, 0, 30 * 2315).as_dict())
    )
    (tmp_path / "checkpoint-current.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )
    model, optimizer, scheduler = _tiny_state()
    records = (SimpleNamespace(name="a_acc8.h5", slices=1),)
    checkpoint = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=FullSamplerCursor(1, 0, 0),
        records=records,
        bindings={
            "source_sha256": "1" * 64,
            "data_manifest_sha256": "2" * 64,
            "recipe_sha256": "3" * 64,
            "gpu_uuid": "GPU-exact",
            "reflect_padding_adapter": _reflect_receipt(),
        },
        transactions=[],
        metrics={"loss_sum": 0.0, "loss_count": 0},
        reflect_padding_adapter=_reflect_receipt(),
    )

    with pytest.raises(ValueError, match="checkpoint pointer"):
        publish_full_checkpoint(tmp_path, checkpoint, epoch_end=False)

    generations = tmp_path / "checkpoint-generations"
    assert not generations.exists() or not list(generations.iterdir())


def test_publication_validates_constructed_pointer_before_atomic_switch(tmp_path):
    model, optimizer, scheduler = _tiny_state()
    records = (SimpleNamespace(name="a_acc8.h5", slices=1),)
    checkpoint = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=FullSamplerCursor(1, 0, 0),
        records=records,
        bindings={
            "source_sha256": "1" * 64,
            "data_manifest_sha256": "2" * 64,
            "recipe_sha256": "3" * 64,
            "gpu_uuid": "GPU-exact",
            "reflect_padding_adapter": _reflect_receipt(),
        },
        transactions=[],
        metrics={"loss_sum": 0.0, "loss_count": 0},
        reflect_padding_adapter=_reflect_receipt(),
    )

    with patch.object(
        full_training_module,
        "_validate_checkpoint_pointer",
        side_effect=ValueError("invalid constructed checkpoint pointer"),
    ) as validate, patch.object(full_training_module, "_atomic_replace_bytes") as replace:
        with pytest.raises(ValueError, match="constructed checkpoint pointer"):
            publish_full_checkpoint(tmp_path, checkpoint, epoch_end=False)

    validate.assert_called_once()
    replace.assert_not_called()


def test_resume_reference_rejects_latest_with_mismatched_generation_metadata_sampler(tmp_path):
    pointer_sampler = FullSamplerCursor(2, 1, 4).as_dict()
    latest, latest_path = _write_resume_generation(
        tmp_path, "generation-latest", "a" * 64, pointer_sampler
    )
    metadata_path = latest_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["sampler"] = FullSamplerCursor(2, 0, 3).as_dict()
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    )
    pointer = {
        "format_version": 1,
        "latest": latest,
        "previous": None,
        "epoch_generations": [],
    }
    (tmp_path / "checkpoint-current.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(ValueError, match="generation/SHA/sampler"):
        full_training_module._check_resume_checkpoint_reference(
            tmp_path, latest_path, latest["checkpoint_sha256"]
        )


def test_resume_reference_rejects_latest_without_bound_sampler(tmp_path):
    latest, latest_path = _write_resume_generation(
        tmp_path,
        "generation-latest",
        "a" * 64,
        FullSamplerCursor(2, 1, 4).as_dict(),
    )
    latest.pop("sampler")
    metadata_path = latest_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("sampler")
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    )
    pointer = {
        "format_version": 1,
        "latest": latest,
        "previous": None,
        "epoch_generations": [],
    }
    (tmp_path / "checkpoint-current.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(ValueError, match="generation/SHA/sampler"):
        full_training_module._check_resume_checkpoint_reference(
            tmp_path, latest_path, latest["checkpoint_sha256"]
        )


@pytest.mark.parametrize("resume_source", ["epoch-only", "unlisted-generation"])
def test_resume_rejects_noncurrent_generation_before_source_inventory_cuda_or_mutation(
    tmp_path, resume_source
):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "INCOMPLETE").write_bytes(b"incomplete")
    sampler = FullSamplerCursor(2, 1, 4).as_dict()
    latest, _ = _write_resume_generation(
        output, "generation-latest", "a" * 64, sampler
    )
    previous, _ = _write_resume_generation(
        output, "generation-previous", "b" * 64, sampler
    )
    epoch_only, epoch_path = _write_resume_generation(
        output,
        "generation-old-epoch",
        "c" * 64,
        FullSamplerCursor(2, 0, FI_ACC8_FULL_RECIPE.slices_per_epoch).as_dict(),
    )
    unlisted, unlisted_path = _write_resume_generation(
        output, "generation-unlisted", "d" * 64, sampler
    )
    pointer = {
        "format_version": 1,
        "latest": latest,
        "previous": previous,
        "epoch_generations": [epoch_only],
    }
    (output / "checkpoint-current.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )
    checkpoint_path, sha256 = (
        (epoch_path, epoch_only["checkpoint_sha256"])
        if resume_source == "epoch-only"
        else (unlisted_path, unlisted["checkpoint_sha256"])
    )
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    with patch.object(full_training_module, "verify_pinned_upstream_sources") as source, patch.object(
        full_training_module, "inspect_acc8_training_data"
    ) as inventory, patch.object(full_training_module, "preflight_smoke_gpu") as gpu, patch.object(
        full_training_module, "_select_smoke_device"
    ) as cuda:
        with pytest.raises(ValueError, match="latest or previous"):
            full_training_module.run_fi_acc8_full_training(
                _resume_args(checkpoint_path, sha256), output
            )
    source.assert_not_called()
    inventory.assert_not_called()
    gpu.assert_not_called()
    cuda.assert_not_called()
    assert {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == before


def test_resume_reference_must_be_authoritative_in_root_before_inventory_or_cuda(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "INCOMPLETE").write_bytes(b"incomplete")
    args = SimpleNamespace(
        fi_acc8_full_training=True,
        data_path_train=Path("/root/Data/train"),
        resume_checkpoint=tmp_path / "outside" / "generation-x" / "checkpoint.pt",
        resume_checkpoint_sha256="a" * 64,
        input_key="kspace",
        target_key="image_label",
        max_key="max",
        GPU_NUM=0,
        expected_gpu_uuid="GPU-exact",
    )
    with patch.object(full_training_module, "verify_pinned_upstream_sources") as source, patch.object(
        full_training_module, "inspect_acc8_training_data"
    ) as inventory, patch.object(full_training_module, "preflight_smoke_gpu") as gpu, patch.object(
        full_training_module, "_select_smoke_device"
    ) as cuda:
        with pytest.raises(ValueError, match="generation in this run root"):
            full_training_module.run_fi_acc8_full_training(args, output)
    source.assert_not_called()
    inventory.assert_not_called()
    gpu.assert_not_called()
    cuda.assert_not_called()


def test_fresh_incomplete_root_without_checkpoint_is_a_collision_not_recovered(tmp_path):
    output = tmp_path / "incomplete"
    output.mkdir()
    marker = output / "INCOMPLETE"
    marker.write_bytes(b"forensic")
    args = SimpleNamespace(
        fi_acc8_full_training=True,
        data_path_train=Path("/root/Data/train"),
        resume_checkpoint=None,
        resume_checkpoint_sha256=None,
        input_key="kspace",
        target_key="image_label",
        max_key="max",
        GPU_NUM=0,
        expected_gpu_uuid="GPU-exact",
    )
    with patch.object(full_training_module, "inspect_acc8_training_data") as inventory:
        with pytest.raises(FileExistsError, match="already exists"):
            full_training_module.run_fi_acc8_full_training(args, output)
    inventory.assert_not_called()
    assert marker.read_bytes() == b"forensic"


def test_full_training_handoff_documents_non_authoritative_unlaunched_contract():
    handoff = Path("docs/fi_acc8_full_training_handoff.md").read_text()
    required = (
        "FI_ACC8_CKPT_ONE_STEP_ACTUAL_R7_PASS.json",
        "12d57839f2e34039f0e292d09fad810d1193a7c6aa82d33f5e9177cf495dc9f3",
        "--fi-acc8-full-training",
        "LambdaLR step 0",
        "same retained file descriptors",
        "prior durable verified-file boundary",
        "30 unique immutable epoch-end generations",
        "32 retained checkpoint generations",
        "`latest` or `previous`",
        "never Q3 resume sources",
        "authoritative: false",
        "strict-hybrid Q5",
        "LAUNCH IS NOT AUTHORIZED",
        "CUDA split/resume",
    )
    for phrase in required:
        assert phrase in handoff
