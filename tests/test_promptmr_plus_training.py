import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

import train


class PromptMRPlusTrainingCliTests(unittest.TestCase):
    def test_promptmr_plus_cli_resolves_exact_pinned_recipe(self):
        with patch(
            "sys.argv",
            [
                "train.py",
                "--model-family",
                "promptmr-plus",
                "--one-step-smoke",
                "--no-register-experiment",
                "--require-cuda-device-name",
                "NVIDIA GeForce RTX 3090",
                "--net-name",
                "FEATURE_PROMPTMR_PLUS_TEST",
            ],
        ):
            args = train.parse()

        self.assertEqual(args.model_family, "promptmr-plus")
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.lr, 0.0001)
        self.assertEqual(args.promptmr_weight_decay, 0.01)
        self.assertEqual(args.promptmr_lr_step_size, 35)
        self.assertEqual(args.promptmr_lr_gamma, 0.1)
        self.assertEqual(args.promptmr_gradient_clip_norm, 0.01)
        self.assertEqual(args.promptmr_uniform_resolution, (384, 384))
        self.assertTrue(args.promptmr_use_checkpoint)
        self.assertTrue(args.promptmr_compute_sens_per_coil)
        self.assertEqual(args.precision, "fp32")
        self.assertTrue(args.one_step_smoke)
        self.assertTrue(args.no_register_experiment)

    def test_promptmr_plus_cli_rejects_recipe_mutations(self):
        invalid_arguments = (
            ["--batch-size", "2"],
            ["--lr", "0.001"],
            ["--seed", "431"],
            ["--input-key", "other_kspace"],
            ["--target-key", "annotations"],
            ["--max-key", "other_max"],
            ["--score-aligned-loss"],
        )
        for extra in invalid_arguments:
            with self.subTest(extra=extra), patch(
                "sys.argv",
                ["train.py", "--model-family", "promptmr-plus", *extra],
            ):
                with self.assertRaises(SystemExit):
                    train.parse()

    def test_one_step_smoke_is_promptmr_only_and_requires_no_registration(self):
        invalid_argv = (
            ["train.py", "--one-step-smoke", "--no-register-experiment"],
            ["train.py", "--model-family", "promptmr-plus", "--one-step-smoke"],
            [
                "train.py",
                "--model-family",
                "promptmr-plus",
                "--one-step-smoke",
                "--no-register-experiment",
                "--resume-checkpoint",
                "/tmp/model.pt",
            ],
            [
                "train.py",
                "--model-family",
                "promptmr-plus",
                "--one-step-smoke",
                "--no-register-experiment",
                "--require-cuda-device-name",
                "NVIDIA GeForce RTX 3090",
                "--net-name",
                "/tmp/escape",
            ],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv), patch("sys.argv", argv):
                with self.assertRaises(SystemExit):
                    train.parse()

    def test_promptmr_runtime_verifies_source_before_seed_mutation(self):
        events = []
        args = SimpleNamespace(model_family="promptmr-plus", seed=430)
        with patch(
            "utils.learning.promptmr_plus_training.load_promptmr_training_recipe",
            side_effect=lambda: events.append("verify"),
        ), patch.object(train, "seed_fix", side_effect=lambda seed: events.append("seed")):
            train.prepare_runtime(args)
        self.assertEqual(events, ["verify", "seed"])

    def test_promptmr_smoke_requires_exact_rtx3090_device_gate(self):
        with patch(
            "sys.argv",
            [
                "train.py",
                "--model-family",
                "promptmr-plus",
                "--one-step-smoke",
                "--no-register-experiment",
            ],
        ):
            with self.assertRaises(SystemExit):
                train.parse()

    def test_promptmr_smoke_requires_24_gib_class_cuda_capacity(self):
        from utils.learning.promptmr_plus_training import (
            validate_promptmr_cuda_capacity,
        )

        with patch(
            "utils.learning.promptmr_plus_training.torch.cuda.get_device_properties",
            return_value=SimpleNamespace(total_memory=22 * 1024**3),
        ):
            with self.assertRaises(RuntimeError):
                validate_promptmr_cuda_capacity(torch.device("cuda:0"))

        with patch(
            "utils.learning.promptmr_plus_training.torch.cuda.get_device_properties",
            return_value=SimpleNamespace(total_memory=23 * 1024**3),
        ):
            self.assertEqual(
                validate_promptmr_cuda_capacity(torch.device("cuda:0")),
                23 * 1024**3,
            )

    def test_promptmr_plus_recipe_is_verified_from_pinned_bytes(self):
        from utils.learning.promptmr_plus_training import load_promptmr_training_recipe

        recipe = load_promptmr_training_recipe()

        self.assertEqual(recipe["upstream_commit"], "934eeda6d4d18cd39e406fa1eee9e1f70603cb5e")
        self.assertEqual(recipe["optimizer"], "AdamW")
        self.assertEqual(recipe["learning_rate"], 0.0001)
        self.assertEqual(recipe["weight_decay"], 0.01)
        self.assertEqual(recipe["scheduler"], {"name": "StepLR", "step_size": 35, "gamma": 0.1})
        self.assertEqual(recipe["loss"], {"name": "SSIMLoss", "window": 7, "k1": 0.01, "k2": 0.03})
        self.assertEqual(recipe["uniform_resolution"], [384, 384])
        self.assertTrue(recipe["use_checkpoint"])
        self.assertTrue(recipe["compute_sens_per_coil"])

    def test_promptmr_plus_dataset_pairs_files_and_edge_replicates_five_slices(self):
        from utils.data.promptmr_plus import PromptMRPlusSliceData

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            (root / "kspace").mkdir()
            (root / "image").mkdir()
            name = "sample_acc4_volume.h5"
            kspace = np.stack(
                [
                    np.full((1, 384, 384), complex(index + 1, 0), np.complex64)
                    for index in range(3)
                ]
            )
            with h5py.File(root / "kspace" / name, "w") as handle:
                handle.create_dataset("kspace", data=kspace)
                handle.create_dataset(
                    "mask", data=(np.arange(384) % 2 == 0).astype(np.uint8)
                )
            with h5py.File(root / "image" / name, "w") as handle:
                handle.create_dataset(
                    "image_label", data=np.ones((3, 384, 384), dtype=np.float32)
                )
                handle.attrs["max"] = 1.0

            dataset = PromptMRPlusSliceData(root)
            sample = dataset[0]
            with h5py.File(root / "image" / name, "r+") as handle:
                handle["image_label"][0, 0, 0] = 2.0
            with self.assertRaisesRegex(ValueError, "source bytes changed"):
                _ = dataset[0]

        self.assertEqual(len(dataset), 3)
        self.assertRegex(dataset.inventory_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(sample.masked_kspace.shape, (5, 384, 384, 2))
        self.assertEqual(sample.mask.shape, (1, 1, 384, 1))
        self.assertEqual(sample.mask.dtype, torch.bool)
        self.assertEqual(sample.target.shape, (384, 384))
        self.assertEqual(sample.acceleration, 4)
        self.assertEqual(sample.fname, name)
        self.assertEqual(sample.slice_num, 0)
        self.assertEqual(
            sample.masked_kspace[:, 0, 0, 0].tolist(),
            [1.0, 1.0, 1.0, 2.0, 3.0],
        )
        self.assertEqual(sample.masked_kspace[:, 0, 1, 0].tolist(), [0.0] * 5)

    def test_promptmr_plus_dataset_rejects_unresolved_non384_mask_mapping(self):
        from utils.data.promptmr_plus import PromptMRPlusSliceData

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            (root / "kspace").mkdir()
            (root / "image").mkdir()
            name = "sample_acc8_volume.h5"
            with h5py.File(root / "kspace" / name, "w") as handle:
                handle.create_dataset(
                    "kspace",
                    data=np.ones((1, 1, 386, 400), dtype=np.complex64),
                )
                handle.create_dataset("mask", data=np.ones(400, dtype=np.uint8))
            with h5py.File(root / "image" / name, "w") as handle:
                handle.create_dataset(
                    "image_label", data=np.ones((1, 386, 400), dtype=np.float32)
                )
                handle.attrs["max"] = 1.0

            with self.assertRaisesRegex(ValueError, "mask-mapping policy"):
                PromptMRPlusSliceData(root)

    def test_promptmr_plus_dataset_rejects_nonfinite_max_value(self):
        from utils.data.promptmr_plus import PromptMRPlusSliceData

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            (root / "kspace").mkdir()
            (root / "image").mkdir()
            name = "sample_acc4_volume.h5"
            kspace = np.ones((1, 1, 384, 384), dtype=np.complex64)
            target = np.ones((1, 384, 384), dtype=np.float32)
            with h5py.File(root / "kspace" / name, "w") as handle:
                handle.create_dataset("kspace", data=kspace)
                handle.create_dataset("mask", data=np.ones(384, dtype=np.uint8))
            with h5py.File(root / "image" / name, "w") as handle:
                handle.create_dataset("image_label", data=target)
                handle.attrs["max"] = np.nan

            with self.assertRaises(ValueError):
                PromptMRPlusSliceData(root)

    def test_promptmr_plus_components_use_exact_optimizer_scheduler_and_loss(self):
        from utils.learning import promptmr_plus_training as promptmr_training
        from utils.model.promptmr_plus_adapter import PromptMRPlusAdapter

        core = torch.nn.Linear(1, 1)
        with patch(
            "sys.argv", ["train.py", "--model-family", "promptmr-plus"]
        ):
            args = train.parse()
        with patch.object(
            promptmr_training, "build_promptmr_plus", return_value=core
        ):
            components = promptmr_training.build_promptmr_training_components(
                args, torch.device("cpu")
            )

        self.assertIsInstance(components.model, PromptMRPlusAdapter)
        self.assertIs(components.model.core, core)
        self.assertIsInstance(components.optimizer, torch.optim.AdamW)
        self.assertEqual(components.optimizer.param_groups[0]["lr"], 0.0001)
        self.assertEqual(components.optimizer.param_groups[0]["weight_decay"], 0.01)
        self.assertIsInstance(components.scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(components.scheduler.step_size, 35)
        self.assertEqual(components.scheduler.gamma, 0.1)
        self.assertEqual(components.loss.win_size, 7)
        self.assertEqual(components.loss.k1, 0.01)
        self.assertEqual(components.loss.k2, 0.03)
        self.assertEqual(components.provenance["model_family"], "promptmr-plus")
        self.assertEqual(components.provenance["precision"], "fp32")
        self.assertFalse(components.provenance["tf32_allowed"])
        self.assertEqual(components.provenance["seed"], 430)

    def test_promptmr_plus_train_step_is_finite_clipped_and_exactly_one_step(self):
        from utils.learning import promptmr_plus_training as promptmr_training

        class FakePromptMR(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.8))
                self.calls = []

            def forward(self, prepared, **kwargs):
                self.calls.append(kwargs)
                return prepared.kspace[:, 0, :, :, 0] * self.weight

        model = FakePromptMR()
        loss_class = promptmr_training.import_promptmr_plus_module(
            "mri_utils.losses"
        ).SSIMLoss
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        components = SimpleNamespace(
            model=model,
            loss=loss_class(win_size=7, k1=0.01, k2=0.03),
            optimizer=optimizer,
        )
        args = SimpleNamespace(
            promptmr_uniform_resolution=(384, 384),
            promptmr_use_checkpoint=True,
            promptmr_compute_sens_per_coil=True,
            promptmr_gradient_clip_norm=0.01,
        )
        batch = SimpleNamespace(
            masked_kspace=torch.ones((1, 5, 8, 8, 2), dtype=torch.float32),
            mask=torch.ones((1, 1, 1, 8, 1), dtype=torch.bool),
            num_low_frequencies=torch.full((1,), -1, dtype=torch.int64),
            acceleration=torch.tensor([4]),
            target=torch.full((1, 8, 8), 0.5, dtype=torch.float32),
            max_value=torch.ones(1, dtype=torch.float32),
        )
        before = model.weight.detach().clone()

        result = promptmr_training.promptmr_train_step(args, components, batch)

        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(result["gradient_norm"])))
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertNotEqual(model.weight.detach(), before)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(
            model.calls,
            [
                {
                    "crop_size": (384, 384),
                    "use_checkpoint": True,
                    "compute_sens_per_coil": True,
                }
            ],
        )

    def test_train_py_promptmr_smoke_publishes_roundtrip_checkpoint_and_history(self):
        from utils.learning import promptmr_plus_training as promptmr_training

        class FakePromptMR(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.8))

            def forward(self, prepared, **kwargs):
                return prepared.kspace[:, 0, :, :, 0] * self.weight

        class FakeLoader(list):
            def __init__(self, items):
                super().__init__(items)
                self.dataset = SimpleNamespace(inventory_sha256="a" * 64)

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            run_dir = Path(tmp) / "FEATURE_PROMPTMR_PLUS_SMOKE"
            model = FakePromptMR()
            loss_class = promptmr_training.import_promptmr_plus_module(
                "mri_utils.losses"
            ).SSIMLoss
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0001, weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=35, gamma=0.1
            )
            components = SimpleNamespace(
                model=model,
                loss=loss_class(win_size=7, k1=0.01, k2=0.03),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                provenance={"model_family": "promptmr-plus", "seed": 430},
            )
            batch = SimpleNamespace(
                masked_kspace=torch.ones((1, 5, 8, 8, 2), dtype=torch.float32),
                mask=torch.ones((1, 1, 1, 8, 1), dtype=torch.bool),
                num_low_frequencies=torch.full((1,), -1, dtype=torch.int64),
                acceleration=torch.tensor([4]),
                target=torch.full((1, 8, 8), 0.5, dtype=torch.float32),
                max_value=torch.ones(1, dtype=torch.float32),
                fname=["sample_acc4_volume.h5"],
                slice_num=torch.tensor([0]),
            )
            fake_loader = FakeLoader([batch])
            args = SimpleNamespace(
                run_dir=run_dir,
                exp_dir=run_dir / "checkpoints",
                data_path_train=Path("/read-only/train"),
                promptmr_uniform_resolution=(384, 384),
                promptmr_use_checkpoint=True,
                promptmr_compute_sens_per_coil=True,
                promptmr_gradient_clip_norm=0.01,
            )
            with patch.object(
                promptmr_training,
                "build_promptmr_training_components",
                return_value=components,
            ), patch.object(
                promptmr_training,
                "create_promptmr_data_loader",
                return_value=fake_loader,
            ):
                report = promptmr_training.run_promptmr_one_step_smoke(
                    args, torch.device("cpu")
                )

            checkpoint_path = run_dir / "checkpoints" / "smoke_model.pt"
            history_path = run_dir / "smoke_history.json"
            report_path = run_dir / "smoke_report.json"
            self.assertTrue((run_dir / "RUN_COMPLETE.json").is_file())
            self.assertFalse((run_dir / "RUN_INCOMPLETE.json").exists())
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["checkpoint_roundtrip"], "PASS")
        self.assertEqual(report["telemetry"]["precision"], "fp32")
        self.assertIsNone(report["telemetry"]["peak_allocated_mib"])
        self.assertIsNone(report["telemetry"]["peak_reserved_mib"])
        self.assertIsNone(report["telemetry"]["process_gpu_memory_mib"])
        self.assertGreater(report["telemetry"]["peak_rss_mib"], 0)
        self.assertTrue(history_path.name.endswith(".json"))
        self.assertTrue(report_path.name.endswith(".json"))
        self.assertIn("scheduler", checkpoint)
        self.assertEqual(checkpoint["global_optimizer_step"], 1)
        self.assertEqual(checkpoint["model_family"], "promptmr-plus")
        self.assertIsNone(checkpoint["scaler"])
        self.assertEqual(checkpoint["provenance"]["model_family"], "promptmr-plus")
        self.assertEqual(checkpoint["provenance"]["train_inventory_sha256"], "a" * 64)
        self.assertEqual(checkpoint["history"][0]["optimizer_step"], 1)
        self.assertEqual(checkpoint["history"][0]["fname"], "sample_acc4_volume.h5")
        self.assertEqual(checkpoint["history"][0]["slice_num"], 0)
        self.assertEqual(checkpoint["history"][0]["acceleration"], 4)

    def test_checkpoint_comparison_uses_non_aliasing_cpu_snapshot(self):
        from utils.learning import promptmr_plus_training as promptmr_training

        cpu_calls = []

        class SimulatedCudaTensor(torch.Tensor):
            @staticmethod
            def __new__(cls):
                return torch.Tensor._make_subclass(cls, torch.tensor([3.0]))

            def cpu(self, *args, **kwargs):
                cpu_calls.append(True)
                return torch.tensor([3.0])

        source = {
            "tensor": torch.tensor([1.0]),
            "nested": [torch.tensor([2.0])],
            "simulated_cuda": SimulatedCudaTensor(),
        }
        snapshot = promptmr_training._cpu_state_snapshot(source)
        self.assertTrue(cpu_calls)
        self.assertTrue(promptmr_training._states_equal(source["tensor"], snapshot["tensor"]))
        self.assertEqual(snapshot["simulated_cuda"].device.type, "cpu")
        self.assertEqual(snapshot["tensor"].device.type, "cpu")
        self.assertNotEqual(source["tensor"].data_ptr(), snapshot["tensor"].data_ptr())

    def test_descriptor_publication_ignores_replaced_staging_path(self):
        from utils.learning import promptmr_plus_training as promptmr_training

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            directory_fd = os.open(tmp, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with promptmr_training._open_staged_file(
                    directory_fd, mode="w+b"
                ) as (handle, staging_name, _identity):
                    handle.write(b"trusted")
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.unlink(staging_name, dir_fd=directory_fd)
                    attacker_fd = os.open(
                        staging_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=directory_fd,
                    )
                    os.write(attacker_fd, b"attacker")
                    os.close(attacker_fd)
                    with self.assertRaises(OSError):
                        promptmr_training._publish_staged(
                            handle, directory_fd, "published.bin"
                        )
                self.assertFalse((Path(tmp) / "published.bin").exists())
            finally:
                os.close(directory_fd)

    def test_incomplete_marker_failure_does_not_publish_run_directory(self):
        from utils.learning import promptmr_plus_training as promptmr_training

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            run_dir = Path(tmp) / "FEATURE_PROMPTMR_PLUS_MARKER_FAIL"
            args = SimpleNamespace(
                run_dir=run_dir,
                exp_dir=run_dir / "checkpoints",
                data_path_train=Path("/unused"),
            )
            with patch.object(
                promptmr_training,
                "_write_json_exclusive",
                side_effect=OSError("marker publication failed"),
            ):
                with self.assertRaisesRegex(OSError, "marker publication failed"):
                    promptmr_training.run_promptmr_one_step_smoke(
                        args, torch.device("cpu")
                    )
            self.assertFalse(run_dir.exists())

    def test_promptmr_smoke_failure_leaves_durable_incomplete_marker(self):
        from utils.learning import promptmr_plus_training as promptmr_training

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            run_dir = Path(tmp) / "FEATURE_PROMPTMR_PLUS_FAILURE"
            args = SimpleNamespace(
                run_dir=run_dir,
                exp_dir=run_dir / "checkpoints",
                data_path_train=Path("/must-fail"),
            )
            with patch.object(
                promptmr_training,
                "build_promptmr_training_components",
                return_value=object(),
            ), patch.object(
                promptmr_training,
                "create_promptmr_data_loader",
                side_effect=RuntimeError("synthetic loader failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
                    promptmr_training.run_promptmr_one_step_smoke(
                        args, torch.device("cpu")
                    )
            self.assertTrue((run_dir / "RUN_INCOMPLETE.json").is_file())
            self.assertFalse((run_dir / "RUN_COMPLETE.json").exists())

    def test_promptmr_smoke_refuses_existing_run_before_model_or_data_access(self):
        from utils.learning import promptmr_plus_training as promptmr_training

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            run_dir = Path(tmp) / "FEATURE_PROMPTMR_PLUS_SMOKE"
            run_dir.mkdir()
            args = SimpleNamespace(
                run_dir=run_dir,
                exp_dir=run_dir / "checkpoints",
                data_path_train=Path("/must-not-be-read"),
            )
            with patch.object(
                promptmr_training, "build_promptmr_training_components"
            ) as build, patch.object(
                promptmr_training, "create_promptmr_data_loader"
            ) as loader:
                with self.assertRaises(FileExistsError):
                    promptmr_training.run_promptmr_one_step_smoke(
                        args, torch.device("cpu")
                    )

        build.assert_not_called()
        loader.assert_not_called()

    def test_train_part_routes_promptmr_smoke_without_constructing_varnet(self):
        import utils.learning.train_part as train_part

        args = SimpleNamespace(
            require_cuda_device_name=None,
            GPU_NUM=0,
            model_family="promptmr-plus",
            one_step_smoke=True,
        )
        with patch.object(
            train_part.torch.cuda, "is_available", return_value=False
        ), patch.object(
            train_part, "run_promptmr_one_step_smoke", return_value={"status": "PASS"}
        ) as smoke, patch.object(train_part, "VarNet") as varnet:
            result = train_part.train(args)

        self.assertEqual(result, {"status": "PASS"})
        smoke.assert_called_once_with(args, torch.device("cpu"))
        varnet.assert_not_called()

    def test_train_part_runs_promptmr_idle_preflight_before_cuda_identity(self):
        import utils.learning.train_part as train_part

        events = []
        args = SimpleNamespace(
            require_cuda_device_name="NVIDIA GeForce RTX 3090",
            GPU_NUM=0,
            model_family="promptmr-plus",
            one_step_smoke=True,
        )
        with patch.object(
            train_part.torch.cuda, "is_available", return_value=True
        ), patch.object(
            train_part.torch.cuda, "device_count", return_value=1
        ), patch.object(
            train_part,
            "preflight_promptmr_gpu",
            side_effect=lambda: events.append("preflight") or {"memory_used_mib": 0.0},
        ), patch.object(
            train_part.torch.cuda,
            "get_device_name",
            side_effect=lambda _: events.append("identity")
            or "NVIDIA GeForce RTX 3090",
        ), patch.object(
            train_part.torch.cuda, "set_device"
        ), patch.object(
            train_part.torch.cuda, "current_device", return_value=0
        ), patch.object(
            train_part, "run_promptmr_one_step_smoke", return_value={"status": "PASS"}
        ):
            train_part.train(args)

        self.assertEqual(events, ["preflight", "identity"])

    def test_train_py_leaves_promptmr_smoke_output_absent_until_guarded_runner(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp, patch(
            "sys.argv",
            [
                "train.py",
                "--model-family",
                "promptmr-plus",
                "--one-step-smoke",
                "--no-register-experiment",
                "--require-cuda-device-name",
                "NVIDIA GeForce RTX 3090",
                "--net-name",
                "FEATURE_PROMPTMR_PLUS_SMOKE",
            ],
        ):
            args = train.parse()
            train.configure_result_paths(args, Path(tmp))

            self.assertEqual(args.run_dir, Path(tmp) / args.net_name)
            self.assertEqual(args.exp_dir, args.run_dir / "checkpoints")
            self.assertFalse(args.run_dir.exists())
            self.assertFalse(args.exp_dir.exists())


if __name__ == "__main__":
    unittest.main()
