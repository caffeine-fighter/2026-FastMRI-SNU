import importlib.util
import csv
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from utils.learning.resume import build_training_state


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_val_epochs.py"


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("sweep_val_epochs_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_train_part_module(module):
    return importlib.import_module(module._seal_staged_directory.__module__)


def write_generation(checkpoint_dir, generation, epoch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with patch("utils.learning.resume.torch.cuda.is_available", return_value=False):
        state = build_training_state(epoch, model, optimizer, 0.1)
    path = checkpoint_dir / f".checkpoint-generation-{generation}-model.pt"
    torch.save(state, path)
    return path


def assert_evaluator_runner_contract(command, runner_kwargs):
    out_dir = Path(command[command.index("--out-dir") + 1])
    assert out_dir.parent == Path("/proc/self/fd")
    directory_fd = int(out_dir.name)
    assert runner_kwargs["pass_fds"] == (directory_fd,)
    assert runner_kwargs["check"] is True
    assert Path(runner_kwargs["cwd"]) == SCRIPT.parents[1]
    assert runner_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


def write_strict_metrics(command, quality_score, **runner_kwargs):
    assert_evaluator_runner_contract(command, runner_kwargs)
    target_dir = command[command.index("--target-dir") + 1]
    recon_dir = command[command.index("--recon-dir") + 1]
    out_dir = Path(command[command.index("--out-dir") + 1])
    volumes = int(command[command.index("--expected-volumes") + 1])
    slices = int(command[command.index("--expected-slices") + 1])
    boxes = int(command[command.index("--expected-boxes") + 1])
    if volumes % 2 or slices % 2 or boxes % 2:
        raise AssertionError("test strict metrics helper requires even counts")
    full_mean = quality_score + 0.05
    bbox_mean = quality_score - 0.05

    def scope_row(scope, scope_volumes, scope_slices, scope_boxes):
        return {
            "scope": scope,
            "ssim_full_mean": full_mean,
            "ssim_bbox_mean": bbox_mean,
            "ssim_full_count": scope_slices,
            "ssim_bbox_count": scope_boxes,
            "volumes": scope_volumes,
            "slices": scope_slices,
            "bbox_annotations": scope_boxes,
        }

    rows = [
        scope_row("overall", volumes, slices, boxes),
        scope_row("acc4", volumes // 2, slices // 2, boxes // 2),
        scope_row("acc8", volumes // 2, slices // 2, boxes // 2),
        {
            **scope_row("leaderboard_equal_acc", volumes, slices, boxes),
            "quality_score": quality_score,
            "aggregation": "equal mean of acc4 and acc8",
        },
        {
            **scope_row("unknown", 0, 0, 0),
            "ssim_full_mean": None,
            "ssim_bbox_mean": None,
        },
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "target_dir": target_dir,
                "recon_dir": recon_dir,
                "out_dir": str(out_dir),
                "rows": rows,
                "skipped": [],
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "skipped.json").write_text("[]", encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "ssim_full_mean",
                "ssim_bbox_mean",
                "quality_score",
                "aggregation",
                "ssim_full_count",
                "ssim_bbox_count",
                "volumes",
                "slices",
                "bbox_annotations",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


class ValidationEpochSweepTests(unittest.TestCase):
    def test_cli_requires_explicit_expected_epochs(self):
        module = load_sweep_module()
        argv = [
            "sweep_val_epochs.py",
            "--retained-root",
            "/tmp/retained",
            "--checkpoint-dir",
            "/tmp/checkpoints",
            "--target-dir",
            "/tmp/targets",
            "--out-dir",
            "/tmp/report",
            "--expected-epochs",
            "29",
            "30",
            "--expected-volumes",
            "20",
            "--expected-slices",
            "400",
            "--expected-boxes",
            "80",
        ]
        with patch("sys.argv", argv):
            args = module.parse_args()

        self.assertEqual(args.expected_epochs, [29, 30])
        self.assertEqual(args.expected_volumes, 20)
        self.assertEqual(args.expected_slices, 400)
        self.assertEqual(args.expected_boxes, 80)

    def test_maps_epoch_to_immutable_checkpoint_generation_by_state_epoch(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint_dir = Path(tmp)
            artifact = write_generation(checkpoint_dir, "a" * 32, 3)

            mapped = module.map_checkpoint_generations(checkpoint_dir, [3])

        self.assertEqual(mapped[3]["generation"], "a" * 32)
        self.assertEqual(mapped[3]["artifact"], str(artifact.resolve()))

    def test_checkpoint_mapping_rejects_missing_generation(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint_dir = Path(tmp)
            write_generation(checkpoint_dir, "a" * 32, 2)

            with self.assertRaisesRegex(ValueError, "epoch 3, found 0"):
                module.map_checkpoint_generations(checkpoint_dir, [3])

    def test_checkpoint_mapping_rejects_duplicate_epoch_generations(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint_dir = Path(tmp)
            write_generation(checkpoint_dir, "a" * 32, 3)
            write_generation(checkpoint_dir, "b" * 32, 3)

            with self.assertRaisesRegex(ValueError, "epoch 3, found 2"):
                module.map_checkpoint_generations(checkpoint_dir, [3])

    def test_retained_coverage_fails_closed_when_reconstruction_is_missing(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            target_dir.mkdir()
            (retained_root / "epoch_0001").mkdir(parents=True)
            (target_dir / "a.h5").touch()
            (target_dir / "b.h5").touch()
            (retained_root / "epoch_0001" / "a.h5").touch()

            with self.assertRaisesRegex(ValueError, "missing.*b.h5"):
                module.validate_retained_coverage(
                    retained_root, target_dir, [1]
                )

    def test_retained_coverage_rejects_unexpected_epoch_directory(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            target_dir.mkdir()
            (target_dir / "a.h5").touch()
            for epoch in (1, 2):
                epoch_dir = retained_root / f"epoch_{epoch:04d}"
                epoch_dir.mkdir(parents=True)
                (epoch_dir / "a.h5").touch()

            with self.assertRaisesRegex(ValueError, "unexpected.*2"):
                module.validate_retained_coverage(
                    retained_root, target_dir, [1]
                )

    def test_retained_coverage_rejects_symlinked_reconstruction(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            epoch_dir = retained_root / "epoch_0001"
            target_dir.mkdir()
            epoch_dir.mkdir(parents=True)
            (target_dir / "a.h5").touch()
            outside = root / "outside.h5"
            outside.touch()
            (epoch_dir / "a.h5").symlink_to(outside)

            with self.assertRaisesRegex(ValueError, r"regular \.h5 file"):
                module.validate_retained_coverage(
                    retained_root, target_dir, [1]
                )

    def test_retained_coverage_rejects_every_non_h5_entry(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            epoch_dir = retained_root / "epoch_0001"
            target_dir.mkdir()
            epoch_dir.mkdir(parents=True)
            (target_dir / "a.h5").touch()
            (epoch_dir / "a.h5").touch()
            (epoch_dir / "notes.txt").touch()

            with self.assertRaisesRegex(ValueError, r"regular \.h5 file.*notes.txt"):
                module.validate_retained_coverage(retained_root, target_dir, [1])

    def test_target_coverage_rejects_directory_even_when_named_h5(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            target_dir.mkdir()
            (target_dir / "not-a-file.h5").mkdir()
            (retained_root / "epoch_0001").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "regular .h5 file"):
                module.validate_retained_coverage(retained_root, target_dir, [1])

    def test_epoch_evaluation_refuses_stale_output_directory(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            out_dir = root / "metrics"
            out_dir.mkdir()
            stale = out_dir / "metrics.json"
            stale.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "scope": "leaderboard_equal_acc",
                                "quality_score": 0.9,
                            }
                        ],
                        "skipped": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "already exists"):
                module.evaluate_epoch(
                    target_dir=root / "targets",
                    recon_dir=root / "epoch_0001",
                    out_dir=out_dir,
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=lambda *args, **kwargs: None,
                )

            self.assertTrue(stale.is_file())

    def test_epoch_evaluation_real_subprocess_inherits_staging_fd_and_absolute_inputs(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            recon_dir = root / "recons"
            target_dir.mkdir()
            recon_dir.mkdir()
            annotations = json.dumps(
                {"0": [{"x": 0, "y": 0, "width": 12, "height": 12}]}
            )
            data = np.ones((1, 16, 16), dtype=np.float32)
            for name in ("brain_acc4.h5", "brain_acc8.h5"):
                with h5py.File(target_dir / name, "w") as handle:
                    handle.create_dataset("image_label", data=data)
                    handle.attrs["max"] = 1.0
                    handle.attrs["annotations"] = annotations
                with h5py.File(recon_dir / name, "w") as handle:
                    handle.create_dataset("reconstruction", data=data)

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                equal_acc = module.evaluate_epoch(
                    target_dir=Path("targets"),
                    recon_dir=Path("recons"),
                    out_dir=Path("nested") / "metrics",
                    expected_volumes=2,
                    expected_slices=2,
                    expected_boxes=2,
                )
            finally:
                os.chdir(previous_cwd)

            self.assertAlmostEqual(equal_acc["quality_score"], 1.0)
            self.assertTrue((root / "nested" / "metrics" / "metrics.json").is_file())

    def test_epoch_evaluation_removes_partial_output_after_subprocess_failure(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            out_dir = root / "metrics"

            def failing_runner(command, **kwargs):
                assert_evaluator_runner_contract(command, kwargs)
                metrics_dir = Path(command[command.index("--out-dir") + 1])
                metrics_dir.mkdir(parents=True, exist_ok=True)
                (metrics_dir / "metrics.json").write_text(
                    "partial", encoding="utf-8"
                )
                raise subprocess.CalledProcessError(2, command)

            with self.assertRaises(subprocess.CalledProcessError):
                module.evaluate_epoch(
                    target_dir=root / "targets",
                    recon_dir=root / "epoch_0001",
                    out_dir=out_dir,
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=failing_runner,
                )

            self.assertFalse(out_dir.exists())

    def test_epoch_evaluation_cleanup_preserves_another_writers_replacement(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            out_dir = root / "metrics"
            original_publish = module._publish_staged_directory_no_replace
            moved = None
            replacement = None

            def replace_with_directory(staged, final_dir, description):
                nonlocal moved, replacement
                if description != "Evaluator":
                    return original_publish(staged, final_dir, description)
                moved = staged.path.with_name(staged.path.name + "-moved")
                replacement = staged.path
                os.rename(staged.path, moved)
                replacement.mkdir()
                (replacement / "another-writer").write_bytes(b"preserve")
                original_publish(staged, final_dir, description)

            with patch.object(
                module,
                "_publish_staged_directory_no_replace",
                side_effect=replace_with_directory,
            ):
                with self.assertRaisesRegex(ValueError, "identity"):
                    module.evaluate_epoch(
                        target_dir=root / "targets",
                        recon_dir=root / "epoch_0001",
                        out_dir=out_dir,
                        expected_volumes=2,
                        expected_slices=4,
                        expected_boxes=2,
                        runner=lambda command, **kwargs: write_strict_metrics(
                            command, 0.75, **kwargs
                        ),
                    )

            self.assertFalse(out_dir.exists())
            self.assertIsNotNone(replacement)
            self.assertEqual(
                (replacement / "another-writer").read_bytes(), b"preserve"
            )
            self.assertIsNotNone(moved)
            assert moved is not None
            self.assertIn("-unpublished-orphan-", moved.name)
            self.assertTrue((moved / "metrics.json").is_file())

    def test_epoch_evaluation_fails_closed_when_evaluator_reports_skips(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            out_dir = root / "metrics"

            def fake_runner(command, **kwargs):
                assert_evaluator_runner_contract(command, kwargs)
                self.assertEqual(Path(command[1]), module.EVALUATE_VAL_SCRIPT)
                metrics_dir = Path(command[command.index("--out-dir") + 1])
                metrics_dir.mkdir(parents=True, exist_ok=True)
                (metrics_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "target_dir": str(root / "targets"),
                            "recon_dir": str(root / "epoch_0001"),
                            "out_dir": str(metrics_dir),
                            "rows": [],
                            "skipped": [
                                {"file": "b.h5", "reason": "shape mismatch"}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "skips=0"):
                module.evaluate_epoch(
                    target_dir=root / "targets",
                    recon_dir=root / "epoch_0001",
                    out_dir=out_dir,
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=fake_runner,
                )

    def test_epoch_evaluation_rejects_malformed_one_row_metrics_and_invokes_strict_mode(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            out_dir = root / "metrics"

            def fake_runner(command, **kwargs):
                assert_evaluator_runner_contract(command, kwargs)
                self.assertIn("--require-complete", command)
                self.assertEqual(
                    command[command.index("--expected-volumes") + 1], "2"
                )
                self.assertEqual(
                    command[command.index("--expected-slices") + 1], "4"
                )
                self.assertEqual(
                    command[command.index("--expected-boxes") + 1], "2"
                )
                metrics_dir = Path(command[command.index("--out-dir") + 1])
                metrics_dir.mkdir(parents=True, exist_ok=True)
                (metrics_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "target_dir": str(root / "targets"),
                            "recon_dir": str(root / "epoch_0001"),
                            "out_dir": str(metrics_dir),
                            "rows": [
                                {
                                    "scope": "leaderboard_equal_acc",
                                    "ssim_full_mean": 0.8,
                                    "ssim_bbox_mean": 0.7,
                                    "quality_score": 0.75,
                                }
                            ],
                            "skipped": [],
                        }
                    ),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "row scopes"):
                module.evaluate_epoch(
                    target_dir=root / "targets",
                    recon_dir=root / "epoch_0001",
                    out_dir=out_dir,
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=fake_runner,
                )

    def test_epoch_evaluation_rejects_equal_acc_metric_formula_mismatch(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            out_dir = root / "metrics"

            def fake_runner(command, **kwargs):
                write_strict_metrics(command, 0.75, **kwargs)
                metrics_path = Path(
                    command[command.index("--out-dir") + 1]
                ) / "metrics.json"
                summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                equal_acc = next(
                    row
                    for row in summary["rows"]
                    if row["scope"] == "leaderboard_equal_acc"
                )
                equal_acc["ssim_full_mean"] = 0.9
                equal_acc["quality_score"] = (
                    equal_acc["ssim_full_mean"]
                    + equal_acc["ssim_bbox_mean"]
                ) / 2
                metrics_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "equal-acc ssim_full_mean formula mismatch"
            ):
                module.evaluate_epoch(
                    target_dir=root / "targets",
                    recon_dir=root / "epoch_0001",
                    out_dir=out_dir,
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=fake_runner,
                )

    def test_epoch_evaluation_rejects_non_numeric_equal_acc_metric(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            out_dir = root / "metrics"

            def fake_runner(command, **kwargs):
                write_strict_metrics(command, 0.75, **kwargs)
                metrics_path = Path(
                    command[command.index("--out-dir") + 1]
                ) / "metrics.json"
                summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                equal_acc = next(
                    row
                    for row in summary["rows"]
                    if row["scope"] == "leaderboard_equal_acc"
                )
                equal_acc["ssim_full_mean"] = "not-a-number"
                metrics_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "equal-acc ssim_full_mean must be finite"
            ):
                module.evaluate_epoch(
                    target_dir=root / "targets",
                    recon_dir=root / "epoch_0001",
                    out_dir=out_dir,
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=fake_runner,
                )

    def test_sweep_rejects_replaced_staging_symlink_and_preserves_bounded_orphan(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            checkpoint_dir = root / "checkpoints"
            out_dir = root / "sweep"
            target_dir.mkdir()
            checkpoint_dir.mkdir()
            epoch_dir = retained_root / "epoch_0001"
            epoch_dir.mkdir(parents=True)
            (target_dir / "volume.h5").touch()
            (epoch_dir / "volume.h5").touch()
            write_generation(checkpoint_dir, "a" * 32, 1)
            outside = root / "outside"
            outside.mkdir()

            original_publish = module._publish_staged_directory_no_replace
            moved = None
            replacement = None

            def replace_with_symlink(staged, final_dir, description):
                nonlocal moved, replacement
                if description != "Sweep":
                    return original_publish(staged, final_dir, description)
                moved = staged.path.with_name(staged.path.name + "-moved")
                replacement = staged.path
                os.rename(staged.path, moved)
                replacement.symlink_to(outside, target_is_directory=True)
                original_publish(staged, final_dir, description)

            with patch.object(
                module,
                "_publish_staged_directory_no_replace",
                side_effect=replace_with_symlink,
            ):
                with self.assertRaisesRegex(ValueError, "identity"):
                    module.run_sweep(
                        retained_root=retained_root,
                        checkpoint_dir=checkpoint_dir,
                        target_dir=target_dir,
                        out_dir=out_dir,
                        expected_epochs=[1],
                        expected_volumes=2,
                        expected_slices=4,
                        expected_boxes=2,
                        runner=lambda command, **kwargs: write_strict_metrics(
                            command, 0.75, **kwargs
                        ),
                    )

            self.assertFalse(out_dir.exists())
            self.assertIsNotNone(replacement)
            self.assertTrue(replacement.is_symlink())
            self.assertIsNotNone(moved)
            assert moved is not None
            self.assertTrue((moved / "val_epoch_sweep.json").is_file())
            self.assertTrue(outside.is_dir())

    def test_sweep_cleanup_preserves_another_writers_staging_replacement(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            checkpoint_dir = root / "checkpoints"
            out_dir = root / "sweep"
            target_dir.mkdir()
            checkpoint_dir.mkdir()
            epoch_dir = retained_root / "epoch_0001"
            epoch_dir.mkdir(parents=True)
            (target_dir / "volume.h5").touch()
            (epoch_dir / "volume.h5").touch()
            write_generation(checkpoint_dir, "a" * 32, 1)

            original_publish = module._publish_staged_directory_no_replace
            moved = None
            replacement = None

            def replace_with_directory(staged, final_dir, description):
                nonlocal moved, replacement
                if description != "Sweep":
                    return original_publish(staged, final_dir, description)
                moved = staged.path.with_name(staged.path.name + "-moved")
                replacement = staged.path
                os.rename(staged.path, moved)
                replacement.mkdir()
                (replacement / "another-writer").write_bytes(b"preserve")
                original_publish(staged, final_dir, description)

            with patch.object(
                module,
                "_publish_staged_directory_no_replace",
                side_effect=replace_with_directory,
            ):
                with self.assertRaisesRegex(ValueError, "identity"):
                    module.run_sweep(
                        retained_root=retained_root,
                        checkpoint_dir=checkpoint_dir,
                        target_dir=target_dir,
                        out_dir=out_dir,
                        expected_epochs=[1],
                        expected_volumes=2,
                        expected_slices=4,
                        expected_boxes=2,
                        runner=lambda command, **kwargs: write_strict_metrics(
                            command, 0.75, **kwargs
                        ),
                    )

            self.assertFalse(out_dir.exists())
            self.assertIsNotNone(replacement)
            self.assertEqual(
                (replacement / "another-writer").read_bytes(), b"preserve"
            )
            self.assertIsNotNone(moved)
            assert moved is not None
            self.assertTrue((moved / "val_epoch_sweep.json").is_file())

    def test_sweep_fsyncs_nested_tree_leaf_to_root_before_publication(self):
        module = load_sweep_module()
        train_part_module = shared_train_part_module(module)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            checkpoint_dir = root / "checkpoints"
            out_dir = root / "sweep"
            target_dir.mkdir()
            checkpoint_dir.mkdir()
            epoch_dir = retained_root / "epoch_0001"
            epoch_dir.mkdir(parents=True)
            (target_dir / "volume.h5").touch()
            (epoch_dir / "volume.h5").touch()
            write_generation(checkpoint_dir, "a" * 32, 1)

            real_fsync = os.fsync
            fsync_events = []

            def record_fsync(fd):
                file_stat = os.fstat(fd)
                fsync_events.append(
                    (
                        Path(os.readlink(f"/proc/self/fd/{fd}")),
                        stat.S_ISDIR(file_stat.st_mode),
                        out_dir.exists(),
                    )
                )
                real_fsync(fd)

            real_fsync_tree = train_part_module._fsync_directory_tree
            with patch.object(
                train_part_module,
                "_fsync_directory_tree",
                wraps=real_fsync_tree,
            ) as fsync_directory_tree, patch.object(
                train_part_module.os, "fsync", side_effect=record_fsync
            ):
                module.run_sweep(
                    retained_root=retained_root,
                    checkpoint_dir=checkpoint_dir,
                    target_dir=target_dir,
                    out_dir=out_dir,
                    expected_epochs=[1],
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=lambda command, **kwargs: write_strict_metrics(
                        command, 0.75, **kwargs
                    ),
                )

            self.assertGreaterEqual(fsync_directory_tree.call_count, 2)

            def event_index(suffix, is_directory):
                return max(
                    index
                    for index, (path, directory, published) in enumerate(
                        fsync_events
                    )
                    if str(path).endswith(suffix)
                    and directory is is_directory
                    and not published
                )

            epoch_directory_index = event_index(
                "/epoch_metrics/epoch_0001", True
            )
            metrics_root_index = event_index("/epoch_metrics", True)
            staging_root_index = max(
                index
                for index, (path, directory, published) in enumerate(fsync_events)
                if directory
                and not published
                and path.name.startswith(".sweep-unpublished-orphan-")
            )
            for suffix in (
                "/epoch_metrics/epoch_0001/metrics.json",
                "/epoch_metrics/epoch_0001/metrics.csv",
                "/epoch_metrics/epoch_0001/skipped.json",
            ):
                self.assertLess(
                    event_index(suffix, False), epoch_directory_index
                )
            self.assertLess(epoch_directory_index, metrics_root_index)
            self.assertLess(metrics_root_index, staging_root_index)
            for suffix in (
                "/val_epoch_sweep.json",
                "/val_epoch_sweep.csv",
                "/val_epoch_sweep_report.md",
            ):
                self.assertLess(
                    event_index(suffix, False), staging_root_index
                )
            self.assertTrue(
                any(directory and published for _, directory, published in fsync_events)
            )

    def test_sweep_parent_fsync_failure_is_indeterminate_and_preserves_output(self):
        module = load_sweep_module()
        train_part_module = shared_train_part_module(module)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            checkpoint_dir = root / "checkpoints"
            out_dir = root / "sweep"
            target_dir.mkdir()
            checkpoint_dir.mkdir()
            epoch_dir = retained_root / "epoch_0001"
            epoch_dir.mkdir(parents=True)
            (target_dir / "volume.h5").touch()
            (epoch_dir / "volume.h5").touch()
            write_generation(checkpoint_dir, "a" * 32, 1)

            real_fsync = os.fsync
            root_fsync_calls = 0

            def fail_sweep_parent_fsync(fd):
                nonlocal root_fsync_calls
                if Path(os.readlink(f"/proc/self/fd/{fd}")) == root:
                    root_fsync_calls += 1
                    if root_fsync_calls == 2:
                        raise OSError("simulated parent fsync failure")
                real_fsync(fd)

            real_fsync_tree = train_part_module._fsync_directory_tree
            real_publish = train_part_module._publish_staged_directory_no_replace
            with patch.object(
                train_part_module,
                "_fsync_directory_tree",
                wraps=real_fsync_tree,
            ) as fsync_directory_tree, patch.object(
                module,
                "_publish_staged_directory_no_replace",
                wraps=real_publish,
            ) as publish_staged, patch.object(
                train_part_module.os,
                "fsync",
                side_effect=fail_sweep_parent_fsync,
            ):
                with self.assertRaisesRegex(
                    module.PublicationIndeterminateError,
                    "committed but parent fsync failed",
                ):
                    module.run_sweep(
                        retained_root=retained_root,
                        checkpoint_dir=checkpoint_dir,
                        target_dir=target_dir,
                        out_dir=out_dir,
                        expected_epochs=[1],
                        expected_volumes=2,
                        expected_slices=4,
                        expected_boxes=2,
                        runner=lambda command, **kwargs: write_strict_metrics(
                            command, 0.75, **kwargs
                        ),
                    )

            self.assertGreaterEqual(fsync_directory_tree.call_count, 2)
            self.assertEqual(
                [call.args[2] for call in publish_staged.call_args_list],
                ["Evaluator", "Sweep"],
            )
            self.assertTrue(out_dir.is_dir())
            self.assertTrue((out_dir / "val_epoch_sweep.json").is_file())
            self.assertTrue(
                (out_dir / "epoch_metrics" / "epoch_0001" / "metrics.json").is_file()
            )

    def test_sweep_refuses_to_overwrite_existing_output_tree(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            checkpoint_dir = root / "checkpoints"
            out_dir = root / "sweep"
            target_dir.mkdir()
            checkpoint_dir.mkdir()
            epoch_dir = retained_root / "epoch_0001"
            epoch_dir.mkdir(parents=True)
            (target_dir / "volume.h5").touch()
            (epoch_dir / "volume.h5").touch()
            write_generation(checkpoint_dir, "a" * 32, 1)
            out_dir.mkdir()
            stale = out_dir / "val_epoch_sweep.json"
            stale.write_bytes(b"do-not-overwrite")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                module.run_sweep(
                    retained_root=retained_root,
                    checkpoint_dir=checkpoint_dir,
                    target_dir=target_dir,
                    out_dir=out_dir,
                    expected_epochs=[1],
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=lambda *args, **kwargs: None,
                )

            self.assertEqual(stale.read_bytes(), b"do-not-overwrite")

    def test_failed_sweep_never_publishes_partial_reports_or_mutates_best(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            checkpoint_dir = root / "checkpoints"
            out_dir = root / "sweep"
            target_dir.mkdir()
            checkpoint_dir.mkdir()
            (target_dir / "volume.h5").touch()
            for epoch in (1, 2):
                epoch_dir = retained_root / f"epoch_{epoch:04d}"
                epoch_dir.mkdir(parents=True)
                (epoch_dir / "volume.h5").touch()
                write_generation(
                    checkpoint_dir,
                    ("a" if epoch == 1 else "b") * 32,
                    epoch,
                )
            best_path = checkpoint_dir / "best_model.pt"
            best_path.write_bytes(b"do-not-rewrite")

            def fail_second_epoch(command, **kwargs):
                assert_evaluator_runner_contract(command, kwargs)
                recon_dir = Path(command[command.index("--recon-dir") + 1])
                epoch = int(recon_dir.name.split("_")[1])
                metrics_dir = Path(command[command.index("--out-dir") + 1])
                metrics_dir.mkdir(parents=True, exist_ok=True)
                if epoch == 2:
                    (metrics_dir / "metrics.json").write_text(
                        "partial", encoding="utf-8"
                    )
                    raise subprocess.CalledProcessError(2, command)
                write_strict_metrics(command, 0.75, **kwargs)

            with self.assertRaises(subprocess.CalledProcessError):
                module.run_sweep(
                    retained_root=retained_root,
                    checkpoint_dir=checkpoint_dir,
                    target_dir=target_dir,
                    out_dir=out_dir,
                    expected_epochs=[1, 2],
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=fail_second_epoch,
                )

            self.assertFalse(out_dir.exists())
            self.assertEqual(best_path.read_bytes(), b"do-not-rewrite")

    def test_sweep_ranks_equal_acc_and_emits_reports_without_rewriting_best(self):
        module = load_sweep_module()
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            target_dir = root / "targets"
            retained_root = root / "retained"
            checkpoint_dir = root / "checkpoints"
            out_dir = root / "sweep"
            target_dir.mkdir()
            checkpoint_dir.mkdir()
            (target_dir / "volume.h5").touch()
            for epoch in (1, 2):
                epoch_dir = retained_root / f"epoch_{epoch:04d}"
                epoch_dir.mkdir(parents=True)
                (epoch_dir / "volume.h5").touch()
                write_generation(
                    checkpoint_dir,
                    ("a" if epoch == 1 else "b") * 32,
                    epoch,
                )
            best_path = checkpoint_dir / "best_model.pt"
            best_path.write_bytes(b"do-not-rewrite")

            def fake_runner(command, **kwargs):
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                recon_dir = Path(command[command.index("--recon-dir") + 1])
                epoch = int(recon_dir.name.split("_")[1])
                equal_quality = {1: 0.7, 2: 0.8}[epoch]
                write_strict_metrics(command, equal_quality, **kwargs)

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                rankings = module.run_sweep(
                    retained_root=Path("retained"),
                    checkpoint_dir=Path("checkpoints"),
                    target_dir=Path("targets"),
                    out_dir=Path("sweep"),
                    expected_epochs=[1, 2],
                    expected_volumes=2,
                    expected_slices=4,
                    expected_boxes=2,
                    runner=fake_runner,
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual([row["epoch"] for row in rankings], [2, 1])
            self.assertEqual(rankings[0]["checkpoint_generation"], "b" * 32)
            self.assertEqual(
                rankings[0]["checkpoint_artifact"],
                str(
                    (
                        checkpoint_dir
                        / f".checkpoint-generation-{'b' * 32}-model.pt"
                    ).resolve()
                ),
            )
            self.assertEqual(
                rankings[0]["reconstruction_dir"],
                str((retained_root / "epoch_0002").resolve()),
            )
            self.assertEqual(
                rankings[0]["metrics_json"],
                str(
                    (
                        out_dir
                        / "epoch_metrics"
                        / "epoch_0002"
                        / "metrics.json"
                    ).resolve()
                ),
            )
            summary = json.loads(
                (out_dir / "val_epoch_sweep.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["selected"]["epoch"], 2)
            self.assertEqual(
                summary["selected"]["checkpoint_artifact"],
                rankings[0]["checkpoint_artifact"],
            )
            self.assertEqual(summary["selection_metric"], "leaderboard_equal_acc.quality_score")
            with (out_dir / "val_epoch_sweep.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual([row["epoch"] for row in csv_rows], ["2", "1"])
            report = (out_dir / "val_epoch_sweep_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Selected epoch: 2", report)
            self.assertEqual(best_path.read_bytes(), b"do-not-rewrite")


if __name__ == "__main__":
    unittest.main()
