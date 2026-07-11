import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

import train
import utils.learning.train_part as train_part_module


class TrainCliTests(unittest.TestCase):
    def test_validation_epoch_retention_defaults_off(self):
        with patch("sys.argv", ["train.py"]):
            args = train.parse()

        self.assertFalse(args.retain_val_epochs)

    def test_validation_epoch_retention_is_opt_in(self):
        with patch("sys.argv", ["train.py", "--retain-val-epochs"]):
            args = train.parse()

        self.assertTrue(args.retain_val_epochs)

    def test_parse_accepts_resume_checkpoint(self):
        checkpoint = Path("/tmp/EXP030/checkpoints/model.pt")
        with patch(
            "sys.argv",
            ["train.py", "--resume-checkpoint", str(checkpoint)],
        ):
            args = train.parse()

        self.assertEqual(args.resume_checkpoint, checkpoint)
        self.assertFalse(args.allow_inexact_resume)

    def test_parse_rejects_explicit_inexact_resume_without_checkpoint(self):
        with patch(
            "sys.argv",
            ["train.py", "--allow-inexact-resume"],
        ):
            with self.assertRaises(SystemExit):
                train.parse()

    def test_parse_accepts_resume_learning_rate_override(self):
        with patch(
            "sys.argv",
            [
                "train.py",
                "--resume-checkpoint",
                "/tmp/EXP031/checkpoints/best_model.pt",
                "--resume-lr",
                "0.0001",
            ],
        ):
            args = train.parse()

        self.assertEqual(args.resume_lr, 0.0001)

    def test_parse_rejects_nonpositive_resume_learning_rate(self):
        with patch(
            "sys.argv",
            [
                "train.py",
                "--resume-checkpoint",
                "/tmp/model.pt",
                "--resume-lr",
                "0",
            ],
        ):
            with self.assertRaises(SystemExit):
                train.parse()

    def test_parse_rejects_resume_learning_rate_without_checkpoint(self):
        with patch(
            "sys.argv",
            ["train.py", "--resume-lr", "0.0001"],
        ):
            with self.assertRaises(SystemExit):
                train.parse()

    def test_opt_in_retention_writes_reconstruction_only_to_deterministic_epoch_dir(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            args = SimpleNamespace(
                GPU_NUM=0,
                cascade=1,
                chans=1,
                sens_chans=1,
                lr=0.001,
                resume_checkpoint=None,
                allow_inexact_resume=False,
                resume_lr=None,
                num_epochs=1,
                data_path_train=root / "train",
                data_path_val=root / "val",
                exp_dir=checkpoint_dir,
                val_loss_dir=root,
                val_dir=root / "reconstructions_val",
                val_epochs_dir=root / "reconstructions_val_epochs",
                retain_val_epochs=True,
                net_name="cpu-retention",
            )
            model = torch.nn.Linear(1, 1)
            loss = torch.nn.Identity()
            reconstructions = {
                "sample.h5": np.arange(4, dtype=np.float32).reshape(1, 2, 2)
            }
            targets = {
                "sample.h5": np.ones((1, 2, 2), dtype=np.float32)
            }
            events = []
            original_publish = train_part_module._publish_retained_epoch

            def publish_then_record(*publish_args):
                original_publish(*publish_args)
                events.append("retention-published")

            with patch.object(
                train_part_module.torch.cuda, "is_available", return_value=False
            ), patch.object(
                train_part_module, "VarNet", return_value=model
            ), patch.object(
                train_part_module, "SSIMLoss", return_value=loss
            ), patch.object(
                train_part_module, "create_data_loaders", side_effect=[[], []]
            ), patch.object(
                train_part_module, "train_epoch", return_value=(0.2, 0.01)
            ), patch.object(
                train_part_module,
                "validate",
                return_value=(0.1, 1, reconstructions, targets, None, 0.01),
            ), patch.object(
                train_part_module,
                "_publish_retained_epoch",
                side_effect=publish_then_record,
            ), patch.object(
                train_part_module,
                "save_model",
                side_effect=lambda *args, **kwargs: events.append("checkpoint-committed"),
            ):
                train_part_module.train(args)

            self.assertEqual(
                events, ["retention-published", "checkpoint-committed"]
            )

            retained = args.val_epochs_dir / "epoch_0001" / "sample.h5"
            self.assertTrue(retained.is_file())
            self.assertEqual(stat.S_IMODE(retained.parent.stat().st_mode), 0o555)
            self.assertEqual(stat.S_IMODE(retained.stat().st_mode), 0o444)
            with h5py.File(retained, "r") as hf:
                self.assertEqual(set(hf.keys()), {"reconstruction"})
                np.testing.assert_array_equal(
                    hf["reconstruction"][:], reconstructions["sample.h5"]
                )

    def test_retention_rejects_content_mutation_after_seal_before_rename(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            staged, final_dir = train_part_module._stage_retained_reconstructions(
                {"sample.h5": np.zeros((1, 2, 2), dtype=np.float32)},
                root / "retained",
                1,
            )
            staged_file = Path(f"/proc/self/fd/{staged.directory_fd}/sample.h5")
            staged_file.chmod(0o644)
            with staged_file.open("ab") as handle:
                handle.write(b"mutation")

            with self.assertRaisesRegex(ValueError, "tree changed after sealing"):
                train_part_module._publish_retained_epoch(staged, final_dir)
            train_part_module._cleanup_staged_directory(staged)

            self.assertFalse(final_dir.exists())
            self.assertTrue(staged.path.is_dir())

    def test_opt_in_retention_rejects_non_h5_basename(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            retained_root = root / "retained"
            with self.assertRaisesRegex(ValueError, r"regular \.h5 basename"):
                train_part_module._stage_retained_reconstructions(
                    {"sample.txt": np.zeros((1, 2, 2), dtype=np.float32)},
                    retained_root,
                    1,
                )

            self.assertFalse(retained_root.exists())

    def test_retention_publication_rejects_replaced_symlink_and_preserves_bounded_orphan(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            staged, final_dir = train_part_module._stage_retained_reconstructions(
                {"sample.h5": np.zeros((1, 2, 2), dtype=np.float32)},
                root / "retained",
                1,
            )
            self.assertIn("-orphan-", staged.path.name)
            owned_moved = staged.path.with_name(staged.path.name + "-moved")
            outside = root / "outside"
            outside.mkdir()
            os.rename(staged.path, owned_moved)
            staged.path.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "identity"):
                train_part_module._publish_retained_epoch(staged, final_dir)
            train_part_module._cleanup_staged_directory(staged)

            self.assertFalse(final_dir.exists())
            self.assertTrue(staged.path.is_symlink())
            self.assertTrue((owned_moved / "sample.h5").is_file())
            self.assertTrue(outside.is_dir())

    def test_retention_cleanup_never_removes_owned_or_replacement_paths(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            staged, _ = train_part_module._stage_retained_reconstructions(
                {"sample.h5": np.zeros((1, 2, 2), dtype=np.float32)},
                root / "retained",
                1,
            )
            owned_moved = staged.path.with_name(staged.path.name + "-moved")
            os.rename(staged.path, owned_moved)
            staged.path.mkdir()
            replacement_marker = staged.path / "another-writer"
            replacement_marker.write_bytes(b"preserve")

            with patch.object(
                train_part_module.os,
                "unlink",
                side_effect=AssertionError("cleanup must never unlink"),
            ), patch.object(
                train_part_module.os,
                "rmdir",
                side_effect=AssertionError("cleanup must never rmdir"),
            ):
                train_part_module._cleanup_staged_directory(staged)

            self.assertEqual(replacement_marker.read_bytes(), b"preserve")
            self.assertTrue((owned_moved / "sample.h5").is_file())

    def test_missing_parent_chain_and_staging_entry_are_durably_created(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            final_dir = root / "one" / "two" / "three" / "epoch_0001"
            real_fsync = os.fsync
            fsynced_directories = []

            def record_fsync(fd):
                file_stat = os.fstat(fd)
                if stat.S_ISDIR(file_stat.st_mode):
                    fsynced_directories.append(
                        Path(os.readlink(f"/proc/self/fd/{fd}"))
                    )
                real_fsync(fd)

            with patch.object(
                train_part_module.os, "fsync", side_effect=record_fsync
            ):
                staged = train_part_module._create_staged_directory(
                    final_dir, "Retained epoch"
                )
            train_part_module._cleanup_staged_directory(staged)

            self.assertEqual(staged.path.parent, final_dir.parent)
            for containing_parent in (
                root,
                root / "one",
                root / "one" / "two",
                root / "one" / "two" / "three",
            ):
                self.assertIn(containing_parent, fsynced_directories)

    def test_retention_parent_fsync_failure_is_indeterminate_and_keeps_published_epoch(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            staged, final_dir = train_part_module._stage_retained_reconstructions(
                {"sample.h5": np.zeros((1, 2, 2), dtype=np.float32)},
                root / "retained",
                1,
            )

            with patch.object(
                train_part_module.os,
                "fsync",
                side_effect=OSError("simulated parent fsync failure"),
            ):
                with self.assertRaisesRegex(
                    train_part_module.PublicationIndeterminateError,
                    "committed but parent fsync failed",
                ):
                    train_part_module._publish_retained_epoch(staged, final_dir)
            train_part_module._cleanup_staged_directory(staged)

            self.assertTrue(final_dir.is_dir())
            self.assertTrue((final_dir / "sample.h5").is_file())

    def test_opt_in_retention_rejects_traversal_before_checkpoint_commit(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            args = SimpleNamespace(
                GPU_NUM=0,
                cascade=1,
                chans=1,
                sens_chans=1,
                lr=0.001,
                resume_checkpoint=None,
                allow_inexact_resume=False,
                resume_lr=None,
                num_epochs=1,
                data_path_train=root / "train",
                data_path_val=root / "val",
                exp_dir=checkpoint_dir,
                val_loss_dir=root,
                val_dir=root / "reconstructions_val",
                val_epochs_dir=root / "reconstructions_val_epochs",
                retain_val_epochs=True,
                net_name="cpu-retention",
            )
            model = torch.nn.Linear(1, 1)
            loss = torch.nn.Identity()
            reconstructions = {
                "../escaped.h5": np.zeros((1, 2, 2), dtype=np.float32)
            }

            with patch.object(
                train_part_module.torch.cuda, "is_available", return_value=False
            ), patch.object(
                train_part_module, "VarNet", return_value=model
            ), patch.object(
                train_part_module, "SSIMLoss", return_value=loss
            ), patch.object(
                train_part_module, "create_data_loaders", side_effect=[[], []]
            ), patch.object(
                train_part_module, "train_epoch", return_value=(0.2, 0.01)
            ), patch.object(
                train_part_module,
                "validate",
                return_value=(0.1, 1, reconstructions, {}, None, 0.01),
            ), patch.object(train_part_module, "save_model") as save_model:
                with self.assertRaisesRegex(ValueError, "basename"):
                    train_part_module.train(args)

            save_model.assert_not_called()
            self.assertFalse((args.val_epochs_dir / "escaped.h5").exists())
            self.assertFalse((root / "escaped.h5").exists())

    def test_opt_in_retention_refuses_to_overwrite_existing_epoch(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            epoch_dir = root / "reconstructions_val_epochs" / "epoch_0001"
            epoch_dir.mkdir(parents=True)
            existing = epoch_dir / "sample.h5"
            existing.write_bytes(b"keep-existing-epoch")
            args = SimpleNamespace(
                GPU_NUM=0,
                cascade=1,
                chans=1,
                sens_chans=1,
                lr=0.001,
                resume_checkpoint=None,
                allow_inexact_resume=False,
                resume_lr=None,
                num_epochs=1,
                data_path_train=root / "train",
                data_path_val=root / "val",
                exp_dir=checkpoint_dir,
                val_loss_dir=root,
                val_dir=root / "reconstructions_val",
                val_epochs_dir=root / "reconstructions_val_epochs",
                retain_val_epochs=True,
                net_name="cpu-retention",
            )
            model = torch.nn.Linear(1, 1)
            loss = torch.nn.Identity()
            reconstructions = {
                "sample.h5": np.zeros((1, 2, 2), dtype=np.float32)
            }

            with patch.object(
                train_part_module.torch.cuda, "is_available", return_value=False
            ), patch.object(
                train_part_module, "VarNet", return_value=model
            ), patch.object(
                train_part_module, "SSIMLoss", return_value=loss
            ), patch.object(
                train_part_module, "create_data_loaders", side_effect=[[], []]
            ), patch.object(
                train_part_module, "train_epoch", return_value=(0.2, 0.01)
            ), patch.object(
                train_part_module,
                "validate",
                return_value=(0.1, 1, reconstructions, {}, None, 0.01),
            ), patch.object(train_part_module, "save_model") as save_model:
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    train_part_module.train(args)

            save_model.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"keep-existing-epoch")

    def test_opt_in_retention_never_publishes_partial_epoch(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            args = SimpleNamespace(
                GPU_NUM=0,
                cascade=1,
                chans=1,
                sens_chans=1,
                lr=0.001,
                resume_checkpoint=None,
                allow_inexact_resume=False,
                resume_lr=None,
                num_epochs=1,
                data_path_train=root / "train",
                data_path_val=root / "val",
                exp_dir=checkpoint_dir,
                val_loss_dir=root,
                val_dir=root / "reconstructions_val",
                val_epochs_dir=root / "reconstructions_val_epochs",
                retain_val_epochs=True,
                net_name="cpu-retention",
            )
            model = torch.nn.Linear(1, 1)
            loss = torch.nn.Identity()
            reconstructions = {
                "sample.h5": np.zeros((1, 2, 2), dtype=np.float32)
            }

            def fail_after_partial_write(recons, out_dir, **kwargs):
                del recons, kwargs
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "sample.h5").write_bytes(b"partial")
                raise OSError("simulated retention write failure")

            with patch.object(
                train_part_module.torch.cuda, "is_available", return_value=False
            ), patch.object(
                train_part_module, "VarNet", return_value=model
            ), patch.object(
                train_part_module, "SSIMLoss", return_value=loss
            ), patch.object(
                train_part_module, "create_data_loaders", side_effect=[[], []]
            ), patch.object(
                train_part_module, "train_epoch", return_value=(0.2, 0.01)
            ), patch.object(
                train_part_module,
                "validate",
                return_value=(0.1, 1, reconstructions, {}, None, 0.01),
            ), patch.object(
                train_part_module,
                "save_reconstructions",
                side_effect=fail_after_partial_write,
            ), patch.object(train_part_module, "save_model") as save_model:
                with self.assertRaisesRegex(OSError, "simulated"):
                    train_part_module.train(args)

            save_model.assert_not_called()
            self.assertFalse(
                (args.val_epochs_dir / "epoch_0001").exists()
            )

    def test_fresh_cpu_training_never_queries_or_selects_cuda_device(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            args = SimpleNamespace(
                GPU_NUM=0,
                cascade=1,
                chans=1,
                sens_chans=1,
                lr=0.001,
                resume_checkpoint=None,
                allow_inexact_resume=False,
                resume_lr=None,
                num_epochs=1,
                data_path_train=root / "train",
                data_path_val=root / "val",
                exp_dir=checkpoint_dir,
                val_loss_dir=root,
                val_dir=root / "reconstructions",
                val_epochs_dir=root / "reconstructions_val_epochs",
                retain_val_epochs=False,
                net_name="cpu-regression",
            )
            model = torch.nn.Linear(1, 1)
            loss = torch.nn.Identity()

            with patch.object(
                train_part_module.torch.cuda, "is_available", return_value=False
            ), patch.object(
                train_part_module.torch.cuda, "set_device"
            ) as set_device, patch.object(
                train_part_module.torch.cuda, "current_device"
            ) as current_device, patch.object(
                train_part_module, "VarNet", return_value=model
            ), patch.object(
                train_part_module, "SSIMLoss", return_value=loss
            ), patch.object(
                train_part_module, "create_data_loaders", side_effect=[[], []]
            ), patch.object(
                train_part_module, "train_epoch", return_value=(0.2, 0.01)
            ), patch.object(
                train_part_module,
                "validate",
                return_value=(0.1, 1, {}, {}, None, 0.01),
            ), patch.object(
                train_part_module, "save_model"
            ) as save_model, patch.object(
                train_part_module, "save_val_loss_history"
            ), patch.object(
                train_part_module, "save_reconstructions"
            ) as save_reconstructions:
                train_part_module.train(args)

            set_device.assert_not_called()
            current_device.assert_not_called()
            self.assertEqual(save_model.call_count, 1)
            saved_best_loss = save_model.call_args.args[4]
            self.assertAlmostEqual(float(saved_best_loss), 0.1, places=6)
            save_reconstructions.assert_called_once()
            self.assertEqual(save_reconstructions.call_args.args[1], args.val_dir)
            self.assertFalse(args.val_epochs_dir.exists())


if __name__ == "__main__":
    unittest.main()
