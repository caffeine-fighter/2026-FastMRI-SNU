import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train
import utils.learning.train_part as train_part_module


class TrainCliTests(unittest.TestCase):
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
            ):
                train_part_module.train(args)

            set_device.assert_not_called()
            current_device.assert_not_called()
            self.assertEqual(save_model.call_count, 1)
            saved_best_loss = save_model.call_args.args[4]
            self.assertAlmostEqual(float(saved_best_loss), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
