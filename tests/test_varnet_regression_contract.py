import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

if "fcntl" not in sys.modules:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1, LOCK_UN=2, flock=lambda *_args, **_kwargs: None
    )

MODEL_UTILS = Path(__file__).resolve().parents[1] / "utils" / "model"
if str(MODEL_UTILS) not in sys.path:
    sys.path.insert(1, str(MODEL_UTILS))

import train
from utils.learning import train_part


class _Value:
    device = torch.device("cpu")

    def to(self, **_kwargs):
        return self


class _Model:
    def __init__(self):
        self.parameter = _Value()

    def parameters(self):
        return iter([self.parameter])

    def to(self, **_kwargs):
        return self


class VarNetRegressionContractTests(unittest.TestCase):
    def test_optionless_parser_preserves_varnet_defaults(self):
        args = train.parse([])
        self.assertEqual(args.model_family, "varnet")
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.num_epochs, 1)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.net_name, Path("test_varnet"))
        self.assertEqual(args.result_root, Path("../result"))
        self.assertEqual(args.seed, 430)
        self.assertEqual((args.cascade, args.chans, args.sens_chans), (1, 9, 4))
        self.assertEqual(args.model_contract, {"model_family": "varnet"})
        self.assertFalse(args.retain_val_epochs)

    def test_varnet_resume_keeps_adam_and_no_scheduler_or_scaler(self):
        model = _Model()
        loss = _Value()
        optimizer = object()
        checkpoint = Path("legacy-model.pt")
        args = SimpleNamespace(
            GPU_NUM=0,
            require_cuda_device_name=None,
            model_family="varnet",
            model_contract={"model_family": "varnet"},
            cascade=1,
            chans=9,
            sens_chans=4,
            lr=1e-3,
            score_aligned_loss=False,
            resume_checkpoint=checkpoint,
            allow_inexact_resume=False,
            resume_lr=None,
            resume_checkpoint_sha256=None,
            num_epochs=1,
            data_path_train=Path("train"),
            data_path_val=Path("val"),
            retain_val_epochs=False,
            val_epochs_dir=Path("retained"),
            val_loss_dir=Path("loss"),
            exp_dir=Path("checkpoints"),
            val_dir=Path("val-output"),
            net_name=Path("varnet-test"),
            report_interval=500,
        )
        captured = {}

        with patch("torch.cuda.is_available", return_value=False), patch.object(
            train_part, "VarNet", return_value=model
        ) as varnet, patch.object(
            train_part, "SSIMLoss", return_value=loss
        ), patch(
            "torch.optim.Adam", return_value=optimizer
        ) as adam, patch.object(
            train_part, "load_training_state", return_value=(0, 1.0)
        ) as resume, patch.object(
            train_part, "load_val_loss_history",
            return_value=torch.empty((0, 2)).numpy(),
        ), patch.object(train_part, "preserve_best_checkpoint"), patch.object(
            train_part, "create_data_loaders", side_effect=[[], []]
        ) as loaders, patch.object(
            train_part, "train_epoch", return_value=(0.2, 0.01)
        ), patch.object(
            train_part, "validate",
            return_value=(0.1, 1, {"sample.h5": None}, {}, None, 0.01),
        ), patch.object(
            train_part, "save_model",
            side_effect=lambda *positional, **keywords: captured.update(keywords),
        ), patch.object(train_part, "save_reconstructions"):
            train_part.train(args)

        varnet.assert_called_once_with(num_cascades=1, chans=9, sens_chans=4)
        adam.assert_called_once()
        self.assertEqual(list(adam.call_args.args[0]), [model.parameter])
        self.assertEqual(adam.call_args.args[1], 1e-3)
        self.assertEqual(loaders.call_count, 2)
        self.assertIsNone(captured["scheduler"])
        self.assertIsNone(captured["scaler"])
        self.assertIsNone(captured["model_contract"])
        self.assertIsNone(resume.call_args.kwargs["scheduler"])
        self.assertIsNone(resume.call_args.kwargs["scaler"])
        self.assertEqual(
            resume.call_args.kwargs["expected_model_contract"],
            {"model_family": "varnet"},
        )


if __name__ == "__main__":
    unittest.main()
