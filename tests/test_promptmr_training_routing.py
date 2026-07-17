import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

if "fcntl" not in sys.modules:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1,
        LOCK_UN=2,
        flock=lambda *_args, **_kwargs: None,
    )

MODEL_UTILS = Path(__file__).resolve().parents[1] / "utils" / "model"
if str(MODEL_UTILS) not in sys.path:
    sys.path.insert(1, str(MODEL_UTILS))

from utils.learning import train_part
from utils.promptmr.contracts import checkpoint_model_contract


class _DeviceValue:
    device = torch.device("cpu")

    def to(self, **_kwargs):
        return self


class _LossValue:
    def __init__(self, events):
        self.events = events

    def backward(self):
        self.events.append("backward")

    def item(self):
        return 0.25


class _FakeModel:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.parameter = _DeviceValue()

    def train(self):
        self.events.append("model.train")

    def eval(self):
        self.events.append("model.eval")

    def parameters(self):
        return iter([self.parameter])

    def to(self, **_kwargs):
        self.events.append("model.to")
        return self

    def __call__(self, _kspace, _mask):
        self.events.append("forward")
        return _DeviceValue()


class _FakeLoss:
    def __init__(self, events):
        self.events = events

    def to(self, **_kwargs):
        self.events.append("loss.to")
        return self

    def __call__(self, _output, _target, _maximum):
        self.events.append("loss")
        return _LossValue(self.events)


class _FakeOptimizer:
    def __init__(self, events):
        self.events = events

    def zero_grad(self):
        self.events.append("zero_grad")


class _FakeScaler:
    def __init__(self, events):
        self.events = events

    def scale(self, loss):
        self.events.append("scale")
        return loss

    def unscale_(self, _optimizer):
        self.events.append("unscale")

    def step(self, _optimizer):
        self.events.append("optimizer_step")

    def update(self):
        self.events.append("scaler_update")


class _FakeScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class PromptMRTrainingRoutingTests(unittest.TestCase):
    def test_fake_promptmr_step_orders_unscale_clip_and_optimizer(self):
        events = []
        args = SimpleNamespace(
            model_family="promptmr_plus",
            score_aligned_loss=False,
            gradient_clip_val=0.01,
            report_interval=99,
            num_epochs=1,
        )
        model = _FakeModel(events)
        optimizer = _FakeOptimizer(events)
        scaler = _FakeScaler(events)
        loss = _FakeLoss(events)
        data = [(
            _DeviceValue(), _DeviceValue(), _DeviceValue(),
            _DeviceValue(), ["sample.h5"], [0],
        )]

        with patch(
            "utils.promptmr.data.align_promptmr_output_target",
            side_effect=lambda output, target: (output, target),
        ), patch(
            "torch.nn.utils.clip_grad_norm_",
            side_effect=lambda *_args, **_kwargs: events.append("clip"),
        ):
            train_part.train_epoch(
                args, 0, model, data, optimizer, loss, scaler=scaler
            )

        ordered = [
            "zero_grad", "forward", "loss", "scale", "backward",
            "unscale", "clip", "optimizer_step", "scaler_update",
        ]
        positions = [events.index(name) for name in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_train_routes_promptmr_factories_and_checkpoint_state(self):
        events = []
        model = _FakeModel(events)
        loss = _FakeLoss(events)
        optimizer = _FakeOptimizer(events)
        scheduler = _FakeScheduler()
        scaler = _FakeScaler(events)
        contract = checkpoint_model_contract("promptmr_plus")
        args = SimpleNamespace(
            GPU_NUM=0,
            require_cuda_device_name=None,
            model_family="promptmr_plus",
            model_contract=contract,
            lr=1e-4,
            resume_checkpoint=None,
            allow_inexact_resume=False,
            resume_lr=None,
            resume_checkpoint_sha256=None,
            num_epochs=1,
            data_path_train=Path("train"),
            data_path_val=Path("val"),
            batch_size=1,
            input_key="kspace",
            target_key="image_label",
            max_key="max",
            num_workers=0,
            retain_val_epochs=False,
            val_epochs_dir=Path("retained"),
            val_loss_dir=Path("loss"),
            exp_dir=Path("checkpoints"),
            val_dir=Path("val-output"),
            net_name=Path("promptmr-test"),
            report_interval=10,
            score_aligned_loss=False,
            gradient_clip_val=0.01,
        )
        captured = {}

        with patch("torch.cuda.is_available", return_value=False), patch(
            "utils.learning.train_part.VarNet",
            side_effect=AssertionError("VarNet route was touched"),
        ), patch(
            "utils.learning.train_part.SSIMLoss",
            side_effect=AssertionError("legacy loss route was touched"),
        ), patch(
            "utils.learning.train_part.create_data_loaders",
            side_effect=AssertionError("legacy loader route was touched"),
        ), patch(
            "utils.promptmr.runtime.build_promptmr_plus_model", return_value=model
        ) as model_factory, patch(
            "utils.promptmr.runtime.build_promptmr_plus_loss", return_value=loss
        ) as loss_factory, patch(
            "torch.optim.AdamW", return_value=optimizer
        ) as adamw, patch(
            "torch.optim.lr_scheduler.StepLR", return_value=scheduler
        ) as step_lr, patch(
            "torch.amp.GradScaler", return_value=scaler
        ) as grad_scaler, patch(
            "utils.promptmr.data.create_promptmr_data_loaders",
            return_value=([], []),
        ) as data_factory, patch.object(
            train_part, "train_epoch", return_value=(0.2, 0.01)
        ), patch.object(
            train_part, "validate",
            return_value=(0.1, 1, {"sample.h5": None}, {}, None, 0.01),
        ), patch.object(
            train_part, "save_model",
            side_effect=lambda *positional, **keywords: captured.update(keywords),
        ), patch.object(train_part, "save_reconstructions"):
            train_part.train(args)

        model_factory.assert_called_once_with()
        loss_factory.assert_called_once_with()
        adamw.assert_called_once()
        self.assertEqual(list(adamw.call_args.args[0]), [model.parameter])
        self.assertEqual(
            adamw.call_args.kwargs, {"lr": 1e-4, "weight_decay": 1e-2}
        )
        step_lr.assert_called_once_with(optimizer, step_size=35, gamma=0.1)
        grad_scaler.assert_called_once_with("cuda", enabled=False)
        data_factory.assert_called_once_with(args)
        self.assertEqual(scheduler.steps, 1)
        self.assertIs(captured["scheduler"], scheduler)
        self.assertIs(captured["scaler"], scaler)
        self.assertEqual(captured["model_contract"], contract)


if __name__ == "__main__":
    unittest.main()
