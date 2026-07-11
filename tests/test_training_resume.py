import copy
import json
import random
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import utils.learning.resume as resume_module

MODEL_UTILS = Path(__file__).resolve().parents[1] / "utils" / "model"
if str(MODEL_UTILS) not in sys.path:
    sys.path.insert(1, str(MODEL_UTILS))

import utils.learning.train_part as train_part_module
from utils.learning.train_part import save_model
from utils.learning.resume import (
    _copy_without_overwrite,
    build_training_state,
    load_training_state,
    load_val_loss_history,
    preserve_best_checkpoint,
    sanitize_legacy_training_state,
    set_optimizer_learning_rate,
    validate_training_checkpoint,
)


def load_converter_module(name):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "convert_trusted_training_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load checkpoint converter from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tmpfile_test_root():
    candidates = [
        Path(tempfile.gettempdir()),
        Path(__file__).resolve().parents[1],
        Path.home(),
    ]
    for candidate in candidates:
        directory_fd = None
        anonymous_fd = None
        try:
            directory_fd = os.open(candidate, os.O_RDONLY | os.O_DIRECTORY)
            anonymous_fd = os.open(
                ".",
                os.O_RDWR | os.O_TMPFILE,
                0o600,
                dir_fd=directory_fd,
            )
            return str(candidate)
        except OSError:
            continue
        finally:
            if anonymous_fd is not None:
                os.close(anonymous_fd)
            if directory_fd is not None:
                os.close(directory_fd)
    raise unittest.SkipTest("resume publication tests require Linux O_TMPFILE")


def _assert_nested_state_equal(test_case, actual, expected):
    if torch.is_tensor(expected):
        test_case.assertTrue(torch.equal(actual, expected))
    elif isinstance(expected, dict):
        test_case.assertEqual(set(actual), set(expected))
        for key in expected:
            _assert_nested_state_equal(test_case, actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        test_case.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_state_equal(test_case, actual_item, expected_item)
    else:
        test_case.assertEqual(actual, expected)


def _capture_live_state(model, optimizer):
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }


def _assert_live_state_unchanged(test_case, model, optimizer, before):
    _assert_nested_state_equal(test_case, model.state_dict(), before["model"])
    _assert_nested_state_equal(test_case, optimizer.state_dict(), before["optimizer"])
    test_case.assertEqual(random.getstate(), before["python"])
    numpy_actual = np.random.get_state()
    test_case.assertEqual(numpy_actual[0], before["numpy"][0])
    np.testing.assert_array_equal(numpy_actual[1], before["numpy"][1])
    test_case.assertEqual(numpy_actual[2:], before["numpy"][2:])
    test_case.assertTrue(torch.equal(torch.get_rng_state(), before["torch"]))


class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_tempdir = tempfile.tempdir
        tempfile.tempdir = _tmpfile_test_root()

    @classmethod
    def tearDownClass(cls):
        tempfile.tempdir = cls._previous_tempdir

    def test_load_training_state_restores_full_training_state_safely(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        loss = source_model(torch.tensor([[1.0, 2.0]])).sum()
        loss.backward()
        source_optimizer.step()

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint = Path(tmp) / "model.pt"
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                state = build_training_state(
                    epoch=20,
                    model=source_model,
                    optimizer=source_optimizer,
                    best_val_loss=0.1068,
                )
            torch.save(state, checkpoint)

            expected_random = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = float(torch.rand(1))
            random.random()
            np.random.rand()
            torch.rand(1)

            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.001)
            with patch(
                "utils.learning.resume.torch.load",
                wraps=torch.load,
            ) as mocked_load:
                start_epoch, best_val_loss = load_training_state(
                    checkpoint,
                    resumed_model,
                    resumed_optimizer,
                    torch.device("cpu"),
                )

        self.assertEqual(start_epoch, 20)
        self.assertAlmostEqual(float(best_val_loss), 0.1068, places=6)
        for key, value in source_model.state_dict().items():
            self.assertTrue(torch.equal(resumed_model.state_dict()[key], value))
        self.assertTrue(resumed_optimizer.state_dict()["state"])
        self.assertTrue(mocked_load.call_args.kwargs["weights_only"])
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(float(np.random.rand()), expected_numpy)
        self.assertEqual(float(torch.rand(1)), expected_torch)

    def test_load_training_state_applies_explicit_learning_rate_override(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint = Path(tmp) / "model.pt"
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                state = build_training_state(
                    epoch=28,
                    model=source_model,
                    optimizer=source_optimizer,
                    best_val_loss=0.106,
                )
            torch.save(state, checkpoint)
            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.005)

            load_training_state(
                checkpoint,
                resumed_model,
                resumed_optimizer,
                torch.device("cpu"),
                learning_rate_override=0.0001,
            )

        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], 0.0001)

    def test_set_optimizer_learning_rate_updates_all_parameter_groups(self):
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.Adam(
            [
                {"params": [first], "lr": 0.001},
                {"params": [second], "lr": 0.002},
            ]
        )

        set_optimizer_learning_rate(optimizer, 0.0001)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.0001, 0.0001])

    def test_set_optimizer_learning_rate_rejects_nonpositive_or_nonfinite_values(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=0.001)

        for value in (0.0, -0.001, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    set_optimizer_learning_rate(optimizer, value)

    def test_checkpoint_schema_rejects_invalid_state_dict_containers(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with patch(
            "utils.learning.resume.torch.cuda.is_available",
            return_value=False,
        ):
            valid = build_training_state(20, model, optimizer, 0.1068)

        for field in ("model", "optimizer"):
            for invalid in (None, [], "not-a-state-dict"):
                with self.subTest(field=field, invalid=invalid):
                    checkpoint = copy.deepcopy(valid)
                    checkpoint[field] = invalid
                    with self.assertRaisesRegex(ValueError, field):
                        validate_training_checkpoint(checkpoint)

    def test_semantic_schema_failures_precede_state_mutation(self):
        source_model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            source_model.weight.fill_(7.0)
            source_model.bias.fill_(7.0)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        source_model(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        with patch(
            "utils.learning.resume.torch.cuda.is_available",
            return_value=False,
        ):
            valid = build_training_state(
                20, source_model, source_optimizer, 0.1068
            )

        corruptions = {
            "python-version": lambda state: state["rng_state"].__setitem__(
                "python_version", 999
            ),
            "numpy-name": lambda state: state["rng_state"].__setitem__(
                "numpy_name", "not-a-bit-generator"
            ),
            "boolean-loss": lambda state: state.__setitem__("best_val_loss", True),
            "empty-model": lambda state: state.__setitem__("model", {}),
            "shallow-optimizer": lambda state: state.__setitem__("optimizer", {}),
        }

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            for label, corrupt in corruptions.items():
                with self.subTest(label=label):
                    checkpoint = copy.deepcopy(valid)
                    corrupt(checkpoint)
                    checkpoint_path = Path(tmp) / f"{label}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    target_model = torch.nn.Linear(2, 1)
                    target_optimizer = torch.optim.Adam(
                        target_model.parameters(), lr=0.005
                    )
                    before = copy.deepcopy(target_model.state_dict())

                    with self.assertRaises(ValueError):
                        load_training_state(
                            checkpoint_path,
                            target_model,
                            target_optimizer,
                            torch.device("cpu"),
                        )

                    for key, value in before.items():
                        self.assertTrue(
                            torch.equal(target_model.state_dict()[key], value), label
                        )
                    self.assertFalse(target_optimizer.state_dict()["state"], label)

    def test_model_compatibility_failure_is_transactional(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        with patch(
            "utils.learning.resume.torch.cuda.is_available", return_value=False
        ):
            checkpoint = build_training_state(
                20, source_model, source_optimizer, 0.1068
            )
        del checkpoint["model"]["bias"]

        target_model = torch.nn.Linear(2, 1)
        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.005)
        target_model(torch.ones(1, 2)).sum().backward()
        target_optimizer.step()
        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        before = _capture_live_state(target_model, target_optimizer)

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "model.*keys"):
                load_training_state(
                    checkpoint_path,
                    target_model,
                    target_optimizer,
                    torch.device("cpu"),
                )

        _assert_live_state_unchanged(
            self, target_model, target_optimizer, before
        )

    def test_optimizer_topology_failure_is_transactional(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        with patch(
            "utils.learning.resume.torch.cuda.is_available", return_value=False
        ):
            checkpoint = build_training_state(
                20, source_model, source_optimizer, 0.1068
            )

        target_model = torch.nn.Linear(2, 1)
        target_optimizer = torch.optim.Adam(
            [
                {"params": [target_model.weight], "lr": 0.005},
                {"params": [target_model.bias], "lr": 0.006},
            ]
        )
        target_model(torch.ones(1, 2)).sum().backward()
        target_optimizer.step()
        random.seed(201)
        np.random.seed(202)
        torch.manual_seed(203)
        before = _capture_live_state(target_model, target_optimizer)

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "optimizer.*topology"):
                load_training_state(
                    checkpoint_path,
                    target_model,
                    target_optimizer,
                    torch.device("cpu"),
                )

        _assert_live_state_unchanged(
            self, target_model, target_optimizer, before
        )

    def test_cuda_rng_restoration_failure_rolls_back_all_live_state(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        with patch(
            "utils.learning.resume.torch.cuda.is_available", return_value=False
        ):
            checkpoint = build_training_state(
                20, source_model, source_optimizer, 0.1068
            )
        checkpoint_cuda_state = torch.arange(32, dtype=torch.uint8)
        checkpoint["rng_state"]["torch_cuda"] = [checkpoint_cuda_state]

        target_model = torch.nn.Linear(2, 1)
        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.005)
        target_model(torch.ones(1, 2)).sum().backward()
        target_optimizer.step()
        random.seed(301)
        np.random.seed(302)
        torch.manual_seed(303)
        before = _capture_live_state(target_model, target_optimizer)
        original_cuda_state = torch.arange(32, 64, dtype=torch.uint8)
        live_cuda_state = [original_cuda_state.clone()]

        def set_fake_cuda_rng(state, device=None):
            if torch.equal(state.cpu(), checkpoint_cuda_state):
                raise RuntimeError("invalid CUDA RNG state")
            live_cuda_state[0] = state.cpu().clone()

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            with patch.object(
                resume_module, "_selected_cuda_device_index", return_value=0
            ), patch.object(
                resume_module.torch.cuda,
                "get_rng_state",
                return_value=original_cuda_state.clone(),
            ), patch.object(
                resume_module.torch.cuda,
                "set_rng_state",
                side_effect=set_fake_cuda_rng,
            ):
                with self.assertRaisesRegex(ValueError, "CUDA RNG.*restorable"):
                    load_training_state(
                        checkpoint_path,
                        target_model,
                        target_optimizer,
                        torch.device("cpu"),
                    )

        _assert_live_state_unchanged(
            self, target_model, target_optimizer, before
        )
        self.assertTrue(torch.equal(live_cuda_state[0], original_cuda_state))

    def test_checkpoint_pair_rejects_model_architecture_mismatch(self):
        latest_model = torch.nn.Linear(2, 1)
        latest_optimizer = torch.optim.Adam(latest_model.parameters(), lr=0.001)
        best_model = torch.nn.Linear(3, 1)
        best_optimizer = torch.optim.Adam(best_model.parameters(), lr=0.001)

        with patch(
            "utils.learning.resume.torch.cuda.is_available", return_value=False
        ):
            latest = build_training_state(
                20, latest_model, latest_optimizer, 0.1
            )
            best = build_training_state(10, best_model, best_optimizer, 0.1)

        with self.assertRaisesRegex(ValueError, "architecture"):
            resume_module.validate_checkpoint_pair(latest, best)

    def test_checkpoint_pair_rejects_optimizer_topology_mismatch(self):
        latest_model = torch.nn.Linear(2, 1)
        latest_optimizer = torch.optim.Adam(latest_model.parameters(), lr=0.001)
        best_model = torch.nn.Linear(2, 1)
        best_optimizer = torch.optim.Adam(
            [
                {"params": [best_model.weight], "lr": 0.001},
                {"params": [best_model.bias], "lr": 0.001},
            ]
        )
        with patch(
            "utils.learning.resume.torch.cuda.is_available", return_value=False
        ):
            latest = build_training_state(
                20, latest_model, latest_optimizer, 0.1
            )
            best = build_training_state(10, best_model, best_optimizer, 0.1)

        with self.assertRaisesRegex(ValueError, "optimizer.*topology"):
            resume_module.validate_checkpoint_pair(latest, best)

    def test_checkpoint_schema_rejects_bool_and_fractional_epochs(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        legacy = {
            "epoch": 20,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": 0.1068,
        }

        for invalid in (True, 20.5):
            with self.subTest(invalid=invalid):
                invalid_legacy = dict(legacy, epoch=invalid)
                with self.assertRaisesRegex(ValueError, "epoch"):
                    sanitize_legacy_training_state(invalid_legacy)

    def test_checkpoint_schema_rejects_malformed_rng_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with patch(
            "utils.learning.resume.torch.cuda.is_available",
            return_value=False,
        ):
            valid = build_training_state(20, model, optimizer, 0.1068)

        malformed_states = []
        missing_key = copy.deepcopy(valid)
        del missing_key["rng_state"]["torch_cpu"]
        malformed_states.append(missing_key)
        invalid_python_state = copy.deepcopy(valid)
        invalid_python_state["rng_state"]["python_state"] = "invalid"
        malformed_states.append(invalid_python_state)
        invalid_cuda_state = copy.deepcopy(valid)
        invalid_cuda_state["rng_state"]["torch_cuda"] = ["invalid"]
        malformed_states.append(invalid_cuda_state)

        for checkpoint in malformed_states:
            with self.subTest(rng_state=checkpoint["rng_state"]):
                with self.assertRaisesRegex(ValueError, "RNG"):
                    validate_training_checkpoint(checkpoint)

    def test_sanitized_legacy_checkpoint_requires_explicit_inexact_resume(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        legacy = {
            "epoch": 20,
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "best_val_loss": torch.tensor(0.1068),
        }

        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint = Path(tmp) / "safe-legacy.pt"
            torch.save(sanitize_legacy_training_state(legacy), checkpoint)
            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.001)

            with self.assertRaisesRegex(ValueError, "allow-inexact-resume"):
                load_training_state(
                    checkpoint,
                    resumed_model,
                    resumed_optimizer,
                    torch.device("cpu"),
                )
            with self.assertWarnsRegex(RuntimeWarning, "without RNG state"):
                start_epoch, _ = load_training_state(
                    checkpoint,
                    resumed_model,
                    resumed_optimizer,
                    torch.device("cpu"),
                    allow_inexact=True,
                )

        self.assertEqual(start_epoch, 20)

    def test_exact_resume_rejects_missing_cuda_rng_for_selected_device(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint = Path(tmp) / "model.pt"
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                state = build_training_state(
                    20, source_model, source_optimizer, 0.1068
                )
            state["rng_state"]["torch_cuda"] = [torch.get_rng_state()]
            torch.save(state, checkpoint)
            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.Adam(
                resumed_model.parameters(), lr=0.001
            )

            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=True,
            ), patch(
                "utils.learning.resume.torch.cuda.current_device",
                return_value=1,
            ):
                with self.assertRaisesRegex(ValueError, "CUDA RNG.*allow-inexact"):
                    load_training_state(
                        checkpoint,
                        resumed_model,
                        resumed_optimizer,
                        torch.device("cuda:1"),
                    )

    def test_inexact_resume_warns_when_selected_cuda_rng_is_unavailable(self):
        source_model = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            checkpoint = Path(tmp) / "model.pt"
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                state = build_training_state(
                    20, source_model, source_optimizer, 0.1068
                )
            state["rng_state"]["torch_cuda"] = []
            torch.save(state, checkpoint)
            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.Adam(
                resumed_model.parameters(), lr=0.001
            )

            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=True,
            ), patch(
                "utils.learning.resume.torch.cuda.current_device",
                return_value=0,
            ):
                with self.assertWarnsRegex(RuntimeWarning, "CUDA RNG"):
                    start_epoch, _ = load_training_state(
                        checkpoint,
                        resumed_model,
                        resumed_optimizer,
                        torch.device("cuda:0"),
                        allow_inexact=True,
                    )

        self.assertEqual(start_epoch, 20)

    def test_legacy_converter_requires_trust_and_writes_safe_checkpoint(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "convert_trusted_training_checkpoint.py"
        )
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            source = Path(tmp) / "legacy.pt"
            output = Path(tmp) / "resume_model.safe.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            legacy_state = {
                "epoch": 20,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": torch.tensor(0.1068),
            }
            torch.save(legacy_state, source)
            torch.save(legacy_state, source.parent / "best_model.pt")

            refused = subprocess.run(
                [sys.executable, str(script), str(source), str(output)],
                capture_output=True,
                text=True,
            )
            converted = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(source),
                    str(output),
                    "--trusted-input",
                ],
                capture_output=True,
                text=True,
            )
            safe_state = torch.load(output, map_location="cpu", weights_only=True)
            safe_best = torch.load(
                output.parent / "best_model.safe.pt",
                map_location="cpu",
                weights_only=True,
            )

        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("--trusted-input", refused.stderr)
        self.assertEqual(converted.returncode, 0, converted.stderr)
        self.assertIsNone(safe_state["rng_state"])
        self.assertIsNone(safe_best["rng_state"])

    def test_legacy_converter_rejects_colliding_model_and_best_outputs(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "convert_trusted_training_checkpoint.py"
        )
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            source = Path(tmp) / "model.pt"
            output = Path(tmp) / "best_model.safe.pt"
            legacy_state = {
                "epoch": 20,
                "model": {},
                "optimizer": {},
                "best_val_loss": torch.tensor(0.1068),
            }
            torch.save(legacy_state, source)
            torch.save(legacy_state, source.parent / "best_model.pt")

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(source),
                    str(output),
                    "--trusted-input",
                ],
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("distinct", result.stderr)

    def test_legacy_converter_rejects_best_epoch_after_model_epoch(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "convert_trusted_training_checkpoint.py"
        )
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            source = Path(tmp) / "legacy.pt"
            output = Path(tmp) / "resume_model.safe.pt"
            base = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": 0.1,
            }
            torch.save(dict(base, epoch=20), source)
            torch.save(dict(base, epoch=21), source.parent / "best_model.pt")

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(source),
                    str(output),
                    "--trusted-input",
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("best checkpoint epoch", result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse((output.parent / "best_model.safe.pt").exists())

    def test_converter_atomic_publish_refuses_to_overwrite_race_winner(self):
        module = load_converter_module("checkpoint_converter")
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            source = Path(tmp) / "temporary.pt"
            destination = Path(tmp) / "model.safe.pt"
            source.write_text("converted")
            destination.write_text("race-winner")

            with self.assertRaises(FileExistsError):
                module._publish_without_overwrite(source, destination)

            self.assertEqual(destination.read_text(), "race-winner")

    def test_converter_publishes_open_descriptor_not_substituted_temp_path(self):
        module = load_converter_module("checkpoint_converter_descriptor_publish")
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        legacy_state = {
            "epoch": 20,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": 0.1,
        }
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            directory = Path(tmp)
            source = directory / "model.pt"
            source_best = directory / "best_model.pt"
            output = directory / "model.safe.pt"
            torch.save(legacy_state, source)
            torch.save(legacy_state, source_best)
            substituted = sanitize_legacy_training_state(dict(legacy_state, epoch=10))
            real_link = os.link

            def substitute_path_then_link(temporary, destination):
                displaced = directory / f"displaced-{Path(temporary).name}"
                os.replace(temporary, displaced)
                torch.save(substituted, temporary)
                return real_link(temporary, destination)

            with patch.object(
                module,
                "parse_args",
                return_value=SimpleNamespace(
                    input=source,
                    output=output,
                    trusted_input=True,
                ),
            ), patch.object(module.os, "link", side_effect=substitute_path_then_link):
                module.main()

            published = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(published["epoch"], 20)
            self.assertEqual(list(directory.glob("hermes-verify-*")), [])

    def test_converter_uses_anonymous_temporary_file_descriptor(self):
        module = load_converter_module("checkpoint_converter_open_temporary")
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            temporary = module._temporary_path(Path(tmp))
            try:
                self.assertFalse(temporary.closed)
                self.assertGreaterEqual(temporary.fileno(), 0)
                self.assertEqual(os.fstat(temporary.fileno()).st_nlink, 0)
                self.assertEqual(list(Path(tmp).iterdir()), [])
            finally:
                temporary.close()

    def test_converter_interruption_leaks_no_temporary_link_names(self):
        module = load_converter_module("checkpoint_converter_temporary_replacement")
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        legacy_state = {
            "epoch": 20,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": 0.1,
        }
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            directory = Path(tmp)
            source = directory / "model.pt"
            source_best = directory / "best_model.pt"
            output = directory / "model.safe.pt"
            torch.save(legacy_state, source)
            torch.save(legacy_state, source_best)

            def interrupt_before_publish(
                temporary_model,
                _output_model,
                temporary_best,
                _output_best,
            ):
                self.assertEqual(os.fstat(temporary_model.fileno()).st_nlink, 0)
                self.assertEqual(os.fstat(temporary_best.fileno()).st_nlink, 0)
                raise KeyboardInterrupt("fault before descriptor publication")

            with patch.object(
                module,
                "parse_args",
                return_value=SimpleNamespace(
                    input=source,
                    output=output,
                    trusted_input=True,
                ),
            ), patch.object(
                module,
                "_publish_pair_without_overwrite",
                side_effect=interrupt_before_publish,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    module.main()

            self.assertFalse(output.exists())
            self.assertFalse((directory / "best_model.safe.pt").exists())
            self.assertEqual(list(directory.glob("hermes-verify-*")), [])

    def test_converter_pair_publish_keeps_first_output_on_second_collision(self):
        module = load_converter_module("checkpoint_converter_pair")
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            temporary_model = Path(tmp) / "temporary-model.pt"
            temporary_best = Path(tmp) / "temporary-best.pt"
            output_model = Path(tmp) / "model.safe.pt"
            output_best = Path(tmp) / "best_model.safe.pt"
            temporary_model.write_text("model")
            temporary_best.write_text("best")
            output_best.write_text("race-winner")

            with self.assertRaises(FileExistsError):
                module._publish_pair_without_overwrite(
                    temporary_model,
                    output_model,
                    temporary_best,
                    output_best,
                )

            self.assertEqual(output_model.read_text(), "model")
            self.assertEqual(output_best.read_text(), "race-winner")

    def test_converter_pair_publish_keeps_completed_link_when_interrupted(self):
        module = load_converter_module("checkpoint_converter_interrupted_publish")
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            temporary_model = Path(tmp) / "temporary-model.pt"
            temporary_best = Path(tmp) / "temporary-best.pt"
            output_model = Path(tmp) / "model.safe.pt"
            output_best = Path(tmp) / "best_model.safe.pt"
            temporary_model.write_text("model")
            temporary_best.write_text("best")
            real_publish = module._publish_without_overwrite

            def interrupt_after_publish(source, destination):
                real_publish(source, destination)
                raise KeyboardInterrupt("fault after link")

            with patch.object(
                module,
                "_publish_without_overwrite",
                side_effect=interrupt_after_publish,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    module._publish_pair_without_overwrite(
                        temporary_model,
                        output_model,
                        temporary_best,
                        output_best,
                    )

            self.assertEqual(output_model.read_text(), "model")
            self.assertFalse(output_best.exists())

    def test_converter_pair_failure_does_not_use_check_then_unlink(self):
        module = load_converter_module("checkpoint_converter_no_toctou_unlink")
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            temporary_model = Path(tmp) / "temporary-model.pt"
            temporary_best = Path(tmp) / "temporary-best.pt"
            output_model = Path(tmp) / "model.safe.pt"
            output_best = Path(tmp) / "best_model.safe.pt"
            temporary_model.write_text("model")
            temporary_best.write_text("best")
            output_best.write_text("race-winner")

            with patch.object(
                module.os.path,
                "samefile",
                side_effect=AssertionError("unsafe check-then-unlink"),
            ):
                with self.assertRaises(FileExistsError):
                    module._publish_pair_without_overwrite(
                        temporary_model,
                        output_model,
                        temporary_best,
                        output_best,
                    )

            self.assertEqual(output_model.read_text(), "model")
            self.assertEqual(output_best.read_text(), "race-winner")

    def test_converter_rollback_preserves_noncooperating_writer_replacement(self):
        module = load_converter_module("checkpoint_converter_writer_replacement")
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            temporary_model = Path(tmp) / "temporary-model.pt"
            temporary_best = Path(tmp) / "temporary-best.pt"
            output_model = Path(tmp) / "model.safe.pt"
            output_best = Path(tmp) / "best_model.safe.pt"
            displaced_model = Path(tmp) / "displaced-model.pt"
            temporary_model.write_text("model")
            temporary_best.write_text("best")
            real_publish = module._publish_without_overwrite

            def replace_after_first_publish(source, destination):
                real_publish(source, destination)
                os.replace(destination, displaced_model)
                destination.write_text("noncooperating-writer")
                raise KeyboardInterrupt("fault after replacement")

            with patch.object(
                module,
                "_publish_without_overwrite",
                side_effect=replace_after_first_publish,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    module._publish_pair_without_overwrite(
                        temporary_model,
                        output_model,
                        temporary_best,
                        output_best,
                    )

            self.assertEqual(output_model.read_text(), "noncooperating-writer")
            self.assertEqual(displaced_model.read_text(), "model")

    def test_save_model_interrupted_write_preserves_previous_checkpoint(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            checkpoint = exp_dir / "model.pt"
            checkpoint.write_bytes(b"previous-checkpoint")

            def interrupt_save(_state, f):
                if hasattr(f, "write"):
                    f.write(b"partial-checkpoint")
                else:
                    Path(f).write_bytes(b"partial-checkpoint")
                raise KeyboardInterrupt("fault during torch.save")

            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ), patch(
                "utils.learning.train_part.torch.save",
                side_effect=interrupt_save,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    save_model(exp_dir, 21, model, optimizer, 0.106, False)

            self.assertEqual(checkpoint.read_bytes(), b"previous-checkpoint")
            self.assertEqual(list(exp_dir.glob("hermes-checkpoint-*")), [])

    def test_save_model_publishes_open_descriptor_not_substituted_temp_path(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                substituted = build_training_state(10, model, optimizer, 0.1)
            real_replace = os.replace

            def substitute_path_then_replace(source, destination):
                destination = Path(destination)
                if destination.name.endswith("-model.pt"):
                    displaced = exp_dir / "displaced-model-temp.pt"
                    real_replace(source, displaced)
                    torch.save(substituted, source)
                return real_replace(source, destination)

            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ), patch(
                "utils.learning.train_part.os.replace",
                side_effect=substitute_path_then_replace,
            ):
                save_model(exp_dir, 21, model, optimizer, 0.1, True)

            manifest = resume_module._read_checkpoint_manifest(exp_dir)
            published = torch.load(
                exp_dir / manifest["model"], map_location="cpu", weights_only=True
            )
            self.assertEqual(published["epoch"], 21)

    def test_manifest_model_symlink_to_outside_is_rejected(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        resumed_model = torch.nn.Linear(2, 1)
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            exp_dir = root / "checkpoints"
            exp_dir.mkdir()
            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ):
                save_model(exp_dir, 21, model, optimizer, 0.1, True)
                outside_state = build_training_state(10, model, optimizer, 0.1)
            outside = root / "outside.pt"
            torch.save(outside_state, outside)
            manifest = resume_module._read_checkpoint_manifest(exp_dir)
            artifact = exp_dir / manifest["model"]
            artifact.unlink()
            artifact.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "securely open"):
                load_training_state(
                    exp_dir / "model.pt",
                    resumed_model,
                    resumed_optimizer,
                    torch.device("cpu"),
                )

    def test_manifest_rejects_traversal_and_wrong_generation_role(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ):
                save_model(exp_dir, 20, model, optimizer, 0.1, True)
                save_model(exp_dir, 21, model, optimizer, 0.1, False)
            manifest_path = exp_dir / resume_module.CHECKPOINT_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())

            traversing = dict(manifest, model="../outside.pt")
            manifest_path.write_text(json.dumps(traversing))
            with self.assertRaisesRegex(ValueError, "model artifact"):
                resume_module._read_checkpoint_manifest(exp_dir)

            wrong_role = dict(manifest, model=manifest["best"])
            manifest_path.write_text(json.dumps(wrong_role))
            with self.assertRaisesRegex(ValueError, "model artifact"):
                resume_module._read_checkpoint_manifest(exp_dir)

    def test_save_model_interrupted_best_publication_preserves_previous_pair(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            checkpoint = exp_dir / "model.pt"
            best_checkpoint = exp_dir / "best_model.pt"
            checkpoint.write_bytes(b"previous-model")
            best_checkpoint.write_bytes(b"previous-best")
            real_replace = os.replace

            def interrupt_best(source, destination):
                if Path(destination).name == "best_model.pt":
                    raise KeyboardInterrupt("fault before best publication")
                return real_replace(source, destination)

            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ), patch(
                "utils.learning.train_part.os.replace",
                side_effect=interrupt_best,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    save_model(exp_dir, 21, model, optimizer, 0.105, True)

            self.assertEqual(checkpoint.read_bytes(), b"previous-model")
            self.assertEqual(best_checkpoint.read_bytes(), b"previous-best")
            self.assertEqual(list(exp_dir.glob("hermes-checkpoint-*")), [])

    def test_save_model_retains_compatible_pre_manifest_best_for_recovery(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                best_state = build_training_state(10, model, optimizer, 0.1)
                torch.save(best_state, exp_dir / "best_model.pt")
                save_model(exp_dir, 20, model, optimizer, 0.1, False)

            manifest = json.loads(
                (exp_dir / resume_module.CHECKPOINT_MANIFEST_NAME).read_text()
            )
            self.assertIsNotNone(manifest["best"])
            self.assertTrue((exp_dir / manifest["best"]).is_file())

            (exp_dir / "best_model.pt").write_bytes(b"interrupted-alias")
            resume_module.recover_checkpoint_publication(exp_dir)
            recovered_best = torch.load(
                exp_dir / "best_model.pt", map_location="cpu", weights_only=True
            )

            self.assertEqual(recovered_best["epoch"], 10)
            self.assertAlmostEqual(float(recovered_best["best_val_loss"]), 0.1)

    def test_interrupted_pre_manifest_best_seeding_keeps_generation_immutable(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                best_state = build_training_state(10, model, optimizer, 0.1)
                torch.save(best_state, exp_dir / "best_model.pt")
                with patch.object(
                    train_part_module,
                    "_publish_stable_alias",
                    side_effect=KeyboardInterrupt("fault after manifest commit"),
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        save_model(exp_dir, 20, model, optimizer, 0.1, False)

            (exp_dir / "best_model.pt").write_bytes(b"noncooperating-writer")
            resume_module.recover_checkpoint_publication(exp_dir)
            recovered_best = torch.load(
                exp_dir / "best_model.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(recovered_best["epoch"], 10)

    def test_save_model_interrupted_alias_update_is_atomically_recoverable(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                save_model(exp_dir, 20, model, optimizer, 0.2, True)
                real_replace = os.replace

                def interrupt_before_model_alias(source, destination):
                    if Path(destination).name == "model.pt":
                        raise KeyboardInterrupt("fault between stable aliases")
                    return real_replace(source, destination)

                with patch(
                    "utils.learning.train_part.os.replace",
                    side_effect=interrupt_before_model_alias,
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        save_model(exp_dir, 21, model, optimizer, 0.1, True)

            resume_module.recover_checkpoint_publication(exp_dir)
            recovered_model = torch.load(
                exp_dir / "model.pt", map_location="cpu", weights_only=True
            )
            recovered_best = torch.load(
                exp_dir / "best_model.pt", map_location="cpu", weights_only=True
            )

            self.assertEqual(recovered_model["epoch"], 21)
            self.assertEqual(recovered_best["epoch"], 21)
            self.assertEqual(
                float(recovered_model["best_val_loss"]),
                float(recovered_best["best_val_loss"]),
            )

    def test_lower_or_equal_epoch_commit_is_rejected_without_publication_changes(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            experiment = Path(tmp) / "EXP030"
            exp_dir = experiment / "checkpoints"
            exp_dir.mkdir(parents=True)
            history_path = experiment / "val_loss_log.npy"
            history = np.array([[0.0, 0.4], [1.0, 0.3]])
            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ):
                save_model(
                    exp_dir,
                    2,
                    model,
                    optimizer,
                    0.3,
                    True,
                    val_loss_history=history,
                    history_path=history_path,
                )
                before = {
                    path.name: path.read_bytes()
                    for path in exp_dir.iterdir()
                }
                before_history = history_path.read_bytes()

                for stale_epoch, stale_history in (
                    (2, history),
                    (1, history[:1]),
                ):
                    with self.subTest(stale_epoch=stale_epoch):
                        with torch.no_grad():
                            model.weight.add_(1.0)
                        with self.assertRaisesRegex(ValueError, "newer epoch"):
                            save_model(
                                exp_dir,
                                stale_epoch,
                                model,
                                optimizer,
                                0.2,
                                True,
                                val_loss_history=stale_history,
                                history_path=history_path,
                            )
                        self.assertEqual(
                            {
                                path.name: path.read_bytes()
                                for path in exp_dir.iterdir()
                            },
                            before,
                        )
                        self.assertEqual(history_path.read_bytes(), before_history)

    def test_concurrent_generation_saves_and_recovery_keep_manifest_pair_valid(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                save_model(exp_dir, 20, model, optimizer, 0.1, True)

                def save_epoch(epoch):
                    try:
                        save_model(exp_dir, epoch, model, optimizer, 0.1, False)
                    except ValueError as error:
                        if "newer epoch" not in str(error):
                            raise
                        return epoch, False
                    return epoch, True

                def recover(_):
                    return resume_module.recover_checkpoint_publication(exp_dir)

                with ThreadPoolExecutor(max_workers=8) as executor:
                    save_futures = [
                        executor.submit(save_epoch, epoch) for epoch in range(21, 25)
                    ]
                    recovery_futures = [
                        executor.submit(recover, index) for index in range(4)
                    ]
                    save_results = dict(future.result() for future in save_futures)
                    recovery_results = [
                        future.result() for future in recovery_futures
                    ]

            self.assertEqual(set(save_results), {21, 22, 23, 24})
            self.assertTrue(save_results[24])
            self.assertTrue(all(recovery_results))
            manifest = resume_module._read_checkpoint_manifest(exp_dir)
            if manifest is None:
                self.fail("checkpoint manifest disappeared")
            if manifest["best"] is None:
                self.fail("checkpoint manifest lost its best generation")
            model_state = torch.load(
                exp_dir / manifest["model"], map_location="cpu", weights_only=True
            )
            best_state = torch.load(
                exp_dir / manifest["best"], map_location="cpu", weights_only=True
            )
            resume_module.validate_checkpoint_pair(model_state, best_state)
            self.assertEqual(model_state["epoch"], 24)
            self.assertEqual(best_state["epoch"], 20)

            for alias in ("model.pt", "best_model.pt"):
                replacement = exp_dir / f"corrupt-{alias}"
                replacement.write_bytes(b"interrupted-alias")
                os.replace(replacement, exp_dir / alias)
            with ThreadPoolExecutor(max_workers=8) as executor:
                recovered = list(executor.map(recover, range(8)))
            self.assertTrue(all(recovered))

            alias_model = torch.load(
                exp_dir / "model.pt", map_location="cpu", weights_only=True
            )
            alias_best = torch.load(
                exp_dir / "best_model.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(alias_model["epoch"], model_state["epoch"])
            self.assertEqual(alias_best["epoch"], best_state["epoch"])
            self.assertEqual(list(exp_dir.glob("hermes-checkpoint-*")), [])
            self.assertEqual(list(exp_dir.glob(".checkpoint-alias-*")), [])
            self.assertEqual(list(exp_dir.glob(".checkpoint-manifest-*")), [])
            self.assertEqual(
                len(list(exp_dir.glob(".checkpoint-generation-*-model.pt"))),
                1 + sum(save_results.values()),
            )

    def test_manifest_history_is_authoritative_over_stale_compatibility_alias(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = np.column_stack((np.arange(2), np.array([0.4, 0.3])))
        stale = np.column_stack((np.arange(2), np.array([9.0, 8.0])))
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            experiment = Path(tmp) / "EXP030"
            checkpoints = experiment / "checkpoints"
            checkpoints.mkdir(parents=True)

            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ):
                save_model(
                    checkpoints,
                    2,
                    model,
                    optimizer,
                    0.3,
                    True,
                    val_loss_history=history,
                    history_path=experiment / "val_loss_log.npy",
                )
            np.save(experiment / "val_loss_log.npy", stale)

            manifest = resume_module._read_checkpoint_manifest(checkpoints)
            actual = load_val_loss_history(checkpoints / "model.pt", start_epoch=2)
            preserve_best_checkpoint(
                checkpoints / "model.pt",
                Path(tmp) / "destination" / "checkpoints",
            )
            recovered_alias = np.load(
                experiment / "val_loss_log.npy", allow_pickle=False
            )

        self.assertIn("history", manifest)
        self.assertRegex(
            manifest["history"],
            r"^\.checkpoint-generation-[0-9a-f]{32}-history\.npy$",
        )
        np.testing.assert_array_equal(actual, history)
        np.testing.assert_array_equal(recovered_alias, history)

    def test_interruption_before_history_alias_refresh_keeps_committed_history(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        old_history = np.array([[0.0, 0.4]])
        new_history = np.array([[0.0, 0.4], [1.0, 0.3]])
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            experiment = Path(tmp) / "EXP030"
            checkpoints = experiment / "checkpoints"
            checkpoints.mkdir(parents=True)
            alias = experiment / "val_loss_log.npy"
            np.save(alias, old_history)

            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ), patch.object(
                train_part_module,
                "_publish_history_alias",
                side_effect=KeyboardInterrupt("fault after manifest commit"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    save_model(
                        checkpoints,
                        2,
                        model,
                        optimizer,
                        0.3,
                        True,
                        val_loss_history=new_history,
                        history_path=alias,
                    )

            np.testing.assert_array_equal(
                np.load(alias, allow_pickle=False), old_history
            )
            committed = load_val_loss_history(
                checkpoints / "model.pt", start_epoch=2
            )

        np.testing.assert_array_equal(committed, new_history)

    def test_load_val_loss_history_reuses_completed_epochs(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            experiment = Path(tmp) / "EXP030"
            checkpoint = experiment / "checkpoints" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            expected = np.column_stack((np.arange(20), np.linspace(4.0, 3.2, 20)))
            np.save(experiment / "val_loss_log.npy", expected)

            actual = load_val_loss_history(checkpoint, start_epoch=20)

        np.testing.assert_array_equal(actual, expected)

    def test_load_val_loss_history_uses_prefix_for_earlier_best_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            experiment = Path(tmp) / "EXP031"
            checkpoint = experiment / "checkpoints" / "best_model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            complete = np.column_stack((np.arange(30), np.linspace(4.0, 3.1, 30)))
            np.save(experiment / "val_loss_log.npy", complete)

            actual = load_val_loss_history(checkpoint, start_epoch=28)

        np.testing.assert_array_equal(actual, complete[:28])

    def test_load_val_loss_history_rejects_inconsistent_history(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            experiment = Path(tmp) / "EXP030"
            checkpoint = experiment / "checkpoints" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            np.save(
                experiment / "val_loss_log.npy",
                np.array([[0.0, 4.0], [2.0, 3.5]]),
            )

            with self.assertRaisesRegex(ValueError, "validation history"):
                load_val_loss_history(checkpoint, start_epoch=2)

    def test_atomic_history_write_preserves_previous_file_on_interruption(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            history_path = Path(tmp) / "val_loss_log.npy"
            previous = np.array([[0.0, 0.4]])
            updated = np.array([[0.0, 0.4], [1.0, 0.3]])
            np.save(history_path, previous)

            def interrupt_save(_path_or_handle, _array):
                _path_or_handle.write(b"partial-numpy")
                raise KeyboardInterrupt("fault during history serialization")

            with patch(
                "utils.learning.train_part.np.save",
                side_effect=interrupt_save,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    train_part_module.save_val_loss_history(history_path, updated)

            np.testing.assert_array_equal(
                np.load(history_path, allow_pickle=False), previous
            )

    def test_preserve_best_checkpoint_initializes_new_output(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            source = Path(tmp) / "source" / "checkpoints"
            destination = Path(tmp) / "destination" / "checkpoints"
            source.mkdir(parents=True)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                state = build_training_state(20, model, optimizer, 0.1068)
            torch.save(state, source / "model.pt")
            torch.save(state, source / "best_model.pt")

            preserved = preserve_best_checkpoint(source / "model.pt", destination)

            self.assertEqual(preserved, destination / "best_model.pt")
            copied = torch.load(preserved, map_location="cpu", weights_only=True)
            self.assertEqual(copied["epoch"], 20)

    def test_preserve_best_uses_manifest_and_recovers_stale_aliases(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ):
                save_model(source, 20, model, optimizer, 0.1, True)
                save_model(source, 21, model, optimizer, 0.1, False)
                stale = build_training_state(99, model, optimizer, 0.1)
            torch.save(stale, source / "model.pt")
            torch.save(stale, source / "best_model.pt")

            preserved = preserve_best_checkpoint(source / "model.pt", destination)

            copied = torch.load(preserved, map_location="cpu", weights_only=True)
            recovered_model = torch.load(
                source / "model.pt", map_location="cpu", weights_only=True
            )
            recovered_best = torch.load(
                source / "best_model.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(copied["epoch"], 20)
            self.assertEqual(recovered_model["epoch"], 21)
            self.assertEqual(recovered_best["epoch"], 20)
            self.assertEqual(list(destination.glob("hermes-verify-*")), [])

    def test_manifest_null_best_rejects_injected_alias_and_remains_null(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ):
                save_model(source, 20, model, optimizer, 0.1, False)
                injected = build_training_state(10, model, optimizer, 0.1)
            torch.save(injected, source / "best_model.pt")

            with self.assertRaisesRegex(FileNotFoundError, "no best artifact"):
                preserve_best_checkpoint(source / "model.pt", destination)
            with self.assertRaisesRegex(FileNotFoundError, "no best artifact"):
                load_training_state(
                    source / "best_model.pt",
                    torch.nn.Linear(2, 1),
                    torch.optim.Adam(torch.nn.Linear(2, 1).parameters(), lr=0.001),
                    torch.device("cpu"),
                )

            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ):
                save_model(source, 21, model, optimizer, 0.1, False)
            manifest = resume_module._read_checkpoint_manifest(source)
            self.assertIsNone(manifest["best"])
            self.assertFalse(destination.exists())

    def test_preserve_best_rejects_manifest_best_symlink_to_outside(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            with patch(
                "utils.learning.resume.torch.cuda.is_available", return_value=False
            ):
                save_model(source, 20, model, optimizer, 0.1, True)
                outside_state = build_training_state(20, model, optimizer, 0.1)
            outside = root / "outside-best.pt"
            torch.save(outside_state, outside)
            manifest = resume_module._read_checkpoint_manifest(source)
            best_artifact = source / manifest["best"]
            best_artifact.unlink()
            best_artifact.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "securely open"):
                preserve_best_checkpoint(
                    source / "model.pt", root / "destination"
                )

    def test_preserve_best_checkpoint_rejects_incompatible_pair(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            source = Path(tmp) / "source" / "checkpoints"
            destination = Path(tmp) / "destination" / "checkpoints"
            source.mkdir(parents=True)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                model_state = build_training_state(20, model, optimizer, 0.1)
                best_state = build_training_state(10, model, optimizer, 0.2)
            torch.save(model_state, source / "model.pt")
            torch.save(best_state, source / "best_model.pt")

            with self.assertRaisesRegex(ValueError, "same best validation loss"):
                preserve_best_checkpoint(source / "model.pt", destination)

            self.assertFalse(destination.exists())

    def test_copy_without_overwrite_does_not_reopen_temporary_path(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            source = Path(tmp) / "source.pt"
            destination = Path(tmp) / "destination.pt"
            source.write_bytes(b"checkpoint")

            with patch(
                "utils.learning.resume.shutil.copyfile",
                side_effect=AssertionError("temporary pathname was reopened"),
            ):
                _copy_without_overwrite(source, destination)

            self.assertEqual(destination.read_bytes(), b"checkpoint")

    def test_copy_without_overwrite_uses_anonymous_descriptor_without_leak(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            directory = Path(tmp)
            source = directory / "source.pt"
            destination = directory / "destination.pt"
            source.write_bytes(b"checkpoint")

            with patch(
                "utils.learning.resume.os.link",
                side_effect=AssertionError("pathname publication was used"),
            ):
                _copy_without_overwrite(source, destination)

            self.assertEqual(destination.read_bytes(), b"checkpoint")
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                ["destination.pt", "source.pt"],
            )

    def test_save_cleanup_preserves_noncooperating_temporary_replacement(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            exp_dir = Path(tmp)
            real_replace = os.replace
            replacement = None

            def replace_temporary_after_publish(source, destination):
                nonlocal replacement
                real_replace(source, destination)
                if Path(destination).name == "model.pt":
                    replacement = Path(source)
                    replacement.write_bytes(b"noncooperating-writer")

            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ), patch(
                "utils.learning.train_part.os.replace",
                side_effect=replace_temporary_after_publish,
            ):
                save_model(exp_dir, 21, model, optimizer, 0.106, False)

            self.assertEqual(replacement.read_bytes(), b"noncooperating-writer")

    def test_preserve_best_checkpoint_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory(prefix="hermes-verify-") as tmp:
            source = Path(tmp) / "source" / "checkpoints"
            destination = Path(tmp) / "destination" / "checkpoints"
            source.mkdir(parents=True)
            destination.mkdir(parents=True)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            with patch(
                "utils.learning.resume.torch.cuda.is_available",
                return_value=False,
            ):
                state = build_training_state(20, model, optimizer, 0.1068)
            torch.save(state, source / "best_model.pt")
            existing = destination / "best_model.pt"
            existing.write_text("existing")

            with self.assertRaisesRegex(FileExistsError, "destination"):
                preserve_best_checkpoint(source / "best_model.pt", destination)

            self.assertEqual(existing.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
