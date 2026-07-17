import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

if "fcntl" not in sys.modules:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1,
        LOCK_UN=2,
        flock=lambda *_args, **_kwargs: None,
    )

import train as training_entrypoint
from utils.promptmr import planner


class PromptMRPlannerCliTests(unittest.TestCase):
    def test_documented_script_help_runs_without_ambient_pythonpath(self):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)

        completed = subprocess.run(
            [sys.executable, "scripts/plan_promptmr_run.py", "--help"],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Plan a pinned PromptMR+ Vessl run", completed.stdout)

    def test_dependency_status_requires_torch_2_3_or_newer(self):
        with patch.object(
            planner.importlib.util, "find_spec", return_value=object()
        ), patch.object(planner.importlib_metadata, "version", return_value="2.2.2"):
            self.assertFalse(planner._dependency_status()["torch"])

        with patch.object(
            planner.importlib.util, "find_spec", return_value=object()
        ), patch.object(planner.importlib_metadata, "version", return_value="2.3.0"):
            self.assertTrue(planner._dependency_status()["torch"])

    def test_dependency_status_requires_opencv_runtime_import(self):
        def find_spec(name):
            return None if name == "cv2" else object()

        with patch.object(
            planner.importlib.util, "find_spec", side_effect=find_spec
        ), patch.object(planner.importlib_metadata, "version", return_value="2.3.0"):
            status = planner._dependency_status()

        self.assertIn("cv2", status)
        self.assertFalse(status["cv2"])

    def test_root_requirements_pin_supported_torch_floor(self):
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("torch>=2.3", requirements.splitlines())

    def test_commands_match_real_training_options_and_create_nothing(self):
        stats = {
            "path": "unused",
            "volumes": {4: 2, 8: 2},
            "slices": {4: 20, 8: 20},
            "max_sample_bytes": 1024,
            "total_volume_bytes": 4096,
        }
        with tempfile.TemporaryDirectory(prefix="promptmr-plan-cli-") as tmp:
            root = Path(tmp)
            output = root / "does-not-exist" / "result"
            stream = io.StringIO()
            registry = Path("experiments/experiment_log.csv")
            registry_before = registry.read_bytes()
            with patch.object(
                planner, "collect_dataset_stats", side_effect=[stats, stats]
            ), patch.object(
                planner,
                "_dependency_status",
                return_value={
                    "torch": True, "einops": True, "h5py": True,
                    "numpy": True, "skimage": True,
                },
            ), redirect_stdout(stream):
                result = planner.main([
                    "--train-data-path", str(root / "train"),
                    "--val-data-path", str(root / "val"),
                    "--output-parent", str(output),
                    "--epochs", "5",
                    "--license-confirmed",
                ])
            plan = json.loads(stream.getvalue())
            self.assertFalse(output.exists())
            self.assertEqual(registry.read_bytes(), registry_before)

        self.assertEqual(result, 0)
        self.assertFalse(plan["creates_output_directories"])
        self.assertTrue(plan["candidate_launch_ready"])
        self.assertEqual(
            plan["estimate"]["disk_estimate_bytes"],
            2 * ((5 + 2) * (8 * 1024 ** 3) + 4096 * 5 + 5 * 1024 ** 2),
        )
        for family, command in plan["commands"].items():
            tokens = shlex.split(command)
            self.assertEqual(tokens[:3], ["python", "-u", "train.py"])
            self.assertIn("--model-family", tokens)
            self.assertIn("--result-root", tokens)
            self.assertIn("--data-path-train", tokens)
            self.assertIn("--data-path-val", tokens)
            self.assertIn("--retain-val-epochs", tokens)
            self.assertNotIn("...", tokens)
            if family == "candidate":
                self.assertIn("--confirm-promptmr-noncommercial-use", tokens)
                self.assertIn("promptmr_plus", tokens)
                namespace = training_entrypoint.parse(tokens[3:])
                self.assertEqual(namespace.model_family, "promptmr_plus")
                self.assertEqual(namespace.net_name, Path(
                    "EXP036_promptmr_plus_default_e5_seed430"
                ))
                self.assertEqual(namespace.result_root, output)
                self.assertEqual(namespace.seed, 430)
                self.assertEqual(namespace.batch_size, 1)
                self.assertEqual(namespace.lr, 1e-4)
                self.assertTrue(namespace.confirm_promptmr_noncommercial_use)
            else:
                self.assertIn("varnet", tokens)
                namespace = training_entrypoint.parse(tokens[3:])
                self.assertEqual(namespace.model_family, "varnet")
                self.assertEqual(namespace.result_root, output)
                self.assertEqual(namespace.seed, 430)
                self.assertEqual(namespace.batch_size, 1)
                self.assertEqual(namespace.lr, 1e-3)
                self.assertEqual(
                    namespace.model_contract, {"model_family": "varnet"}
                )

    def test_unconfirmed_plan_keeps_candidate_launch_blocked(self):
        stats = {
            "path": "unused",
            "volumes": {4: 1, 8: 1},
            "slices": {4: 1, 8: 1},
            "max_sample_bytes": 1,
            "total_volume_bytes": 1,
        }
        stream = io.StringIO()
        with patch.object(
            planner, "collect_dataset_stats", side_effect=[stats, stats]
        ), redirect_stdout(stream):
            planner.main([
                "--train-data-path", ".",
                "--val-data-path", ".",
            ])
        plan = json.loads(stream.getvalue())
        self.assertFalse(plan["candidate_launch_ready"])
        self.assertNotIn(
            "--confirm-promptmr-noncommercial-use",
            shlex.split(plan["commands"]["candidate"]),
        )
        with self.assertRaises(SystemExit):
            training_entrypoint.parse(
                shlex.split(plan["commands"]["candidate"])[3:]
            )

    def test_explicit_one_epoch_promptmr_canary_is_preserved(self):
        args = training_entrypoint.parse([
            "--model-family", "promptmr_plus",
            "--confirm-promptmr-noncommercial-use",
            "--num-epochs", "1",
        ])
        self.assertEqual(args.num_epochs, 1)

    def test_train_and_planner_reject_special_path_run_names(self):
        unsafe_names = (
            ".", "..", "C:", "CON", "con.txt", "CON .txt",
            "COM1 .log", "NUL  .bin", "COM¹", "COM²", "COM³",
            "LPT¹", "LPT²", "LPT³", "foo:bar", "foo.", "foo "
        )
        for value in unsafe_names:
            with self.subTest(parser="train", value=value), self.assertRaises(SystemExit):
                training_entrypoint.parse(["--net-name", value])
            for option in ("--control-run-name", "--candidate-run-name"):
                with self.subTest(parser="planner", option=option, value=value), self.assertRaises(SystemExit):
                    planner.build_parser().parse_args([
                        "--train-data-path", "train",
                        "--val-data-path", "val",
                        option, value,
                    ])

    def test_promptmr_rejects_inexact_resume(self):
        with self.assertRaises(SystemExit):
            training_entrypoint.parse([
                "--model-family", "promptmr_plus",
                "--confirm-promptmr-noncommercial-use",
                "--resume-checkpoint", "model.pt",
                "--allow-inexact-resume",
            ])


    def test_planner_rejects_batch_size_that_candidate_parser_rejects(self):
        with self.assertRaises(SystemExit):
            planner.build_parser().parse_args([
                "--train-data-path", ".",
                "--val-data-path", ".",
                "--batch-size", "2",
            ])

    def test_missing_dependency_blocks_confirmed_candidate(self):
        stats = {
            "path": "unused", "volumes": {4: 1, 8: 1},
            "slices": {4: 1, 8: 1}, "max_sample_bytes": 1,
            "total_volume_bytes": 1,
        }
        stream = io.StringIO()
        with patch.object(
            planner, "collect_dataset_stats", side_effect=[stats, stats]
        ), patch.object(
            planner,
            "_dependency_status",
            return_value={"torch": True, "einops": False},
        ), redirect_stdout(stream):
            planner.main([
                "--train-data-path", ".", "--val-data-path", ".",
                "--license-confirmed",
            ])
        self.assertFalse(json.loads(stream.getvalue())["candidate_launch_ready"])


if __name__ == "__main__":
    unittest.main()
