import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import stat
import subprocess
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_promptmr_plus_8gb.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("probe_promptmr_plus_8gb", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptMRPlusProbeTests(unittest.TestCase):
    def test_selects_slice_and_retained_volume_cases_independently(self):
        module = load_probe_module()
        cases = [
            {"shape": (10, 15, 640, 480), "name": "slice"},
            {"shape": (40, 15, 640, 368), "name": "volume"},
        ]
        selected = module.choose_cases(cases)
        self.assertEqual(selected["maximum_slice_input"]["name"], "slice")
        self.assertEqual(selected["maximum_retained_volume"]["name"], "volume")

    def test_synthetic_cases_match_official_shapes_without_h5(self):
        module = load_probe_module()
        for acceleration in (4, 8):
            with self.subTest(acceleration=acceleration):
                selected = module.synthetic_cases(acceleration)
                slice_case = selected["maximum_slice_input"]
                volume_case = selected["maximum_retained_volume"]
                self.assertEqual(
                    slice_case["shape"][1:], module.EXPECTED_SLICE_SHAPES[acceleration]
                )
                self.assertEqual(
                    (volume_case["shape"][0], *volume_case["shape"][2:]),
                    module.EXPECTED_VOLUME_SHAPES[acceleration],
                )
                with mock.patch.object(
                    module.h5py, "File", side_effect=AssertionError("H5 opened")
                ):
                    volume, mask = module.load_case_volume(volume_case)
                self.assertEqual(volume.shape, volume_case["shape"])
                self.assertEqual(volume.dtype, module.np.complex64)
                self.assertEqual(mask.shape, (volume_case["shape"][3],))
                self.assertEqual(mask.dtype, module.np.bool_)
                self.assertTrue(mask.any())
                self.assertEqual(volume.strides[0], 0)

    def test_device_level_override_requires_synthetic_random_exact_gpu(self):
        module = load_probe_module()
        valid = types.SimpleNamespace(
            telemetry_mode="device-level-unattributed",
            input_source="synthetic",
            expected_gpu_index=0,
            expected_gpu_uuid="GPU-abc-123",
            checkpoint=None,
            checkpoint_kind=None,
        )
        module.validate_probe_mode(valid)
        for field, value in (
            ("input_source", "actual-h5"),
            ("expected_gpu_index", 1),
            ("expected_gpu_uuid", None),
            ("checkpoint", Path("weights.ckpt")),
        ):
            invalid = types.SimpleNamespace(**vars(valid))
            setattr(invalid, field, value)
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(RuntimeError, "device-level override"):
                    module.validate_probe_mode(invalid)

    def test_runtime_and_unsupported_failures_are_distinct(self):
        module = load_probe_module()
        self.assertEqual(
            module.classify_failure(
                module.PromptMRContractError("returned an invalid real image")
            ),
            "FAIL_SHAPE_CONTRACT",
        )
        self.assertEqual(
            module.classify_failure(
                module.PromptMRNonFiniteError("invalid numeric output")
            ),
            "FAIL_NONFINITE",
        )
        self.assertEqual(
            module.classify_failure(RuntimeError("ordinary failure")), "FAIL_RUNTIME"
        )
        self.assertEqual(
            module.classify_failure(NotImplementedError("kernel")),
            "FAIL_UNSUPPORTED_OPERATION",
        )
        self.assertEqual(
            module.classify_failure(torch.cuda.OutOfMemoryError("oom")), "FAIL_CUDA_OOM"
        )
        self.assertEqual(module.classify_failure(MemoryError()), "FAIL_HOST_OOM")
        self.assertEqual(
            module.classify_failure(RuntimeError("non-finite PromptMR+ output")),
            "FAIL_NONFINITE",
        )
        self.assertEqual(
            module.classify_failure(RuntimeError("unexpected retained output shape")),
            "FAIL_SHAPE_CONTRACT",
        )
        self.assertEqual(module.classify_failure(OSError("read failed")), "FAIL_DATA_IO")
        self.assertEqual(
            module.classify_failure(RuntimeError("GPU is not exclusive to the probe")),
            "FAIL_GPU_BUSY",
        )
        self.assertEqual(
            module.classify_failure(
                RuntimeError("required current-process GPU memory evidence unavailable")
            ),
            "FAIL_MEMORY_EVIDENCE",
        )

    def test_process_gpu_memory_evidence_is_required(self):
        module = load_probe_module()
        with mock.patch.object(
            module.subprocess, "check_output", return_value="999, 123\n"
        ), mock.patch.object(module.os, "getpid", return_value=1000):
            with self.assertRaisesRegex(RuntimeError, "required current-process"):
                module.process_gpu_memory_mib()
        for failure in (
            FileNotFoundError("nvidia-smi"),
            subprocess.CalledProcessError(1, ["nvidia-smi"]),
        ):
            with self.subTest(failure=type(failure).__name__), mock.patch.object(
                module.subprocess, "check_output", side_effect=failure
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU memory evidence") as caught:
                    module.process_gpu_memory_mib()
                self.assertEqual(
                    module.classify_failure(caught.exception), "FAIL_MEMORY_EVIDENCE"
                )
        for malformed in (
            "not,a,valid,row\n",
            f"{os.getpid()}, nan\n",
            f"{os.getpid()}, -1\n",
            "0, 2\n",
        ):
            with self.subTest(malformed=malformed), mock.patch.object(
                module.subprocess, "check_output", return_value=malformed
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU memory evidence") as caught:
                    module.process_gpu_memory_mib()
                self.assertEqual(
                    module.classify_failure(caught.exception), "FAIL_MEMORY_EVIDENCE"
                )

    def test_process_gpu_memory_evidence_rejects_duplicate_pid_rows(self):
        module = load_probe_module()
        for duplicated in (
            f"{os.getpid()}, 123\n{os.getpid()}, 123\n",
            f"{os.getpid()}, 123\n{os.getpid()}, 456\n",
        ):
            with self.subTest(duplicated=duplicated), mock.patch.object(
                module.subprocess, "check_output", return_value=duplicated
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU memory evidence") as caught:
                    module.process_gpu_memory_mib()
                self.assertEqual(
                    module.classify_failure(caught.exception), "FAIL_MEMORY_EVIDENCE"
                )

    def test_gpu_preflight_precedes_every_torch_cuda_call(self):
        module = load_probe_module()
        source = inspect.getsource(module.run_probe)
        preflight = source.index("preflight_telemetry(args)")
        first_torch_cuda = source.index("torch.cuda.")
        self.assertLess(preflight, first_torch_cuda)
        self.assertNotIn("assert_exclusive_gpu_process()", source)

    def test_gpu_idle_gate_rejects_other_compute_processes(self):
        module = load_probe_module()
        with mock.patch.object(
            module.subprocess, "check_output", return_value="999, 123\n"
        ), mock.patch.object(module.os, "getpid", return_value=1000):
            with self.assertRaisesRegex(RuntimeError, "GPU is not exclusive"):
                module.assert_exclusive_gpu_process()

    def test_device_level_snapshot_is_uuid_bound_unattributed_and_strict(self):
        module = load_probe_module()
        gpu = types.SimpleNamespace(
            returncode=0,
            stdout="0, GPU-abc-123, 8192, 128, 7\n",
            stderr="",
        )
        processes = types.SimpleNamespace(
            returncode=0,
            stdout="GPU-abc-123, 64\n",
            stderr="",
        )
        with mock.patch.object(
            module.subprocess, "run", side_effect=(gpu, processes)
        ):
            snapshot = module.device_level_gpu_snapshot(
                expected_gpu_index=0,
                expected_gpu_uuid="GPU-abc-123",
                expected_compute_rows=1,
            )
        self.assertEqual(snapshot["evidence_label"], "DEVICE_LEVEL_UNATTRIBUTED")
        self.assertEqual(snapshot["compute_row_count"], 1)
        self.assertNotIn("pid", json.dumps(snapshot).lower())
        self.assertEqual(snapshot["command_exit_codes"], [0, 0])

        invalid_pairs = (
            ("0, GPU-abc-123, 8192, nan, 0\n", ""),
            ("0, GPU-wrong, 8192, 2, 0\n", ""),
            ("0, GPU-abc-123, 8192, 2, 101\n", ""),
            ("0, GPU-abc-123, 8192, 2, 0\n", "GPU-abc-123, -1\n"),
            ("0, GPU-abc-123, 8192, 2, 0\n", "GPU-abc-123, 1\nGPU-abc-123, 2\n"),
        )
        for gpu_stdout, process_stdout in invalid_pairs:
            with self.subTest(gpu=gpu_stdout, processes=process_stdout):
                responses = (
                    types.SimpleNamespace(returncode=0, stdout=gpu_stdout, stderr=""),
                    types.SimpleNamespace(returncode=0, stdout=process_stdout, stderr=""),
                )
                with mock.patch.object(module.subprocess, "run", side_effect=responses):
                    with self.assertRaisesRegex(RuntimeError, "device-level GPU evidence"):
                        module.device_level_gpu_snapshot(
                            expected_gpu_index=0,
                            expected_gpu_uuid="GPU-abc-123",
                            expected_compute_rows=1,
                        )

    def test_device_level_preflight_uses_zero_row_aggregate_only(self):
        module = load_probe_module()
        args = types.SimpleNamespace(
            telemetry_mode="device-level-unattributed",
            input_source="synthetic",
            expected_gpu_index=0,
            expected_gpu_uuid="GPU-abc-123",
            checkpoint=None,
            checkpoint_kind=None,
        )
        idle = {
            "evidence_label": "DEVICE_LEVEL_UNATTRIBUTED",
            "gpu_index": 0,
            "gpu_uuid": "GPU-abc-123",
            "memory_total_mib": 8192.0,
            "memory_used_mib": 2.0,
            "utilization_percent": 0.0,
            "compute_row_count": 0,
        }
        with mock.patch.object(
            module, "device_level_gpu_snapshot", return_value=idle
        ) as snapshot, mock.patch.object(
            module,
            "assert_exclusive_gpu_process",
            side_effect=AssertionError("PID fallback used"),
        ):
            self.assertEqual(module.preflight_telemetry(args), idle)
        snapshot.assert_called_once_with(
            expected_gpu_index=0,
            expected_gpu_uuid="GPU-abc-123",
            expected_compute_rows=0,
        )
        for field, value in (("utilization_percent", 1.0), ("memory_used_mib", 64.0)):
            busy = dict(idle)
            busy[field] = value
            with self.subTest(field=field), mock.patch.object(
                module, "device_level_gpu_snapshot", return_value=busy
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU is not idle"):
                    module.preflight_telemetry(args)

    def test_gpu_idle_gate_requires_zero_utilization_and_low_baseline_memory(self):
        module = load_probe_module()
        with mock.patch.object(
            module.subprocess, "check_output", return_value="0, 2\n"
        ):
            self.assertEqual(
                module.assert_gpu_idle_before_probe(),
                {"utilization_percent": 0.0, "memory_used_mib": 2.0},
            )
        for failure in (
            FileNotFoundError("nvidia-smi"),
            subprocess.CalledProcessError(1, ["nvidia-smi"]),
        ):
            with self.subTest(failure=type(failure).__name__), mock.patch.object(
                module.subprocess, "check_output", side_effect=failure
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU idle evidence") as caught:
                    module.assert_gpu_idle_before_probe()
                self.assertEqual(
                    module.classify_failure(caught.exception),
                    "FAIL_ENVIRONMENT_EVIDENCE",
                )
        for evidence in (
            "1, 2\n",
            "0, 64\n",
            "0, nan\n",
            "nan, 0\n",
            "0, -1\n",
            "101, 0\n",
        ):
            with self.subTest(evidence=evidence), mock.patch.object(
                module.subprocess, "check_output", return_value=evidence
            ):
                expected = (
                    "GPU is not idle"
                    if evidence in ("1, 2\n", "0, 64\n")
                    else "GPU idle evidence"
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    module.assert_gpu_idle_before_probe()

    def test_driver_evidence_failure_is_distinct(self):
        module = load_probe_module()
        properties = types.SimpleNamespace(
            name="NVIDIA GeForce GTX 1080",
            total_memory=8192 * 2**20,
            major=6,
            minor=1,
        )
        with mock.patch.object(
            module.torch.cuda, "get_device_properties", return_value=properties
        ), mock.patch.object(
            module.subprocess, "check_output", return_value="535.104.05\n"
        ):
            self.assertEqual(module.environment()["driver_version"], "535.104.05")
        for failure in (
            FileNotFoundError("nvidia-smi"),
            "garbage\n",
            "nan\n",
            "\n",
        ):
            with self.subTest(failure=repr(failure)), mock.patch.object(
                module.torch.cuda, "get_device_properties", return_value=properties
            ), mock.patch.object(
                module.subprocess,
                "check_output",
                side_effect=failure if isinstance(failure, Exception) else None,
                return_value=None if isinstance(failure, Exception) else failure,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "driver-version evidence"
                ) as caught:
                    module.environment()
                self.assertEqual(
                    module.classify_failure(caught.exception),
                    "FAIL_ENVIRONMENT_EVIDENCE",
                )

    def test_report_creation_is_exclusive_private_and_rejects_symlink_parent(self):
        module = load_probe_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "result.json"
            module.write_report_exclusive(path, {"status": "PASS"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                module.write_report_exclusive(path, {"status": "PASS"})

            real_parent = root / "real"
            real_parent.mkdir()
            symlink_parent = root / "redirected"
            symlink_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "unsafe output path"):
                module.write_report_exclusive(
                    symlink_parent / "leak.json", {"status": "FAIL_RUNTIME"}
                )
            self.assertFalse((real_parent / "leak.json").exists())

    def test_report_serialization_rejects_nonfinite_values_before_publication(self):
        module = load_probe_module()
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate((float("nan"), float("inf"), float("-inf"))):
                output = Path(directory) / f"nonfinite-{index}.json"
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        module.write_report_exclusive(output, {"metric": value})
                    self.assertFalse(output.exists())

    def test_checkpoint_must_match_official_manifest_hash(self):
        module = load_probe_module()
        manifest = module.verify_promptmr_plus_source()
        self.assertEqual(
            module.OFFICIAL_CHECKPOINT_SHA256,
            {
                "brain": manifest["checkpoints"]["fastmri_brain_promptmr_plus"]["sha256"],
                "knee": manifest["checkpoints"]["fastmri_knee_promptmr_plus"]["sha256"],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "arbitrary.ckpt"
            checkpoint.write_bytes(b"arbitrary pickle bytes")
            args = types.SimpleNamespace(
                checkpoint=checkpoint,
                checkpoint_kind="brain",
            )
            with self.assertRaisesRegex(RuntimeError, "official checkpoint checksum"):
                module.checkpoint_metadata(
                    args,
                    torch.nn.Linear(2, 2),
                    module.verify_promptmr_plus_source(),
                    module.initial_probe_state(args),
                )

            symlink = Path(directory) / "redirected.ckpt"
            symlink.symlink_to(checkpoint)
            args.checkpoint = symlink
            with self.assertRaisesRegex(RuntimeError, "unsafe official checkpoint"):
                module.checkpoint_metadata(
                    args,
                    torch.nn.Linear(2, 2),
                    module.verify_promptmr_plus_source(),
                    module.initial_probe_state(args),
                )

            args.checkpoint = checkpoint
            args.checkpoint_kind = None
            self.assertEqual(
                module.initial_probe_state(args),
                {
                    "weights_requested": "checkpoint_supplied_without_required_kind",
                    "checkpoint_loaded": False,
                    "required_process_gpu_memory_evidence_present": False,
                },
            )

    def test_checkpoint_loaded_state_becomes_true_after_strict_same_fd_load(self):
        module = load_probe_module()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "official.ckpt"
            torch.save(
                {"state_dict": {"promptmr.weight": torch.ones(2, 2)}},
                checkpoint_path,
            )
            manifest = module.verify_promptmr_plus_source()
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            manifest["checkpoints"]["fastmri_brain_promptmr_plus"][
                "sha256"
            ] = digest
            args = types.SimpleNamespace(
                checkpoint=checkpoint_path,
                checkpoint_kind="brain",
            )
            state = module.initial_probe_state(args)
            with mock.patch.dict(
                module.OFFICIAL_CHECKPOINT_SHA256, {"brain": digest}
            ):
                module.checkpoint_metadata(
                    args,
                    torch.nn.Linear(2, 2, bias=False),
                    manifest,
                    state,
                )
            self.assertTrue(state["checkpoint_loaded"])

    def test_each_slice_output_shape_is_typed_and_gated(self):
        module = load_probe_module()
        module.validate_output_shape(
            torch.zeros(1, 64, 64), (1, 64, 64), "maximum-slice"
        )
        with self.assertRaises(module.PromptMRContractError) as caught:
            module.validate_output_shape(
                torch.zeros(2, 64, 64), (1, 64, 64), "full-volume slice"
            )
        self.assertEqual(
            module.classify_failure(caught.exception), "FAIL_SHAPE_CONTRACT"
        )
        source = SCRIPT.read_text()
        self.assertGreaterEqual(source.count("validate_output_shape("), 3)

    def test_every_maximum_slice_output_is_shape_gated_before_transfer_or_release(self):
        source = inspect.getsource(load_probe_module().run_probe)

        warmup_start = source.index("warmup = adapter(")
        warmup_release = source.index("del warmup", warmup_start)
        warmup_region = source[warmup_start:warmup_release]
        self.assertIn("validate_output_shape(\n            warmup", warmup_region)

        harness_start = source.index("harness_output_gpu = adapter(")
        harness_transfer = source.index("harness_output_gpu.cpu()", harness_start)
        harness_region = source[harness_start:harness_transfer]
        self.assertIn(
            "validate_output_shape(\n            harness_output_gpu", harness_region
        )

    def test_main_redacts_report_publication_failure_path(self):
        module = load_probe_module()
        secret_path = Path("/secret/result/path.json")
        args = types.SimpleNamespace(
            acceleration=4,
            checkpoint=None,
            checkpoint_kind=None,
            compute_sens_per_coil=False,
            use_checkpoint=False,
            output=secret_path,
        )
        stderr = io.StringIO()
        with mock.patch.object(module, "parse_args", return_value=args), mock.patch.object(
            module, "run_probe", return_value={"status": "PASS"}
        ), mock.patch.object(
            module,
            "write_report_exclusive",
            side_effect=FileExistsError(str(secret_path)),
        ), contextlib.redirect_stderr(stderr):
            self.assertEqual(module.main(), 2)
        self.assertNotIn(str(secret_path), stderr.getvalue())
        self.assertIn("FAIL_REPORT_PUBLICATION", stderr.getvalue())

    def test_main_returns_nonzero_for_published_probe_failure(self):
        module = load_probe_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure.json"
            args = types.SimpleNamespace(
                acceleration=4,
                checkpoint=None,
                checkpoint_kind=None,
                compute_sens_per_coil=False,
                use_checkpoint=False,
                output=output,
            )
            stdout = io.StringIO()
            with mock.patch.object(module, "parse_args", return_value=args), mock.patch.object(
                module, "run_probe", side_effect=RuntimeError("ordinary failure")
            ), contextlib.redirect_stdout(stdout):
                self.assertEqual(module.main(), 1)
            self.assertEqual(json.loads(output.read_text())["status"], "FAIL_RUNTIME")

    def test_probe_contains_no_training_or_official_evaluation_calls(self):
        source = SCRIPT.read_text()
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer", source.lower())
        self.assertNotIn("recon_eval", source)
        self.assertIn("torch.inference_mode()", source)
        self.assertIn("h5py.File(path, \"r\")", source)
        self.assertNotIn('"error": str(error)', source)
        self.assertNotIn("--checkpoint-sha256", source)
        self.assertIn("adapter_forward_latency_scope", source)
        self.assertIn("single_slice_harness_latency_scope", source)
        self.assertIn("full_volume_retention_latency_scope", source)
        self.assertNotIn('"harness_like_latency_scope"', source)
        self.assertIn("validate_output_shape(", source)
        self.assertIn('"maximum-slice"', source)
        self.assertIn('"full-volume slice"', source)
        self.assertIn("item_output.detach().cpu()", source)
        self.assertIn("retained_host = retained.detach().cpu()", source)
        self.assertIn("assert_exclusive_gpu_process()", source)
        self.assertIn('"driver_version"', source)
        self.assertNotIn("8192.0 - volume_reserved", source)


if __name__ == "__main__":
    unittest.main()
