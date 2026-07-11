import copy
import errno
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "average_checkpoints.py"


def load_average_module():
    spec = importlib.util.spec_from_file_location("average_checkpoints", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load averaging script from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(model, *, epoch=1, metadata=None):
    state = {
        "format_version": 1,
        "epoch": epoch,
        "model": model,
        "optimizer": {
            "state": {},
            "param_groups": [{"params": list(range(len(model)))}],
        },
        "best_val_loss": 0.1,
        "rng_state": None,
    }
    if metadata is not None:
        state["metadata"] = metadata
    return state


class AverageCheckpointsTests(unittest.TestCase):
    def test_missing_parent_chain_sync_failure_stops_before_publication(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            publication_directory = root / "new-parent" / "new-child"
            output = publication_directory / "average.pt"
            manifest = publication_directory / "average.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)
            real_fsync = module.os.fsync
            synced_parents = []

            def fail_second_parent_sync(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    synced_path = Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()
                    expected_parents = [root.resolve(), (root / "new-parent").resolve()]
                    if synced_path in expected_parents:
                        synced_parents.append(synced_path)
                    if synced_path == expected_parents[1]:
                        raise OSError(errno.EIO, "injected new-child parent sync failure")
                return real_fsync(fd)

            with patch.object(module.os, "fsync", side_effect=fail_second_parent_sync):
                with self.assertRaisesRegex(
                    module.PublicationIndeterminateError, "parent.*sync failed"
                ):
                    module.average_checkpoints(
                        [first, second], output, manifest_path=manifest
                    )

            self.assertTrue((root / "new-parent").is_dir())
            self.assertTrue((root / "new-parent" / "new-child").is_dir())
            self.assertEqual(
                synced_parents,
                [root.resolve(), (root / "new-parent").resolve()],
            )
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())

    def test_manifest_is_authoritative_pointer_to_immutable_generation(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            generation = root / "average-generation.pt"
            manifest_path = root / "average-current.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)

            module.average_checkpoints(
                [first, second], generation, manifest_path=manifest_path
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["artifact"],
                {
                    "path": generation.name,
                    "sha256": hashlib.sha256(generation.read_bytes()).hexdigest(),
                },
            )
            self.assertTrue(manifest["committed"])
            averaged = torch.load(generation, map_location="cpu", weights_only=True)
            self.assertEqual(
                averaged["averaging_provenance"],
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in {"artifact", "committed"}
                },
            )

    def test_interruption_after_generation_publish_leaves_recoverable_orphan(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            generation = root / "average-generation.pt"
            manifest = root / "average-current.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)
            publish = module._publish_temporary_without_overwrite
            publications = 0

            def interrupt_manifest(temporary, directory_fd, destination_name):
                nonlocal publications
                publications += 1
                if publications == 2:
                    raise KeyboardInterrupt("injected crash before manifest")
                return publish(temporary, directory_fd, destination_name)

            with patch.object(
                module,
                "_publish_temporary_without_overwrite",
                side_effect=interrupt_manifest,
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "crash before manifest"):
                    module.average_checkpoints(
                        [first, second], generation, manifest_path=manifest
                    )

            self.assertTrue(generation.is_file())
            self.assertFalse(manifest.exists())
            averaged = torch.load(generation, map_location="cpu", weights_only=True)
            self.assertTrue(averaged["inference_only"])

    def test_retry_recovers_matching_orphan_generation(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            generation = root / "average-generation.pt"
            manifest = root / "average-current.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)
            real_publish = module._publish_temporary_without_overwrite
            publications = 0

            def crash_before_manifest(temporary, directory_fd, destination_name):
                nonlocal publications
                publications += 1
                if publications == 2:
                    raise OSError("injected crash before manifest")
                return real_publish(temporary, directory_fd, destination_name)

            with patch.object(
                module,
                "_publish_temporary_without_overwrite",
                side_effect=crash_before_manifest,
            ):
                with self.assertRaisesRegex(OSError, "crash before manifest"):
                    module.average_checkpoints(
                        [first, second], generation, manifest_path=manifest
                    )

            original_generation_bytes = generation.read_bytes()
            module.average_checkpoints(
                [first, second], generation, manifest_path=manifest
            )

            self.assertEqual(generation.read_bytes(), original_generation_bytes)
            committed = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(committed["artifact"]["path"], generation.name)
            self.assertEqual(
                committed["artifact"]["sha256"],
                hashlib.sha256(original_generation_bytes).hexdigest(),
            )

    def test_retry_of_committed_generation_is_idempotent(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            generation = root / "average-generation.pt"
            manifest = root / "average-current.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)

            module.average_checkpoints(
                [first, second], generation, manifest_path=manifest
            )
            generation_bytes = generation.read_bytes()
            manifest_bytes = manifest.read_bytes()

            module.average_checkpoints(
                [first, second], generation, manifest_path=manifest
            )

            self.assertEqual(generation.read_bytes(), generation_bytes)
            self.assertEqual(manifest.read_bytes(), manifest_bytes)

    def test_cli_help_documents_sources_and_named_outputs(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=SCRIPT.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout)
        self.assertIn("--output", completed.stdout)
        self.assertIn("--manifest", completed.stdout)
        self.assertIn("--weights", completed.stdout)
        self.assertIn("--template", completed.stdout)
        self.assertIn("at least two", " ".join(completed.stdout.split()))

    def test_cli_rejects_fewer_than_two_sources_with_usage(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "only.pt",
                "--output",
                "average.pt",
                "--manifest",
                "average.json",
            ],
            cwd=SCRIPT.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("usage:", completed.stderr)
        self.assertIn("at least two source checkpoints", completed.stderr)

    def test_cli_writes_requested_checkpoint_and_manifest_with_options(self):
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-cli-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            manifest = root / "provenance.json"
            torch.save(
                checkpoint(
                    {"weight": torch.tensor([0.0])},
                    metadata={"template": "first"},
                ),
                first,
            )
            torch.save(
                checkpoint(
                    {"weight": torch.tensor([8.0])},
                    metadata={"template": "second"},
                ),
                second,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(first),
                    str(second),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--weights",
                    "1",
                    "3",
                    "--template",
                    str(second),
                ],
                cwd=SCRIPT.parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(manifest.is_file())
            averaged = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(averaged["model"]["weight"], torch.tensor([6.0]))
            )
            self.assertEqual(averaged["metadata"], {"template": "second"})
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["sources"][1][
                    "weight"
                ],
                0.75,
            )

    def test_cli_reports_runtime_errors_without_tracebacks_or_outputs(self):
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-cli-") as tmp:
            root = Path(tmp)
            output = root / "average.pt"
            manifest = root / "average.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(root / "missing-one.pt"),
                    str(root / "missing-two.pt"),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                ],
                cwd=SCRIPT.parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("usage:", completed.stderr)
            self.assertIn("averaging failed:", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())

    def test_averages_floating_tensors_in_float64_and_restores_dtype(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            torch.save(
                checkpoint({"weight": torch.tensor([1.0, 3.0], dtype=torch.float32)}),
                first,
            )
            torch.save(
                checkpoint({"weight": torch.tensor([3.0, 7.0], dtype=torch.float32)}),
                second,
            )

            module.average_checkpoints([first, second], output)

            averaged = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(averaged["model"]["weight"], torch.tensor([2.0, 5.0]))
            )
            self.assertEqual(averaged["model"]["weight"].dtype, torch.float32)

    def test_rejects_nonfinite_floating_model_tensors(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            finite = root / "finite.pt"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), finite)

            for label, value in (("nan", float("nan")), ("inf", float("inf"))):
                with self.subTest(value=label):
                    nonfinite = root / f"{label}.pt"
                    generation = root / f"{label}-generation.pt"
                    manifest = root / f"{label}-current.json"
                    torch.save(
                        checkpoint({"weight": torch.tensor([value])}), nonfinite
                    )

                    with self.assertRaisesRegex(
                        ValueError, "nonfinite.*weight|weight.*nonfinite"
                    ):
                        module.average_checkpoints(
                            [finite, nonfinite], generation, manifest_path=manifest
                        )

                    self.assertFalse(generation.exists())
                    self.assertFalse(manifest.exists())

    def test_requires_two_sources_and_validates_weights(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            torch.save(checkpoint({"weight": torch.tensor([0.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([10.0])}), second)

            with self.assertRaisesRegex(ValueError, "at least two"):
                module.average_checkpoints([first], root / "one.pt")
            with self.assertRaisesRegex(ValueError, "weights"):
                module.average_checkpoints(
                    [first, second], root / "bad.pt", weights=[1.0]
                )

            output = root / "weighted.pt"
            module.average_checkpoints(
                [first, second], output, weights=[1.0, 3.0]
            )
            averaged = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(averaged["model"]["weight"], torch.tensor([7.5]))
            )

    def test_rejects_incompatible_model_signatures_and_optimizer_topology(self):
        module = load_average_module()
        base = checkpoint(
            {
                "weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "bias": torch.tensor([3.0], dtype=torch.float32),
            }
        )
        incompatible = {
            "keys": checkpoint(
                {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
            ),
            "shapes": checkpoint(
                {
                    "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                    "bias": torch.tensor([3.0], dtype=torch.float32),
                }
            ),
            "dtypes": checkpoint(
                {
                    "weight": torch.tensor([1.0, 2.0], dtype=torch.float64),
                    "bias": torch.tensor([3.0], dtype=torch.float32),
                }
            ),
        }
        optimizer_mismatch = copy.deepcopy(base)
        optimizer_mismatch["optimizer"]["param_groups"] = [
            {"params": [0]},
            {"params": [1]},
        ]
        incompatible["optimizer topology"] = optimizer_mismatch

        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            torch.save(base, first)
            for label, other_state in incompatible.items():
                with self.subTest(label=label):
                    second = root / f"{label.replace(' ', '-')}.pt"
                    output = root / f"{label.replace(' ', '-')}-average.pt"
                    torch.save(other_state, second)
                    with self.assertRaisesRegex(
                        ValueError, "model|keys|shapes|dtypes|optimizer.*topology"
                    ):
                        module.average_checkpoints([first, second], output)
                    self.assertFalse(output.exists())
                    self.assertFalse(
                        Path(str(output) + ".manifest.json").exists()
                    )

    def test_copies_identical_nonfloating_buffers_and_rejects_mismatches(self):
        module = load_average_module()
        first_state = checkpoint(
            {
                "weight": torch.tensor([1.0]),
                "counter": torch.tensor([7], dtype=torch.int64),
                "mask": torch.tensor([True, False]),
            }
        )
        second_state = copy.deepcopy(first_state)
        second_state["model"]["weight"] = torch.tensor([3.0])

        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            torch.save(first_state, first)
            torch.save(second_state, second)
            module.average_checkpoints([first, second], output)
            averaged = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(averaged["model"]["counter"], torch.tensor([7]))
            )
            self.assertTrue(
                torch.equal(
                    averaged["model"]["mask"], torch.tensor([True, False])
                )
            )

            mismatched = copy.deepcopy(second_state)
            mismatched["model"]["counter"] = torch.tensor([8], dtype=torch.int64)
            torch.save(mismatched, second)
            with self.assertRaisesRegex(ValueError, "nonfloating.*counter"):
                module.average_checkpoints(
                    [first, second], root / "mismatched-average.pt"
                )

    def test_uses_designated_template_metadata_and_removes_training_only_state(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            torch.save(
                checkpoint(
                    {"weight": torch.tensor([1.0])},
                    epoch=10,
                    metadata={"trajectory": "first"},
                ),
                first,
            )
            torch.save(
                checkpoint(
                    {"weight": torch.tensor([3.0])},
                    epoch=11,
                    metadata={"trajectory": "template"},
                ),
                second,
            )

            module.average_checkpoints(
                [first, second], output, template_path=second
            )
            averaged = torch.load(output, map_location="cpu", weights_only=True)

            self.assertEqual(averaged["epoch"], 11)
            self.assertEqual(averaged["metadata"], {"trajectory": "template"})
            self.assertNotIn("optimizer", averaged)
            self.assertNotIn("rng_state", averaged)
            self.assertTrue(averaged["inference_only"])
            self.assertEqual(
                averaged["checkpoint_type"], "inference_only_model_average"
            )
            self.assertEqual(
                averaged["training_state_removed"], ["optimizer", "rng_state"]
            )

    def test_writes_deterministic_manifest_with_hashes_weights_and_safe_loading(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([5.0])}), second)
            first_output = root / "average-one.pt"
            second_output = root / "average-two.pt"

            with patch.object(module.torch, "load", wraps=torch.load) as safe_load:
                module.average_checkpoints(
                    [first, second],
                    first_output,
                    weights=[1.0, 3.0],
                    template_path=second,
                )
            module.average_checkpoints(
                [first, second],
                second_output,
                weights=[1.0, 3.0],
                template_path=second,
            )

            first_manifest_path = Path(str(first_output) + ".manifest.json")
            second_manifest_path = Path(str(second_output) + ".manifest.json")
            manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
            second_manifest = json.loads(
                second_manifest_path.read_text(encoding="utf-8")
            )
            provenance_keys = set(manifest) - {"artifact", "committed"}
            self.assertEqual(
                {key: manifest[key] for key in provenance_keys},
                {key: second_manifest[key] for key in provenance_keys},
            )
            self.assertEqual(manifest["artifact"]["path"], first_output.name)
            self.assertEqual(second_manifest["artifact"]["path"], second_output.name)
            self.assertEqual(
                manifest["artifact"]["sha256"],
                second_manifest["artifact"]["sha256"],
            )
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(manifest["operation"], "same_basin_model_weight_average")
            self.assertEqual(manifest["template"], str(second.resolve()))
            self.assertEqual(
                manifest["sources"],
                [
                    {
                        "path": str(first.resolve()),
                        "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                        "weight": 0.25,
                    },
                    {
                        "path": str(second.resolve()),
                        "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                        "weight": 0.75,
                    },
                ],
            )
            self.assertGreaterEqual(safe_load.call_count, 2)
            self.assertTrue(
                all(call.kwargs.get("weights_only") is True for call in safe_load.call_args_list)
            )
            self.assertTrue(
                all(call.kwargs.get("map_location") == "cpu" for call in safe_load.call_args_list)
            )
            averaged = torch.load(first_output, map_location="cpu", weights_only=True)
            self.assertEqual(
                averaged["averaging_provenance"],
                {key: manifest[key] for key in provenance_keys},
            )

    def test_refuses_to_overwrite_output_or_manifest(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)

            occupied_output = root / "occupied.pt"
            occupied_output.write_bytes(b"existing-output")
            with self.assertRaises(FileExistsError):
                module.average_checkpoints([first, second], occupied_output)
            self.assertEqual(occupied_output.read_bytes(), b"existing-output")
            self.assertFalse(Path(str(occupied_output) + ".manifest.json").exists())

            output = root / "blocked-by-manifest.pt"
            occupied_manifest = Path(str(output) + ".manifest.json")
            occupied_manifest.write_bytes(b"existing-manifest")
            with self.assertRaises(FileExistsError):
                module.average_checkpoints([first, second], output)
            self.assertFalse(output.exists())
            self.assertEqual(occupied_manifest.read_bytes(), b"existing-manifest")

    def test_manifest_publication_failure_preserves_immutable_generation(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            manifest = root / "average.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)
            publish = module._publish_temporary_without_overwrite
            publications = 0

            def fail_manifest_publication(temporary, directory_fd, destination_name):
                nonlocal publications
                publications += 1
                if publications == 2:
                    raise OSError("injected manifest publication failure")
                return publish(temporary, directory_fd, destination_name)

            with patch.object(
                module,
                "_publish_temporary_without_overwrite",
                side_effect=fail_manifest_publication,
            ):
                with self.assertRaisesRegex(OSError, "injected manifest"):
                    module.average_checkpoints(
                        [first, second], output, manifest_path=manifest
                    )

            self.assertTrue(output.is_file())
            orphan = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(orphan["inference_only"])
            self.assertFalse(manifest.exists())

    def test_post_manifest_sync_failure_is_indeterminate_and_preserves_outputs(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            manifest = root / "average.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)
            real_fsync = module.os.fsync
            fsync_calls = 0

            def fail_final_publication_sync(fd):
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 4:
                    raise OSError("injected final directory sync failure")
                return real_fsync(fd)

            with patch.object(module.os, "fsync", side_effect=fail_final_publication_sync):
                with self.assertRaisesRegex(
                    module.PublicationIndeterminateError,
                    "manifest was published",
                ):
                    module.average_checkpoints(
                        [first, second], output, manifest_path=manifest
                    )

            self.assertTrue(output.is_file())
            self.assertTrue(manifest.is_file())

    def test_indeterminate_manifest_retry_must_sync_matching_publication_directory(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            manifest = root / "average.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([3.0])}), second)
            real_fsync = module.os.fsync
            directory_syncs = 0

            def fail_manifest_directory_sync(fd):
                nonlocal directory_syncs
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    directory_syncs += 1
                    if directory_syncs == 2:
                        raise OSError(errno.EIO, "injected manifest directory sync failure")
                return real_fsync(fd)

            with patch.object(
                module.os, "fsync", side_effect=fail_manifest_directory_sync
            ):
                with self.assertRaises(module.PublicationIndeterminateError):
                    module.average_checkpoints(
                        [first, second], output, manifest_path=manifest
                    )

            self.assertTrue(output.is_file())
            self.assertTrue(manifest.is_file())
            retry_directory_syncs = 0

            def fail_retry_directory_sync(fd):
                nonlocal retry_directory_syncs
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    retry_directory_syncs += 1
                    raise OSError(errno.EIO, "injected retry directory sync failure")
                return real_fsync(fd)

            with patch.object(module.os, "fsync", side_effect=fail_retry_directory_sync):
                with self.assertRaisesRegex(
                    module.PublicationIndeterminateError, "directory sync failed"
                ):
                    module.average_checkpoints(
                        [first, second], output, manifest_path=manifest
                    )

            self.assertEqual(retry_directory_syncs, 1)

    def test_named_temporary_replacement_before_close_is_never_deleted(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            generation = root / "average-generation.pt"
            manifest = root / "average-current.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([5.0])}), second)
            real_publish = module._publish_temporary_without_overwrite
            replacement_path = None

            def replace_temp_after_publish(temporary, directory_fd, destination_name):
                nonlocal replacement_path
                result = real_publish(temporary, directory_fd, destination_name)
                if replacement_path is None and isinstance(temporary.name, str):
                    replacement_path = Path(temporary.name)
                    displaced = root / ".displaced-publisher-temp"
                    if replacement_path.exists():
                        replacement_path.rename(displaced)
                    replacement_path.write_bytes(b"other-writer-temp")
                return result

            unavailable = OSError(errno.EOPNOTSUPP, "O_TMPFILE unavailable")
            with patch.object(
                module, "_open_anonymous_file", side_effect=unavailable
            ), patch.object(
                module,
                "_publish_temporary_without_overwrite",
                side_effect=replace_temp_after_publish,
            ):
                module.average_checkpoints(
                    [first, second], generation, manifest_path=manifest
                )

            self.assertIsNotNone(replacement_path)
            self.assertEqual(replacement_path.read_bytes(), b"other-writer-temp")

    def test_named_temporary_fallback_cleans_owned_temp_before_publication_attempt(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            manifest = root / "average.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([5.0])}), second)
            unavailable = OSError(errno.EOPNOTSUPP, "O_TMPFILE unavailable")
            real_fsync = module.os.fsync

            def fail_named_staging_sync(fd):
                fd_path = Path(os.readlink(f"/proc/self/fd/{fd}"))
                if fd_path.name.startswith(".average-checkpoint-"):
                    raise OSError(errno.EIO, "injected staging sync failure")
                return real_fsync(fd)

            with patch.object(
                module, "_open_anonymous_file", side_effect=unavailable
            ), patch.object(module.os, "fsync", side_effect=fail_named_staging_sync):
                with self.assertRaisesRegex(OSError, "staging sync failure"):
                    module.average_checkpoints(
                        [first, second], output, manifest_path=manifest
                    )

            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())
            self.assertEqual(list(root.glob(".average-checkpoint-*")), [])

    def test_named_temporary_fallback_idempotent_retry_has_no_temp_leak(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            manifest = root / "average.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([5.0])}), second)
            unavailable = OSError(errno.EOPNOTSUPP, "O_TMPFILE unavailable")

            with patch.object(
                module, "_open_anonymous_file", side_effect=unavailable
            ):
                module.average_checkpoints(
                    [first, second], output, manifest_path=manifest
                )
            output_bytes = output.read_bytes()
            manifest_bytes = manifest.read_bytes()

            with patch.object(
                module, "_open_anonymous_file", side_effect=unavailable
            ), patch.object(
                module,
                "_open_publication_temporary",
                wraps=module._open_publication_temporary,
            ) as open_publication_temporary:
                module.average_checkpoints(
                    [first, second], output, manifest_path=manifest
                )

            self.assertEqual(open_publication_temporary.call_count, 0)
            self.assertEqual(output.read_bytes(), output_bytes)
            self.assertEqual(manifest.read_bytes(), manifest_bytes)
            self.assertEqual(list(root.glob(".average-checkpoint-*")), [])

    def test_named_temporary_fallback_publishes_both_outputs(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            manifest = root / "average.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([5.0])}), second)

            unavailable = OSError(errno.EOPNOTSUPP, "O_TMPFILE unavailable")
            with patch.object(
                module, "_open_anonymous_file", side_effect=unavailable
            ):
                module.average_checkpoints(
                    [first, second], output, manifest_path=manifest
                )

            self.assertTrue(output.is_file())
            self.assertTrue(manifest.is_file())
            averaged = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(averaged["model"]["weight"], torch.tensor([3.0]))
            )
            self.assertEqual(list(root.glob(".average-checkpoint-*")), [])

    def test_named_temporary_fallback_never_overwrites_race_winner(self):
        module = load_average_module()
        with tempfile.TemporaryDirectory(prefix="average-checkpoints-") as tmp:
            root = Path(tmp)
            first = root / "first.pt"
            second = root / "second.pt"
            output = root / "average.pt"
            manifest = root / "average.json"
            torch.save(checkpoint({"weight": torch.tensor([1.0])}), first)
            torch.save(checkpoint({"weight": torch.tensor([5.0])}), second)
            real_publish = module._publish_temporary_without_overwrite
            collision_injected = False

            def inject_output_collision(temporary, directory_fd, destination_name):
                nonlocal collision_injected
                if destination_name == output.name and not collision_injected:
                    collision_injected = True
                    output.write_bytes(b"race-winner")
                return real_publish(temporary, directory_fd, destination_name)

            unavailable = OSError(errno.EOPNOTSUPP, "O_TMPFILE unavailable")
            with patch.object(
                module, "_open_anonymous_file", side_effect=unavailable
            ), patch.object(
                module,
                "_publish_temporary_without_overwrite",
                side_effect=inject_output_collision,
            ):
                with self.assertRaises(FileExistsError):
                    module.average_checkpoints(
                        [first, second], output, manifest_path=manifest
                    )

            self.assertEqual(output.read_bytes(), b"race-winner")
            self.assertFalse(manifest.exists())
            # Only the output temporary existed when publication collided.
            # It remains as one bounded uncertain orphan rather than risking
            # deletion of a non-cooperating writer's pathname replacement.
            self.assertEqual(len(list(root.glob(".average-checkpoint-*"))), 1)


if __name__ == "__main__":
    unittest.main()
