import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_checkpoint.py"


def load_materialize_module():
    spec = importlib.util.spec_from_file_location("materialize_checkpoint_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inference_checkpoint(model=None):
    if model is None:
        model = torch.nn.Linear(2, 1)
    return {
        "model": model.state_dict(),
        "checkpoint_type": "inference_only_test",
        "inference_only": True,
    }


def training_checkpoint(model=None):
    if model is None:
        model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return {
        "format_version": 1,
        "epoch": 3,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": 0.25,
        "rng_state": None,
    }


class UnsafePayload:
    pass


class MaterializeCheckpointTests(unittest.TestCase):
    def test_materializes_exact_copy_with_source_hash_provenance(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source_dir = root / "source" / "checkpoints"
            source_dir.mkdir(parents=True)
            source = source_dir / ".checkpoint-generation-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-model.pt"
            source_best = source_dir / "best_model.pt"
            torch.save(inference_checkpoint(), source)
            source_best.write_bytes(b"existing-leader-best")
            source_bytes = source.read_bytes()
            source_best_bytes = source_best.read_bytes()
            candidate = root / "candidate"

            best = module.materialize_checkpoint(source, candidate)

            provenance_path = candidate / "checkpoints" / "materialization.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            expected_hash = hashlib.sha256(source_bytes).hexdigest()
            self.assertEqual(best, candidate / "checkpoints" / "best_model.pt")
            self.assertEqual(best.read_bytes(), source_bytes)
            self.assertNotEqual(best.stat().st_ino, source.stat().st_ino)
            self.assertEqual(
                provenance,
                {
                    "artifact": {"path": "best_model.pt", "sha256": expected_hash},
                    "format_version": 1,
                    "operation": "materialize_immutable_checkpoint",
                    "source": {
                        "path": str(source.absolute()),
                        "sha256": expected_hash,
                    },
                },
            )
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(source_best.read_bytes(), source_best_bytes)

    def test_existing_test_part_loader_consumes_materialized_best(self):
        module = load_materialize_module()
        from utils.learning import test_part

        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            expected_model = torch.nn.Linear(2, 1)
            with torch.no_grad():
                expected_model.weight.fill_(3.0)
                expected_model.bias.fill_(-2.0)
            torch.save(inference_checkpoint(expected_model), source)
            candidate = root / "candidate"
            module.materialize_checkpoint(source, candidate)
            args = SimpleNamespace(
                cascade=1,
                chans=2,
                sens_chans=1,
                exp_dir=candidate / "checkpoints",
            )

            with patch.object(test_part, "VarNet", return_value=torch.nn.Linear(2, 1)):
                loaded = test_part.load_model(args, torch.device("cpu"))

            self.assertFalse(loaded.training)
            self.assertTrue(torch.equal(loaded.weight, expected_model.weight))
            self.assertTrue(torch.equal(loaded.bias, expected_model.bias))

    def test_expected_sha256_mismatch_creates_no_candidate(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            candidate = root / "candidate"

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                module.materialize_checkpoint(source, candidate, "0" * 64)

            self.assertFalse(candidate.exists())

    def test_rejects_source_symlink_and_lexical_traversal(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            source = source_dir / "selected.pt"
            torch.save(inference_checkpoint(), source)
            source_link = root / "selected-link.pt"
            source_link.symlink_to(source)

            with self.assertRaisesRegex(ValueError, "securely open"):
                module.materialize_checkpoint(source_link, root / "candidate-link")
            traversal_source = source_dir / ".." / "source" / "selected.pt"
            with self.assertRaisesRegex(ValueError, "without traversal"):
                module.materialize_checkpoint(
                    traversal_source, root / "candidate-traversal"
                )

            self.assertFalse((root / "candidate-link").exists())
            self.assertFalse((root / "candidate-traversal").exists())

    def test_rejects_non_regular_source(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source_directory = root / "selected.pt"
            source_directory.mkdir()
            candidate = root / "candidate"

            with self.assertRaisesRegex(ValueError, "not a regular file"):
                module.materialize_checkpoint(source_directory, candidate)

            self.assertFalse(candidate.exists())

    def test_rejects_symlinked_candidate_parent(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "contain no symlinks"):
                module.materialize_checkpoint(
                    source, linked_parent / "candidate"
                )

            self.assertEqual(list(real_parent.iterdir()), [])

    def test_rejects_candidate_lexical_traversal(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            candidate = root / "unused" / ".." / "candidate"

            with self.assertRaisesRegex(ValueError, "without traversal"):
                module.materialize_checkpoint(source, candidate)

            self.assertFalse((root / "candidate").exists())

    def test_rejects_nonempty_candidate_without_mutation(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            source_bytes = source.read_bytes()
            candidate = root / "candidate"
            candidate.mkdir()
            occupied = candidate / "keep.txt"
            occupied.write_bytes(b"keep")

            with self.assertRaisesRegex(ValueError, "completely empty"):
                module.materialize_checkpoint(source, candidate)

            self.assertEqual(occupied.read_bytes(), b"keep")
            self.assertEqual(source.read_bytes(), source_bytes)

    def test_checkpoint_directory_collision_preserves_race_winner(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            source_bytes = source.read_bytes()
            candidate = root / "candidate"
            real_publish = module._publish_staged_directory_no_replace

            def collide(staged, final_dir, description):
                final_dir.mkdir()
                (final_dir / "best_model.pt").write_bytes(b"race-winner")
                return real_publish(staged, final_dir, description)

            with patch.object(
                module,
                "_publish_staged_directory_no_replace",
                side_effect=collide,
            ):
                with self.assertRaises(FileExistsError):
                    module.materialize_checkpoint(source, candidate)

            self.assertEqual(
                (candidate / "checkpoints" / "best_model.pt").read_bytes(),
                b"race-winner",
            )
            self.assertEqual(source.read_bytes(), source_bytes)
            entries = sorted(path.name for path in candidate.iterdir())
            self.assertIn("checkpoints", entries)
            self.assertEqual(
                len(
                    [
                        name
                        for name in entries
                        if name.startswith(".checkpoints-unpublished-orphan-")
                    ]
                ),
                1,
            )

    def test_interruption_before_atomic_directory_publish_exposes_no_best(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            source_bytes = source.read_bytes()
            candidate = root / "candidate"

            with patch.object(
                module,
                "_publish_staged_directory_no_replace",
                side_effect=KeyboardInterrupt("injected interruption"),
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "injected interruption"):
                    module.materialize_checkpoint(source, candidate)

            self.assertTrue(candidate.is_dir())
            self.assertFalse((candidate / "checkpoints").exists())
            orphans = [
                path
                for path in candidate.iterdir()
                if path.name.startswith(".checkpoints-unpublished-orphan-")
            ]
            self.assertEqual(len(orphans), 1)
            self.assertTrue((orphans[0] / "best_model.pt").is_file())
            self.assertEqual(source.read_bytes(), source_bytes)

    def test_existing_empty_candidate_is_supported(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            candidate = root / "candidate"
            candidate.mkdir()

            best = module.materialize_checkpoint(source, candidate)

            self.assertTrue(best.is_file())
            self.assertTrue((candidate / "checkpoints" / "materialization.json").is_file())

    def test_accepts_regular_weights_only_training_checkpoint(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected-training.pt"
            torch.save(training_checkpoint(), source)

            best = module.materialize_checkpoint(source, root / "candidate")

            loaded = torch.load(best, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["epoch"], 3)
            self.assertIn("optimizer", loaded)

    def test_rejects_checkpoint_that_is_not_weights_only_compatible(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "unsafe.pt"
            unsafe = inference_checkpoint()
            unsafe["unsafe"] = UnsafePayload()
            torch.save(unsafe, source)
            candidate = root / "candidate"

            with self.assertRaises(Exception):
                module.materialize_checkpoint(source, candidate)

            self.assertFalse(candidate.exists())

    def test_interruption_after_atomic_publish_preserves_complete_candidate(self):
        module = load_materialize_module()
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            source_bytes = source.read_bytes()
            candidate = root / "candidate"
            real_publish = module._publish_staged_directory_no_replace

            def publish_then_interrupt(staged, final_dir, description):
                real_publish(staged, final_dir, description)
                raise KeyboardInterrupt("injected after publication")

            with patch.object(
                module,
                "_publish_staged_directory_no_replace",
                side_effect=publish_then_interrupt,
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "after publication"):
                    module.materialize_checkpoint(source, candidate)

            checkpoints = candidate / "checkpoints"
            self.assertEqual(
                (checkpoints / "best_model.pt").read_bytes(), source_bytes
            )
            self.assertTrue((checkpoints / "materialization.json").is_file())
            self.assertEqual(source.read_bytes(), source_bytes)

    def test_cli_materializes_with_expected_hash_and_documents_contract(self):
        with tempfile.TemporaryDirectory(prefix="materialize-checkpoint-cli-") as tmp:
            root = Path(tmp)
            source = root / "selected.pt"
            torch.save(inference_checkpoint(), source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            candidate = root / "candidate"

            help_result = subprocess.run(
                [sys.executable, str(SCRIPT), "--help"],
                cwd=SCRIPT.parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    "--candidate-exp-dir",
                    str(candidate),
                    "--expected-sha256",
                    digest,
                ],
                cwd=SCRIPT.parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            normalized_help = " ".join(help_result.stdout.split())
            self.assertIn("--candidate-exp-dir", normalized_help)
            self.assertIn("--expected-sha256", normalized_help)
            self.assertIn("new or empty", normalized_help)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout.strip(),
                str(candidate / "checkpoints" / "best_model.pt"),
            )


if __name__ == "__main__":
    unittest.main()
