import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

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


class PromptMRRetainedReconciliationTests(unittest.TestCase):
    def _prepared_publication(self, generation="a" * 32, epoch=1):
        model_name = f".checkpoint-generation-{generation}-model.pt"
        history_name = f".checkpoint-generation-{generation}-history.npy"
        manifest = {
            "format_version": 2,
            "generation": generation,
            "model": model_name,
            "best": model_name,
            "history": history_name,
            "artifacts": {
                model_name: hashlib.sha256(b"model").hexdigest(),
                history_name: hashlib.sha256(b"history").hexdigest(),
            },
            "retained_epochs": [{
                "epoch": epoch,
                "generation": generation,
                "digest": "b" * 64,
            }],
        }
        payloads = {model_name: b"model", history_name: b"history"}
        publication = {
            "format_version": 1,
            "generation": generation,
            "epoch": epoch,
            "parent_generation": None,
            "parent_manifest_sha256": hashlib.sha256(b"").hexdigest(),
            "manifest": manifest,
            "artifacts": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
            },
            "retained": {
                "name": f"epoch_{epoch:04d}", "digest": "b" * 64
            },
        }
        return publication

    def _run_reconcile(
        self,
        publications,
        corrupt_artifact=False,
        records=None,
        partial=None,
        manifest=None,
        checkpoint_epoch=1,
        history_length=1,
    ):
        publication_by_name = {
            f".checkpoint-generation-{item['generation']}-publication.json": item
            for item in publications
        }
        payloads = {}
        for item in publications:
            for name in item["artifacts"]:
                payloads[name] = b"corrupt" if corrupt_artifact else (
                    b"model" if name.endswith("-model.pt") else b"history"
                )

        class Scan:
            def __enter__(self):
                return iter(
                    types.SimpleNamespace(name=name)
                    for name in publication_by_name
                )

            def __exit__(self, *_args):
                return False

        @contextmanager
        def open_regular(_fd, name, _description):
            if name in publication_by_name:
                yield io.StringIO(json.dumps(publication_by_name[name]))
            else:
                yield io.BytesIO(payloads[name])

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            train_part, "_collect_retained_epoch_records",
            return_value=(
                [{"epoch": 1, "digest": "b" * 64}]
                if records is None else records,
                [] if partial is None else partial,
            ),
        ), patch.object(
            train_part, "_open_checkpoint_directory", return_value=123
        ), patch.object(
            train_part, "_read_checkpoint_manifest", return_value=manifest
        ), patch.object(
            train_part, "_open_regular_at", side_effect=open_regular
        ), patch.object(
            train_part.os, "scandir", return_value=Scan()
        ), patch.object(train_part.os, "fsync"), patch.object(
            train_part.os, "close"
        ), patch.object(
            train_part, "_load_checkpoint_from_handle",
            return_value={"epoch": checkpoint_epoch}
        ), patch.object(
            train_part, "validate_training_checkpoint"
        ), patch.object(
            train_part.np,
            "load",
            return_value=train_part.np.column_stack((
                train_part.np.arange(history_length),
                train_part.np.zeros(history_length),
            )),
        ), patch.object(
            train_part, "validate_checkpoint_pair"
        ), patch.object(
            train_part, "_publish_checkpoint_manifest"
        ) as publish_manifest, patch.object(
            train_part, "_publish_history_alias"
        ), patch.object(train_part, "_publish_stable_alias"):
            result = train_part.reconcile_retained_checkpoint_state(
                Path(tmp), Path(tmp) / "retained", Path(tmp) / "history.npy"
            )
        return result, publish_manifest

    def test_restart_adopts_exact_retained_before_manifest_generation(self):
        publication = self._prepared_publication()
        result, publish_manifest = self._run_reconcile([publication])
        self.assertTrue(result)
        publish_manifest.assert_called_once()
        self.assertEqual(
            publish_manifest.call_args.args[2], publication["manifest"]
        )

    def test_restart_adopts_first_generation_after_external_exact_resume(self):
        publication = self._prepared_publication(epoch=4)
        result, publish_manifest = self._run_reconcile(
            [publication],
            records=[{"epoch": 4, "digest": "b" * 64}],
            checkpoint_epoch=4,
            history_length=4,
        )
        self.assertTrue(result)
        publish_manifest.assert_called_once()

    def test_restart_duplicate_prepared_generations_fail_closed(self):
        first = self._prepared_publication("a" * 32)
        second = self._prepared_publication("c" * 32)
        with self.assertRaisesRegex(RuntimeError, "exactly one provenance"):
            self._run_reconcile([first, second])

    def test_restart_artifact_digest_mismatch_fails_closed(self):
        publication = self._prepared_publication()
        with self.assertRaisesRegex(RuntimeError, "artifact digest mismatch"):
            self._run_reconcile([publication], corrupt_artifact=True)

    def test_boolean_publication_version_and_epoch_fail_closed(self):
        for field in ("format_version", "epoch"):
            publication = self._prepared_publication()
            publication[field] = True
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self._run_reconcile([publication])

    def test_reconciled_manifest_recovers_aliases_before_model_construction(self):
        class StopBeforeModelConstruction(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix="promptmr-reconcile-train-") as tmp:
            root = Path(tmp)
            args = types.SimpleNamespace(
                model_family="promptmr_plus",
                retain_val_epochs=True,
                exp_dir=root / "checkpoints",
                val_epochs_dir=root / "val_epochs",
                val_loss_dir=root,
                resume_checkpoint=None,
                resume_checkpoint_sha256=None,
                require_cuda_device_name=None,
                GPU_NUM=0,
            )
            with patch.object(
                train_part, "reconcile_retained_checkpoint_state", return_value=True
            ), patch.object(
                train_part, "recover_checkpoint_publication", return_value=True
            ) as recover, patch.object(
                train_part.torch.cuda, "is_available", return_value=False
            ), patch(
                "utils.promptmr.runtime.build_promptmr_plus_model",
                side_effect=StopBeforeModelConstruction,
            ):
                with self.assertRaises(StopBeforeModelConstruction):
                    train_part.train(args)

        recover.assert_called_once_with(args.exp_dir)

    def test_restart_ignores_hidden_unpublished_staging_tree(self):
        result, publish_manifest = self._run_reconcile(
            [],
            records=[],
            partial=[".epoch_0001-unpublished-orphan-deadbeef"],
        )
        self.assertFalse(result)
        publish_manifest.assert_not_called()

    def test_restart_resumes_exact_authoritative_manifest(self):
        publication = self._prepared_publication()
        manifest = publication["manifest"]
        result, publish_manifest = self._run_reconcile(
            [],
            records=[{"epoch": 1, "digest": "b" * 64}],
            manifest=manifest,
        )
        self.assertTrue(result)
        publish_manifest.assert_not_called()

    def test_restart_unknown_retained_entry_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Ambiguous partial"):
            self._run_reconcile([], records=[], partial=["unexpected.txt"])


if __name__ == "__main__":
    unittest.main()
