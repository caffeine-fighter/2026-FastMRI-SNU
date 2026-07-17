import copy
import io
import sys
import types
import unittest
from unittest.mock import patch

import torch

if "fcntl" not in sys.modules:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1,
        LOCK_UN=2,
        flock=lambda *_args, **_kwargs: None,
    )

from utils.learning.resume import (
    _load_checkpoint_from_handle,
    _load_requested_checkpoint,
    _manifest_artifact_sha256,
    _validate_retention_policy_transition,
    _validate_checkpoint_manifest_payload,
    build_training_state,
    load_training_state,
    recover_checkpoint_publication,
)
from utils.promptmr.contracts import checkpoint_model_contract


class _FakeScaler:
    def __init__(self, scale=1.0):
        self.scale = float(scale)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = float(state["scale"])


class PromptMRCheckpointContractTests(unittest.TestCase):
    def test_retained_ledger_rejects_disabling_retention_on_next_generation(self):
        previous = self._retained_manifest()

        with self.assertRaisesRegex(ValueError, "cannot disable retained epoch"):
            _validate_retention_policy_transition(previous, None)

        _validate_retention_policy_transition(None, None)
        _validate_retention_policy_transition(
            previous,
            {"epoch": 3, "generation": "c" * 32, "digest": "d" * 64},
        )

    def test_baseline_v1_manifest_remains_readable_without_artifact_digests(self):
        generation = "a" * 32
        model_name = f".checkpoint-generation-{generation}-model.pt"
        manifest = {
            "format_version": 1,
            "generation": generation,
            "model": model_name,
            "best": None,
            "history": f".checkpoint-generation-{generation}-history.npy",
        }

        validated = _validate_checkpoint_manifest_payload(
            manifest, "checkpoint_manifest.json"
        )

        self.assertIs(validated, manifest)
        self.assertIsNone(_manifest_artifact_sha256(validated, "model"))
        legacy_state = {"legacy": True}
        with patch(
            "utils.learning.resume._open_checkpoint_directory", return_value=7
        ), patch(
            "utils.learning.resume._read_checkpoint_manifest", return_value=validated
        ), patch(
            "utils.learning.resume._open_manifest_artifact",
            return_value=io.BytesIO(b"legacy"),
        ), patch(
            "utils.learning.resume._load_checkpoint_from_handle",
            return_value=legacy_state,
        ) as load_checkpoint, patch("utils.learning.resume.os.close"):
            loaded = _load_requested_checkpoint("model.pt")

        self.assertIs(loaded, legacy_state)
        self.assertIsNone(load_checkpoint.call_args.args[1])

    def test_manifest_expected_digest_mismatch_preserves_error_contract(self):
        model_name = ".checkpoint-generation-" + "a" * 32 + "-model.pt"
        manifest = {
            "model": model_name,
            "artifacts": {model_name: "b" * 64},
        }
        with patch(
            "utils.learning.resume._open_checkpoint_directory", return_value=7
        ), patch(
            "utils.learning.resume._read_checkpoint_manifest", return_value=manifest
        ), patch("utils.learning.resume.os.close"):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _load_requested_checkpoint("model.pt", "c" * 64)

    def test_authoritative_recovery_hash_verifies_model_before_alias(self):
        generation = "a" * 32
        model_name = f".checkpoint-generation-{generation}-model.pt"
        expected_hash = "b" * 64
        manifest = {
            "format_version": 2,
            "generation": generation,
            "model": model_name,
            "best": None,
            "artifacts": {model_name: expected_hash},
        }

        def load_only_if_unverified(_handle, expected_sha256=None):
            if expected_sha256 is not None:
                raise RuntimeError("Checkpoint SHA-256 mismatch")
            return {"epoch": 1}

        with patch(
            "utils.learning.resume._open_checkpoint_directory", return_value=123
        ), patch(
            "utils.learning.resume._read_checkpoint_manifest",
            return_value=manifest,
        ), patch(
            "utils.learning.resume._open_manifest_artifact",
            return_value=io.BytesIO(b"tampered"),
        ), patch(
            "utils.learning.resume._load_checkpoint_from_handle",
            side_effect=load_only_if_unverified,
        ), patch(
            "utils.learning.resume.validate_training_checkpoint"
        ), patch(
            "utils.learning.resume._replace_alias_from_artifact"
        ), patch("utils.learning.resume.os.fsync"), patch(
            "utils.learning.resume.os.close"
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                recover_checkpoint_publication("unused")

    def _retained_manifest(self):
        first = "a" * 32
        second = "b" * 32
        return {
            "format_version": 2,
            "generation": second,
            "model": f".checkpoint-generation-{second}-model.pt",
            "best": f".checkpoint-generation-{first}-model.pt",
            "history": f".checkpoint-generation-{second}-history.npy",
            "artifacts": {
                f".checkpoint-generation-{second}-model.pt": "e" * 64,
                f".checkpoint-generation-{first}-model.pt": "f" * 64,
                f".checkpoint-generation-{second}-history.npy": "0" * 64,
            },
            "retained_epochs": [
                {"epoch": 1, "generation": first, "digest": "c" * 64},
                {"epoch": 2, "generation": second, "digest": "d" * 64},
            ],
        }

    def test_multi_epoch_retained_manifest_is_accepted(self):
        manifest = self._retained_manifest()
        self.assertIs(
            _validate_checkpoint_manifest_payload(manifest, "manifest.json"),
            manifest,
        )

    def test_malformed_retained_manifest_is_rejected(self):
        cases = []
        duplicate_generation = self._retained_manifest()
        duplicate_generation["retained_epochs"][1]["generation"] = "a" * 32
        cases.append(duplicate_generation)
        non_increasing = self._retained_manifest()
        non_increasing["retained_epochs"][1]["epoch"] = 1
        cases.append(non_increasing)
        bad_digest = self._retained_manifest()
        bad_digest["retained_epochs"][1]["digest"] = "not-a-digest"
        cases.append(bad_digest)
        tail_mismatch = self._retained_manifest()
        tail_mismatch["retained_epochs"][-1]["generation"] = "e" * 32
        cases.append(tail_mismatch)
        boolean_version = self._retained_manifest()
        boolean_version["format_version"] = True
        cases.append(boolean_version)
        for manifest in cases:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                _validate_checkpoint_manifest_payload(manifest, "manifest.json")

    def _build_objects(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=35, gamma=0.1
        )
        scaler = _FakeScaler(128.0)
        return model, optimizer, scheduler, scaler

    def test_promptmr_scheduler_scaler_and_contract_resume_together(self):
        source = self._build_objects()
        source[2].gamma = 0.2
        contract = checkpoint_model_contract("promptmr_plus")
        with patch("utils.learning.resume.torch.cuda.is_available", return_value=False):
            checkpoint = build_training_state(
                3,
                source[0],
                source[1],
                0.2,
                scheduler=source[2],
                scaler=source[3],
                model_contract=contract,
            )
        target = self._build_objects()
        target[3].scale = 2.0
        with patch(
            "utils.learning.resume._load_requested_checkpoint",
            return_value=checkpoint,
        ), patch("utils.learning.resume.torch.cuda.is_available", return_value=False):
                epoch, best = load_training_state(
                    "unused.pt",
                    target[0],
                    target[1],
                    torch.device("cpu"),
                    scheduler=target[2],
                    scaler=target[3],
                    expected_model_contract=contract,
                )
        self.assertEqual(epoch, 3)
        self.assertEqual(best, 0.2)
        self.assertEqual(target[2].state_dict(), source[2].state_dict())
        self.assertEqual(target[3].state_dict(), source[3].state_dict())

    def test_incompatible_family_is_rejected_without_live_mutation(self):
        source = self._build_objects()
        with patch("utils.learning.resume.torch.cuda.is_available", return_value=False):
            checkpoint = build_training_state(
                1,
                source[0],
                source[1],
                0.3,
                scheduler=source[2],
                scaler=source[3],
                model_contract=checkpoint_model_contract("promptmr_plus"),
            )
        target = self._build_objects()
        before = (
            copy.deepcopy(target[0].state_dict()),
            copy.deepcopy(target[1].state_dict()),
            copy.deepcopy(target[2].state_dict()),
            copy.deepcopy(target[3].state_dict()),
        )
        with patch(
            "utils.learning.resume._load_requested_checkpoint",
            return_value=checkpoint,
        ):
            with self.assertRaisesRegex(ValueError, "model family"):
                load_training_state(
                    "unused.pt",
                    target[0],
                    target[1],
                    torch.device("cpu"),
                    scheduler=target[2],
                    scaler=target[3],
                    expected_model_contract={"model_family": "varnet"},
                )
        self.assertEqual(target[0].state_dict().keys(), before[0].keys())
        self.assertEqual(target[1].state_dict(), before[1])
        self.assertEqual(target[2].state_dict(), before[2])
        self.assertEqual(target[3].state_dict(), before[3])

    def test_legacy_varnet_checkpoint_remains_accepted(self):
        source = self._build_objects()
        with patch("utils.learning.resume.torch.cuda.is_available", return_value=False):
            checkpoint = build_training_state(1, source[0], source[1], 0.4)
        target = self._build_objects()
        with patch(
            "utils.learning.resume._load_requested_checkpoint",
            return_value=checkpoint,
        ):
            load_training_state(
                "unused.pt",
                target[0],
                target[1],
                torch.device("cpu"),
                expected_model_contract={"model_family": "varnet"},
            )


if __name__ == "__main__":
    unittest.main()
