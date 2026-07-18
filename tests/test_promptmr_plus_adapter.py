import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.model.promptmr_plus_adapter import (
    PROMPTMR_PLUS_ROOT,
    PromptMRPlusAdapter,
    acceleration_from_filename,
    adjacent_slice_indices,
    build_promptmr_plus,
    load_promptmr_plus_checkpoint,
    prepare_promptmr_input,
    verify_promptmr_plus_source,
)


class RecordingPromptMR(torch.nn.Module):
    def __init__(self, finite=True):
        super().__init__()
        self.finite = finite
        self.calls = []

    def forward(
        self,
        kspace,
        mask,
        num_low_frequencies,
        mask_type,
        use_checkpoint=False,
        compute_sens_per_coil=False,
    ):
        self.calls.append(
            (kspace.shape, mask.dtype, mask_type, use_checkpoint, compute_sens_per_coil)
        )
        output = torch.ones(
            kspace.shape[0], kspace.shape[-3], kspace.shape[-2], dtype=torch.float32
        )
        if not self.finite:
            output[0, 0, 0] = float("nan")
        return {"img_pred": output}


class PromptMRPlusAdapterTests(unittest.TestCase):
    def test_routes_authoritative_acceleration_tokens(self):
        self.assertEqual(acceleration_from_filename("scan_acc4_case.h5"), 4)
        self.assertEqual(acceleration_from_filename("scan_acc8_case.h5"), 8)

    def test_unknown_or_ambiguous_acceleration_fails_closed(self):
        for filename in ("scan_case.h5", "scan_acc4_acc8_case.h5"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "exactly one acc4/acc8"):
                    acceleration_from_filename(filename)

    def test_adjacent_slice_indices_replicate_volume_edges(self):
        self.assertEqual(adjacent_slice_indices(0, 4), (0, 0, 0, 1, 2))
        self.assertEqual(adjacent_slice_indices(3, 4), (1, 2, 3, 3, 3))

    def test_prepares_promptmr_kspace_and_bool_mask_schema(self):
        volume = np.ones((4, 3, 8, 10), dtype=np.complex64)
        mask = np.ones(10, dtype=np.uint8)

        prepared = prepare_promptmr_input(
            volume, mask, slice_index=0, filename="scan_acc4_case.h5"
        )

        self.assertEqual(prepared.kspace.shape, (1, 15, 8, 10, 2))
        self.assertEqual(prepared.kspace.dtype, torch.float32)
        self.assertEqual(prepared.mask.shape, (1, 1, 1, 10, 1))
        self.assertEqual(prepared.mask.dtype, torch.bool)
        self.assertEqual(prepared.num_low_frequencies.shape, (1,))
        self.assertEqual(prepared.num_low_frequencies.dtype, torch.int64)
        self.assertEqual(prepared.num_low_frequencies.item(), -1)
        self.assertEqual(prepared.acceleration, 4)

    def test_wrapper_preserves_output_crop_and_one_model_routes_both_accelerations(self):
        core = RecordingPromptMR()
        adapter = PromptMRPlusAdapter(core)
        volume = np.ones((5, 2, 8, 10), dtype=np.complex64)
        mask = np.ones(10, dtype=np.uint8)

        outputs = []
        for acceleration in (4, 8):
            prepared = prepare_promptmr_input(
                volume,
                mask,
                slice_index=2,
                filename=f"scan_acc{acceleration}_case.h5",
            )
            outputs.append(adapter(prepared, crop_size=(6, 8)))

        self.assertEqual([output.shape for output in outputs], [(1, 6, 8), (1, 6, 8)])
        self.assertTrue(all(output.dtype == torch.float32 for output in outputs))
        self.assertIs(adapter.core, core)
        self.assertEqual(len(core.calls), 2)
        self.assertTrue(all(call[1] == torch.bool for call in core.calls))
        self.assertTrue(all(call[2] == ("cartesian",) for call in core.calls))

    def test_nonfinite_output_fails_closed(self):
        adapter = PromptMRPlusAdapter(RecordingPromptMR(finite=False))
        prepared = prepare_promptmr_input(
            np.ones((5, 2, 8, 10), dtype=np.complex64),
            np.ones(10, dtype=np.uint8),
            slice_index=2,
            filename="scan_acc4_case.h5",
        )
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            adapter(prepared)

    def test_checkpoint_namespace_is_stripped_and_loaded_strictly(self):
        model = torch.nn.Linear(2, 2, bias=False)
        checkpoint = {"state_dict": {"promptmr.weight": torch.ones(2, 2)}}
        load_promptmr_plus_checkpoint(model, checkpoint)
        self.assertTrue(torch.equal(model.weight, torch.ones(2, 2)))

        with self.assertRaisesRegex(RuntimeError, "strict PromptMR\+ checkpoint"):
            load_promptmr_plus_checkpoint(
                model, {"state_dict": {"wrong.weight": torch.ones(2, 2)}}
            )

    def test_rejects_preloaded_transitive_promptmr_modules(self):
        repository = Path(__file__).resolve().parents[1]
        code = textwrap.dedent(
            """
            import sys
            import types

            fake_mri = types.ModuleType("mri_utils")
            fake_data = types.ModuleType("data")
            fake_transforms = types.ModuleType("data.transforms")
            fake_data.transforms = fake_transforms
            sys.modules["mri_utils"] = fake_mri
            sys.modules["data"] = fake_data
            sys.modules["data.transforms"] = fake_transforms

            from utils.model.promptmr_plus_adapter import import_promptmr_plus_module
            try:
                import_promptmr_plus_module("models.promptmr_v2")
            except RuntimeError as exc:
                if "conflicting controlled module" in str(exc):
                    raise SystemExit(0)
                raise
            raise SystemExit("unverified transitive module was accepted")
            """
        )
        environment = dict(os.environ, PYTHONPATH=str(repository))
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_builds_exact_upstream_promptmr_plus_config(self):
        model = build_promptmr_plus()
        self.assertEqual(model.num_cascades, 12)
        self.assertEqual(model.num_adj_slices, 5)
        self.assertEqual(model.n_history, 11)
        self.assertFalse(hasattr(model, "acceleration"))
        self.assertTrue(
            all(cascade.n_buffer == cascade.model.n_buffer for cascade in model.cascades)
        )
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_actual_upstream_cpu_forward_is_real_fp32_and_finite(self):
        rng = np.random.default_rng(42)
        volume = (
            rng.standard_normal((5, 2, 64, 64))
            + 1j * rng.standard_normal((5, 2, 64, 64))
        ).astype(np.complex64)
        mask = np.zeros(64, dtype=np.uint8)
        mask[24:40] = 1
        mask[::4] = 1
        prepared = prepare_promptmr_input(
            volume,
            mask,
            slice_index=2,
            filename="cpu_acc4_smoke.h5",
        )
        adapter = PromptMRPlusAdapter(build_promptmr_plus()).eval()
        with torch.inference_mode():
            output = adapter(prepared, crop_size=(64, 64))
        self.assertEqual(output.shape, (1, 64, 64))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())

    def test_source_manifest_preserves_pin_license_and_packaging_allowlist(self):
        manifest = verify_promptmr_plus_source()
        self.assertEqual(
            manifest["commit"], "934eeda6d4d18cd39e406fa1eee9e1f70603cb5e"
        )
        self.assertEqual(manifest["license"]["spdx_like"], "RU-NCRL")
        self.assertEqual(
            manifest["license"]["status"],
            "NONCOMMERCIAL_COMPETITION_USE_ALLOWED",
        )
        self.assertIn("Non-commercial Research License", (PROMPTMR_PLUS_ROOT / "LICENSE.md").read_text())
        for relative_path in manifest["files"]:
            self.assertFalse(relative_path.endswith((".ckpt", ".pt", ".h5")))
            self.assertTrue((PROMPTMR_PLUS_ROOT / relative_path).is_file())

    def test_source_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "promptmr_plus"
            shutil.copytree(PROMPTMR_PLUS_ROOT, copied)
            target = copied / "upstream/models/utils.py"
            target.write_text(target.read_text() + "\n# tampered\n")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                verify_promptmr_plus_source(copied)

    def test_source_manifest_hash_is_independently_anchored(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "promptmr_plus"
            shutil.copytree(PROMPTMR_PLUS_ROOT, copied)
            target = copied / "SOURCE_MANIFEST.json"
            manifest = json.loads(target.read_text())
            manifest["repository"] = "https://attacker.invalid/replacement"
            target.write_text(json.dumps(manifest, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest checksum mismatch"):
                verify_promptmr_plus_source(copied)

    def test_unlisted_executable_bytecode_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "promptmr_plus"
            shutil.copytree(PROMPTMR_PLUS_ROOT, copied)
            cache = copied / "upstream/models/__pycache__"
            cache.mkdir(exist_ok=True)
            (cache / "promptmr_v2.cpython-310.pyc").write_bytes(b"untrusted")
            with self.assertRaisesRegex(ValueError, "unlisted executable"):
                verify_promptmr_plus_source(copied)

    def test_unexpected_vendored_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "promptmr_plus"
            shutil.copytree(PROMPTMR_PLUS_ROOT, copied)
            (copied / "upstream/models/torch.py").write_text("raise RuntimeError()\n")
            with self.assertRaisesRegex(ValueError, "unexpected vendored files"):
                verify_promptmr_plus_source(copied)

    def test_recon_eval_remains_unchanged(self):
        digest = hashlib.sha256(Path("recon_eval.py").read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "a93cf978b4938a060d4f5a204d3f7118fb8c17bf12408cbe44e6e7954ba5a135",
        )


if __name__ == "__main__":
    unittest.main()
