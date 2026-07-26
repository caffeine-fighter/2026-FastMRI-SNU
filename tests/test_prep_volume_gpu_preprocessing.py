import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_UTILS = str(ROOT / "utils" / "model")
if MODEL_UTILS not in sys.path:
    sys.path.insert(0, MODEL_UTILS)

from utils.learning.test_part import prep_volume, recon_slice


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kspace = None
        self.last_mask = None

    def forward(self, masked_kspace, mask):
        self.last_kspace = masked_kspace
        self.last_mask = mask
        batch, _, height, width, _ = masked_kspace.shape
        return torch.zeros((batch, height, width), device=masked_kspace.device)


class PrepVolumeGpuPreprocessingTests(unittest.TestCase):
    def _write_volume(self, root, kspace, mask):
        path = root / "volume.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("kspace", data=kspace)
            handle.create_dataset("mask", data=mask)
        return path

    def test_prep_volume_casts_transfers_and_masks_before_recon_slice(self):
        rng = np.random.default_rng(430)
        kspace = (
            rng.standard_normal((3, 2, 8, 6))
            + 1j * rng.standard_normal((3, 2, 8, 6))
        ).astype(np.complex64)
        mask = np.array([0, 1, 1, 0, 1, 0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_volume(Path(temporary), kspace, mask)
            context = prep_volume(None, path, torch.device("cpu"))

        self.assertEqual(
            set(context),
            {"masked_kspace", "mask", "device", "num_slices"},
        )
        self.assertEqual(context["masked_kspace"].dtype, torch.float32)
        self.assertEqual(tuple(context["masked_kspace"].shape), (3, 2, 8, 6, 2))
        self.assertEqual(context["mask"].dtype, torch.bool)
        self.assertEqual(tuple(context["mask"].shape), (1, 1, 1, 6, 1))
        self.assertEqual(context["num_slices"], 3)

        expected = np.stack((kspace.real, kspace.imag), axis=-1)
        expected *= mask.reshape(1, 1, 1, 6, 1)
        np.testing.assert_array_equal(context["masked_kspace"].numpy(), expected)

        model = _RecordingModel()
        output = recon_slice(model, context, 1)
        self.assertEqual(tuple(output.shape), (8, 6))
        self.assertTrue(
            torch.equal(model.last_kspace, context["masked_kspace"][1:2])
        )
        self.assertIs(model.last_mask, context["mask"])

    def test_prep_volume_rejects_nonbinary_mask(self):
        kspace = np.ones((1, 1, 4, 4), dtype=np.complex64)
        mask = np.array([0, 1, 0.5, 1], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_volume(Path(temporary), kspace, mask)
            with self.assertRaisesRegex(ValueError, "finite and binary"):
                prep_volume(None, path, torch.device("cpu"))

    def test_prep_volume_rejects_width_mismatch(self):
        kspace = np.ones((1, 1, 4, 4), dtype=np.complex64)
        mask = np.ones(3, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_volume(Path(temporary), kspace, mask)
            with self.assertRaisesRegex(ValueError, "does not match"):
                prep_volume(None, path, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_context_is_fully_resident_before_recon_slice(self):
        kspace = np.ones((2, 3, 16, 12), dtype=np.complex64) * (2 + 3j)
        mask = np.array([1, 0] * 6, dtype=np.float32)
        device = torch.device("cuda:0")

        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_volume(Path(temporary), kspace, mask)
            context = prep_volume(None, path, device)

        self.assertEqual(context["masked_kspace"].device, device)
        self.assertEqual(context["mask"].device, device)
        expected = torch.view_as_real(torch.from_numpy(kspace)).to(device)
        expected.mul_(torch.from_numpy(mask).to(device).view(1, 1, 1, -1, 1))
        self.assertTrue(torch.equal(context["masked_kspace"], expected))

        model = _RecordingModel().to(device)
        output = recon_slice(model, context, 0)
        self.assertEqual(output.device, device)


if __name__ == "__main__":
    unittest.main()
