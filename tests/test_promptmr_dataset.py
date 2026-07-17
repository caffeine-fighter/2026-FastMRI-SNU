import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from utils.promptmr.data import PromptMRSliceDataset, resize_sampling_mask


class _CaptureTransform:
    def __init__(self):
        self.last = None

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        self.last = (kspace, mask, target, attrs, fname, slice_num)
        return self.last


class PromptMRDatasetTests(unittest.TestCase):
    def test_training_mask_resize_center_crops_and_zero_pads(self):
        np.testing.assert_array_equal(
            resize_sampling_mask(np.arange(8), 4), np.arange(2, 6)
        )
        np.testing.assert_array_equal(
            resize_sampling_mask(np.array([1, 1]), 5),
            np.array([0, 1, 1, 0, 0]),
        )

    def _write_volume(self, root: Path, name: str, offset: float):
        kspace_path = root / "kspace" / name
        target_path = root / "image" / name
        kspace = np.empty((3, 2, 4, 4), dtype=np.complex64)
        for index in range(3):
            kspace[index] = offset + index + 1j * (offset + index)
        target = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        with h5py.File(kspace_path, "w") as handle:
            handle.create_dataset("kspace", data=kspace)
            handle.create_dataset("mask", data=np.ones(4, dtype=np.uint8))
        with h5py.File(target_path, "w") as handle:
            handle.create_dataset("image_label", data=target)
            handle.attrs["max"] = float(target.max())
        return kspace

    def test_five_slice_edge_replication_and_acceleration_routing(self):
        with tempfile.TemporaryDirectory(prefix="promptmr-data-") as tmp:
            root = Path(tmp)
            (root / "kspace").mkdir()
            (root / "image").mkdir()
            original = self._write_volume(root, "a_acc4.h5", 10.0)
            self._write_volume(root, "b_acc8.h5", 20.0)
            transform = _CaptureTransform()
            dataset = PromptMRSliceDataset(root, transform=transform)
            sample = dataset[0]

        stacked = sample[0]
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.acceleration_volumes, {4: 1, 8: 1})
        self.assertEqual(stacked.shape, (10, 4, 4))
        np.testing.assert_array_equal(stacked[0:2], original[0])
        np.testing.assert_array_equal(stacked[2:4], original[0])
        np.testing.assert_array_equal(stacked[4:6], original[0])
        np.testing.assert_array_equal(stacked[6:8], original[1])
        np.testing.assert_array_equal(stacked[8:10], original[2])

    def test_dataset_rejects_missing_acceleration_route(self):
        with tempfile.TemporaryDirectory(prefix="promptmr-route-") as tmp:
            root = Path(tmp)
            (root / "kspace").mkdir()
            (root / "image").mkdir()
            self._write_volume(root, "only_acc4.h5", 1.0)
            with self.assertRaisesRegex(ValueError, "both acc4 and acc8"):
                PromptMRSliceDataset(root, transform=_CaptureTransform())


if __name__ == "__main__":
    unittest.main()
