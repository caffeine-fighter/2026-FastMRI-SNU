import types
import unittest

import numpy as np
import torch

from utils.promptmr.data import PromptMRDataTransform, resize_sampling_mask


class _UpstreamCapture:
    def __init__(self):
        self.last = None

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        self.last = (kspace, mask, target, attrs, fname, slice_num)
        width = np.asarray(mask).size
        return types.SimpleNamespace(
            mask=torch.ones((1, 1, width, 1), dtype=torch.bool),
            masked_kspace=torch.zeros(
                (kspace.shape[0], kspace.shape[-2], width, 2),
                dtype=torch.float32,
            ),
            target=torch.from_numpy(np.asarray(target)),
            max_value=attrs["max"],
            fname=fname,
            slice_num=slice_num,
        )


def _adapter(*, training):
    adapter = PromptMRDataTransform.__new__(PromptMRDataTransform)
    adapter.training = training
    adapter.crop = (384, 384) if training else None
    adapter.transform = _UpstreamCapture()
    return adapter


class PromptMRSamplingMaskTests(unittest.TestCase):
    def test_equal_width_preserves_values_shape_dtype_and_returns_copy(self):
        original = np.array([[0, 1, 0, 1]], dtype=np.uint8)
        result = resize_sampling_mask(original, 4)
        np.testing.assert_array_equal(result, original)
        self.assertEqual(result.shape, original.shape)
        self.assertEqual(result.dtype, original.dtype)
        self.assertFalse(np.shares_memory(result, original))

    def test_center_crop_uses_floor_left_and_ceil_right_for_odd_difference(self):
        # Upstream center cropping starts at diff // 2: odd excess is removed right.
        even_difference = np.arange(8, dtype=np.int16)
        odd_difference = np.arange(9, dtype=np.int16)
        np.testing.assert_array_equal(
            resize_sampling_mask(even_difference, 4), np.arange(2, 6)
        )
        np.testing.assert_array_equal(
            resize_sampling_mask(odd_difference, 4), np.arange(2, 6)
        )

    def test_center_padding_uses_floor_left_and_ceil_right_for_odd_difference(self):
        # Upstream padding uses left = diff // 2 and assigns the remainder right.
        original = np.array([1, 2], dtype=np.float32)
        np.testing.assert_array_equal(
            resize_sampling_mask(original, 4), np.array([0, 1, 2, 0])
        )
        np.testing.assert_array_equal(
            resize_sampling_mask(original, 5), np.array([0, 1, 2, 0, 0])
        )

    def test_supported_numpy_broadcast_shapes_preserve_rank(self):
        shapes = ((5,), (1, 5), (1, 5, 1), (1, 1, 5, 1))
        for shape in shapes:
            with self.subTest(shape=shape):
                mask = np.ones(shape, dtype=np.float32)
                result = resize_sampling_mask(mask, 7)
                expected_shape = list(shape)
                width_axis = 0 if len(shape) == 1 else (2 if len(shape) == 4 else 1)
                expected_shape[width_axis] = 7
                self.assertEqual(result.shape, tuple(expected_shape))
                self.assertEqual(result.dtype, mask.dtype)

    def test_torch_dtype_device_shape_and_input_are_preserved(self):
        mask = torch.tensor([[[[0], [1], [1], [0]]]], dtype=torch.float64)
        before = mask.clone()
        result = resize_sampling_mask(mask, 6)
        self.assertEqual(result.shape, (1, 1, 6, 1))
        self.assertEqual(result.dtype, mask.dtype)
        self.assertEqual(result.device, mask.device)
        self.assertTrue(torch.equal(mask, before))
        result[..., 0, :] = 9
        self.assertTrue(torch.equal(mask, before))

    def test_invalid_width_and_malformed_mask_fail_closed(self):
        for width in (0, -1, 1.5, True):
            with self.subTest(width=width), self.assertRaises((TypeError, ValueError)):
                resize_sampling_mask(np.ones(4), width)
        malformed = (
            np.array(1),
            np.ones((2, 5)),
            np.ones((5, 1)),
            np.ones((1, 1, 1, 5, 1)),
            np.ones(5, dtype=np.complex64),
        )
        for mask in malformed:
            with self.subTest(shape=mask.shape), self.assertRaises((TypeError, ValueError)):
                resize_sampling_mask(mask, 5)

    def test_training_adapter_aligns_mask_to_384_kspace_width(self):
        adapter = _adapter(training=True)
        kspace = np.zeros((10, 400, 400), dtype=np.complex64)
        target = np.zeros((400, 400), dtype=np.float32)
        mask, masked_kspace, *_ = adapter(
            kspace, np.ones(400, dtype=np.uint8), target, 1.0, "v_acc4.h5", 0
        )
        self.assertEqual(adapter.transform.last[1].shape, (384,))
        self.assertEqual(mask.shape[-2], 384)
        self.assertEqual(masked_kspace.shape[-2], 384)

    def test_validation_stays_full_resolution_and_rejects_width_mismatch(self):
        adapter = _adapter(training=False)
        kspace = np.zeros((10, 400, 412), dtype=np.complex64)
        target = np.zeros((400, 412), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "does not match k-space width"):
            adapter(
                kspace, np.ones(411), target, 1.0, "v_acc8.h5", 0
            )
        mask, masked_kspace, *_ = adapter(
            kspace, np.ones(412), target, 1.0, "v_acc8.h5", 0
        )
        self.assertEqual(adapter.transform.last[0].shape[-1], 412)
        self.assertEqual(adapter.transform.last[1].shape, (412,))
        self.assertEqual(mask.shape[-2], 412)
        self.assertEqual(masked_kspace.shape[-2], 412)


if __name__ == "__main__":
    unittest.main()
