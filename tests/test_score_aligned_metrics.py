import unittest

import numpy as np
import torch

from utils.common.metrics import (
    SSIM,
    ssim_bbox,
    ssim_bbox_tensor,
    ssim_full,
    ssim_full_tensor,
)


class ScoreAlignedMetricTensorTests(unittest.TestCase):
    def test_ssim_bbox_tensor_matches_scalar_and_keeps_gradient(self):
        torch.manual_seed(340712)
        ssim = SSIM(win_size=7)
        target = torch.rand(13, 11, dtype=torch.float32)
        recon = torch.rand(13, 11, dtype=torch.float32, requires_grad=True)
        box = {"x": -2, "y": -3, "width": 13, "height": 16}
        data_range = torch.tensor(1.0, dtype=torch.float64)

        actual = ssim_bbox_tensor(ssim, recon, target, box, data_range)
        expected = ssim_bbox(ssim, recon.detach(), target, box, np.float32(1.0))

        self.assertIsNotNone(actual)
        self.assertEqual(actual.dtype, recon.dtype)
        self.assertEqual(actual.item(), expected)
        (1 - actual).backward()
        self.assertTrue(torch.isfinite(recon.grad).all())
        self.assertGreater(torch.count_nonzero(recon.grad).item(), 0)

    def test_ssim_bbox_tensor_skips_box_smaller_than_window(self):
        ssim = SSIM(win_size=7)
        target = torch.ones(7, 7)
        recon = torch.ones(7, 7, requires_grad=True)
        box = {"x": 0, "y": 0, "width": 6, "height": 7}

        self.assertIsNone(ssim_bbox_tensor(ssim, recon, target, box, 1.0))

    def test_ssim_full_tensor_matches_scalar_and_keeps_gradient(self):
        torch.manual_seed(340711)
        ssim = SSIM(win_size=7)
        target = torch.rand(18, 19, dtype=torch.float32)
        recon = torch.rand(18, 19, dtype=torch.float32, requires_grad=True)
        mask = torch.from_numpy(np.ones((18, 19), dtype=np.float32))
        data_range = torch.tensor(1.0, dtype=torch.float64)

        actual = ssim_full_tensor(ssim, recon, target, mask, data_range)
        expected = ssim_full(ssim, recon.detach(), target, mask, np.float32(1.0))

        self.assertIsNotNone(actual)
        self.assertEqual(actual.dtype, recon.dtype)
        self.assertEqual(actual.item(), expected)
        (1 - actual).backward()
        self.assertTrue(torch.isfinite(recon.grad).all())
        self.assertGreater(torch.count_nonzero(recon.grad).item(), 0)

    def test_ssim_full_tensor_skips_empty_mask(self):
        ssim = SSIM(win_size=7)
        target = torch.ones(7, 7)
        recon = torch.ones(7, 7, requires_grad=True)
        mask = torch.zeros(7, 7)

        self.assertIsNone(ssim_full_tensor(ssim, recon, target, mask, 1.0))


if __name__ == "__main__":
    unittest.main()
