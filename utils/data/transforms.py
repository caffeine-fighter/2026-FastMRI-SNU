import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, score_aligned=False):
        self.isforward = isforward
        self.max_key = max_key
        self.score_aligned = score_aligned and not isforward

    def __call__(
        self, mask, input, target, attrs, fname, slice, score_metadata=None
    ):
        if not self.isforward:
            if self.score_aligned:
                from utils.common.metrics import foreground_mask

                if score_metadata is None:
                    raise ValueError("Score-aligned training metadata is required")
                foreground = foreground_mask(target).astype(np.bool_, copy=False)
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        sample = (mask, kspace, target, maximum, fname, slice)
        if not self.score_aligned:
            return sample

        boxes = torch.zeros(
            (score_metadata["max_boxes"], 4), dtype=torch.int64
        )
        active_boxes = score_metadata["boxes"]
        if active_boxes:
            boxes[:len(active_boxes)] = torch.tensor(
                active_boxes, dtype=torch.int64
            )
        metadata = {
            "acceleration": torch.tensor(
                score_metadata["acceleration"], dtype=torch.int64
            ),
            "boxes": boxes,
            "box_count": torch.tensor(len(active_boxes), dtype=torch.int64),
            "foreground_mask": torch.from_numpy(foreground),
            "full_weight": torch.tensor(
                score_metadata["full_weight"], dtype=torch.float64
            ),
            "box_weight": torch.tensor(
                score_metadata["box_weight"], dtype=torch.float64
            ),
        }
        return (*sample, metadata)
