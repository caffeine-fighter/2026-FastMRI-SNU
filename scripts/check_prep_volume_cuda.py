import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_UTILS = str(ROOT / "utils" / "model")
if MODEL_UTILS not in sys.path:
    sys.path.insert(0, MODEL_UTILS)

from utils.learning.test_part import prep_volume


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure untimed full-volume CUDA input preprocessing."
    )
    parser.add_argument("kspace_h5", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    context = prep_volume(None, args.kspace_h5, device)
    torch.cuda.synchronize(device)

    print("REAL_VOLUME_PREP=PASS")
    print(f"shape={tuple(context['masked_kspace'].shape)}")
    print(f"kspace_dtype={context['masked_kspace'].dtype}")
    print(f"mask_dtype={context['mask'].dtype}")
    print(f"allocated_mib={torch.cuda.max_memory_allocated(device) / 2**20:.1f}")
    print(f"reserved_mib={torch.cuda.max_memory_reserved(device) / 2**20:.1f}")


if __name__ == "__main__":
    main()

