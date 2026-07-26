import h5py
import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet

# ---------------------------------------------------------------------------
# Team-editable reconstruction contract.
# recon_eval.py (the fixed timing harness) only calls the three functions
# below. This branch feeds `kspace` + `mask` (k-space domain) to a VarNet; a
# U-Net branch reimplements the same three functions for the image domain.
# ---------------------------------------------------------------------------
INPUT_KIND = "kspace"      # harness delivers the kspace H5 to prep_volume


def load_model(args, device):
    model = VarNet(num_cascades=args.cascade,
                   chans=args.chans,
                   sens_chans=args.sens_chans).to(device=device)
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def prep_volume(image_path, kspace_path, device):
    """Prepare one complete masked k-space volume outside the timed region.

    This performs input-only work: complex-to-real layout conversion, float32
    normalization, device transfer, and application of the acquired-sample
    mask. It must not perform sensitivity estimation, coil combination, or a
    model forward pass.
    """
    if kspace_path is None:
        raise ValueError("kspace_path is required for k-space reconstruction")

    with h5py.File(kspace_path, 'r') as hf:
        kspace = np.asarray(hf['kspace'][:], dtype=np.complex64)
        mask = np.asarray(hf['mask'], dtype=np.float32).reshape(-1)

    if kspace.ndim != 4:
        raise ValueError(
            "Expected k-space shape [slices, coils, height, width], "
            f"got {tuple(kspace.shape)}"
        )
    if mask.size != kspace.shape[-1]:
        raise ValueError(
            f"Mask width {mask.size} does not match k-space width {kspace.shape[-1]}"
        )
    if not np.isfinite(kspace.real).all() or not np.isfinite(kspace.imag).all():
        raise ValueError("k-space must be finite")
    if not np.isfinite(mask).all() or not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("Mask must be finite and binary")

    # Zero-copy host layout conversion followed by one float32 device copy.
    volume = torch.view_as_real(torch.from_numpy(kspace))
    volume = volume.to(device=device, dtype=torch.float32, non_blocking=True)

    # [1, 1, 1, W, 1] broadcasts across [S, C, H, W, 2].
    mask_float = torch.from_numpy(mask).to(
        device=device, dtype=torch.float32, non_blocking=True
    ).view(1, 1, 1, -1, 1)
    volume.mul_(mask_float)
    model_mask = mask_float.to(dtype=torch.bool)

    # Keep asynchronous input work outside the timed recon_slice() region.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "masked_kspace": volume,
        "mask": model_mask,
        "device": device,
        "num_slices": volume.shape[0],
    }


def recon_slice(model, ctx, s):
    """Reconstruct a single slice (batch=1). Timed by the harness."""
    masked_kspace = ctx["masked_kspace"][s:s + 1]
    return model(masked_kspace, ctx["mask"])[0]


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    model = load_model(args, device)

    forward_loader = create_data_loaders(data_path=args.data_path, args=args, isforward=True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)

