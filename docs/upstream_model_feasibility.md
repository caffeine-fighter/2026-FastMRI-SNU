# Upstream model-family feasibility record

> Source and license audit performed on 2026-07-15. No PromptMR+ source code was copied, installed, or executed. No GPU was queried or used.

## Decision

Do not rebuild a winner-scale model from E2E-VarNet primitives. Pin a licensed upstream implementation, reproduce its own smoke path, then isolate challenge-specific data, mask, checkpoint, and harness differences in a thin adapter.

## Feature/FI-VarNet — cleared for CPU integration work

- Upstream: `facebookresearch/fastMRI`
- Pinned commit: `91f2df4711adbb6d643df1810f234e4abcf5881b`
- Implementation: `fastmri_examples/feature_varnet/feature_varnet.py`
- Families exposed upstream: Feature VarNet, attention Feature VarNet, FI-VarNet, IF-VarNet, and E2E control
- License: MIT at the repository root
- Repository state: archived/read-only; commit pinning is mandatory

The upstream README claims a separate `fastmri_examples/feature_varnet/requirements.txt`, but that file is absent at the pinned commit. Repository metadata instead references older PyTorch/PyTorch-Lightning test versions. Treat dependency reconstruction as compatibility debt; do not silently modernize the model while establishing the first baseline.

A detached, unmodified audit clone was created outside this repository at `upstream-fastMRI-91f2df47`. Import/forward smoke was attempted with `CUDA_VISIBLE_DEVICES` empty, but the available Hermes Windows Python has no PyTorch installation. The CPU forward is therefore **blocked by the local interpreter**, not failed by the model. Do not install CUDA or a GPU PyTorch build merely to clear this documentation gate; rerun in the existing LOCAL training environment when it is available.

## PromptMR+ — license confirmation still required

- Upstream: `hellopipu/PromptMR-plus`
- Pinned commit: `934eeda6d4d18cd39e406fa1eee9e1f70603cb5e`
- License: Rutgers Non-commercial Research License, Rutgers docket `#2025-032`

The license text permits source/binary use and redistribution for noncommercial purposes, including research at a for-profit company, subject to retaining notices and disclaimers. It does not itself decide whether this competition, a resulting submission package, or later organizer use is noncommercial. Obtain written organizer/team confirmation before copying, installing, adapting, or packaging PromptMR+ code. If confirmation is unavailable, terminate this path and use the MIT Feature/FI implementation.

## Next CPU-only checks

1. In the existing LOCAL PyTorch environment, reproduce a tiny CPU forward from the pinned Feature/FI source with CUDA hidden.
2. Record tensor schema: k-space `[batch, coils, height, width, 2]`, boolean mask, low-frequency count, crop size, and output image.
3. Compare upstream FFT, normalization, sensitivity-map, mask-center, crop, and scale semantics against this repository without changing either implementation.
4. Design a thin adapter and checkpoint namespace; keep upstream algorithm modules intact.
5. Only after CPU schema parity, schedule the largest-input GTX 1080 forward preflight. No GPU preflight was run in this change.

## Sources

- PromptMR+ repository and license: <https://github.com/hellopipu/PromptMR-plus/tree/934eeda6d4d18cd39e406fa1eee9e1f70603cb5e>
- fastMRI Feature/FI implementation: <https://github.com/facebookresearch/fastMRI/tree/91f2df4711adbb6d643df1810f234e4abcf5881b/fastmri_examples/feature_varnet>
- fastMRI MIT license: <https://github.com/facebookresearch/fastMRI/blob/91f2df4711adbb6d643df1810f234e4abcf5881b/LICENSE.md>
