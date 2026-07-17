# Upstream model-family feasibility record

> Source and license audit refreshed on 2026-07-16. PromptMR+ is pinned and vendored only through the exact files listed in `third_party/promptmr_plus/SOURCE_MANIFEST.json`; no long training or official evaluation is authorized.

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

## PromptMR+ — noncommercial competition use allowed; feasibility pending

- Upstream: `hellopipu/PromptMR-plus`
- Pinned commit: `934eeda6d4d18cd39e406fa1eee9e1f70603cb5e`
- License: Rutgers Non-commercial Research License, Rutgers docket `#2025-032`
- Status: `NONCOMMERCIAL_COMPETITION_USE_ALLOWED`
- License SHA-256: `c0c4c7d85180b493cd7a213d4509155d3734de26562e4490589963e1c356db21`

For this noncommercial research/competition workflow, the RU-NCRL permits use, modification, and redistribution provided that the copyright notice, license conditions, and disclaimer are preserved. Commercial use requires separate rights and is not implied. Prior SNU FastMRI competition use is useful integration precedent, but it does not prove this year's quality, 8 GB deployment feasibility, checkpoint compatibility, or official harness acceptance.

The exact upstream implementation and representative FastMRI brain/knee configs are preserved byte-for-byte under `third_party/promptmr_plus/`; the manifest verifies every vendored source/config/license hash before import. The local layer is restricted to data, mask, adjacent-slice, checkpoint-namespace, crop/output, and packaging adapters. Architecture feasibility and competition quality remain separate gates.

Official checkpoint metadata is pinned to Hugging Face revision `fd1642e375e29ee515696d802179c099ee08d737`. The brain PromptMR+ checkpoint (`step=1591830`, SHA-256 `42722018604944c567c598ddf5c488d135793ef337359a1da03ad3d4301e177e`) and knee checkpoint (`step=781695`, SHA-256 `3f931e9fd5eed3f755c580760de04bf7b870bc5d0a9b39c5164955359f385a86`) were both trained jointly for acc4/acc8 with `combine_train_val: false`. The brain source config contains a stale filename for the same renamed LFS object. The knee checkpoint's metadata says legacy `promptmr` even though its state shapes strictly load into v2; do not claim semantic identity or force its use. Checkpoints are initialization/namespace evidence only, not independent quality evidence.

## Current feasibility gate

1. Exact source/config/license hash verification, five-adjacent-slice schema, bool-mask layout, acceleration-token routing, crop/output contract, strict checkpoint namespace, and real CPU FP32 forward are implemented and tested.
2. The pinned v2 source has a missing `PromptMRBlock.n_buffer` attribute; the adapter mirrors the already-configured `cascade.model.n_buffer` at runtime without editing upstream algorithm bytes.
3. The next gate is the independently reviewed, no-training maximum-input GTX 1080 FP32 probe. If default inference OOMs, apply only the documented `compute_sens_per_coil` control before considering any other mode.
4. No e5/e15/e30 training, official evaluation, repeated timing cohort, or leaderboard submission is authorized by a feasibility PASS.

## Sources

- PromptMR+ repository and license: <https://github.com/hellopipu/PromptMR-plus/tree/934eeda6d4d18cd39e406fa1eee9e1f70603cb5e>
- fastMRI Feature/FI implementation: <https://github.com/facebookresearch/fastMRI/tree/91f2df4711adbb6d643df1810f234e4abcf5881b/fastmri_examples/feature_varnet>
- fastMRI MIT license: <https://github.com/facebookresearch/fastMRI/blob/91f2df4711adbb6d643df1810f234e4abcf5881b/LICENSE.md>
