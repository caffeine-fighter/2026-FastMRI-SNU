# Final evaluation server contract

> Confirmed by the user on 2026-07-14 KST. This document is the hardware and runtime acceptance contract for the final model. Training may use larger LOCAL GPUs, but final promotion and freeze use this server.

## Confirmed hardware

| Component | Final server |
|---|---|
| CPU | Intel Core i7-8700K @ 3.70 GHz |
| CPU topology | 1 socket, 6 cores, 12 threads with Hyper-Threading |
| System memory | 16 GB |
| GPU | NVIDIA GeForce GTX 1080 |
| GPU memory | 8 GB / 8192 MiB class |
| NVIDIA driver | 550.127.08 |
| Driver-reported CUDA capability | CUDA 12.4 |

The operating system, Python version, PyTorch build, `torch.version.cuda`, cuDNN version, and exact package set are not yet confirmed. The `CUDA Version: 12.4` value reported by the driver is a driver compatibility ceiling; it does not prove that a CUDA 12.4 toolkit or PyTorch CUDA 12.4 runtime is installed.

## Consequences for model selection

1. **FP32 is the reference path.** GTX 1080 is a Pascal GPU without Tensor Cores. FP16/autocast may reduce tensor memory, but no speedup is assumed; complex FFT and data-consistency operations may remain FP32, reject half precision, or become slower.
2. **Training fit and inference fit remain separate.** A model trained on RTX 3090 may be deployed unchanged if the exact GTX 1080 forward contract passes. Activation checkpointing and gradient accumulation do not count as inference-memory compression.
3. **Host RAM is part of the contract.** The evaluator has only 16 GB. Record process peak RSS and system available memory while loading the largest volume; do not cache the full dataset, multiple models, or avoidable volume copies.
4. **CPU and transfer work can affect measured time.** The fixed `recon_eval.py` times `recon_slice`, whose current path includes per-slice NumPy/Torch conversion, mask construction, and host-to-device transfer as well as the model forward. Optimize only team-editable code and preserve output parity.
5. **Sparse or low-precision formats are not presumed wins.** Unstructured pruning, INT8, `torch.compile`, and custom kernels require proof on this exact Pascal server; checkpoint-size reduction alone is irrelevant.

## Required environment capture

Before the first promotion probe and again at final freeze, save the output of an approved environment-capture script or equivalent commands:

- OS/kernel and CPU model/topology;
- total/available system RAM;
- `nvidia-smi` driver, GPU name, and total VRAM;
- Python, PyTorch, `torch.version.cuda`, cuDNN, and GPU compute capability;
- exact installed package lock or environment export;
- repository commit, checkpoint SHA-256, model config, and reconstruction command.

Repository helper:

```bash
python scripts/print_run_context.py \
  --checkpoint /absolute/path/to/immutable-model.pt \
  --include-packages
```

During CPU-only preparation while another process owns the GPU, use `--no-gpu-probe`; this deliberately omits the target acceptance evidence and must be rerun without that flag at promotion/freeze.

The current unpinned `requirements.txt` is an installation convenience, not a reproducible final environment. A fresh-clone install verified on the target server and an exact target-compatible lock/export are freeze requirements; do not guess package pins before the target environment is captured.

## Preflight and promotion gates

Use the official `recon_eval.py` call path without modifying that file.

1. Start from `model.eval()`, batch 1, and the existing no-grad control.
2. Use the largest legal spatial shape and coil count found in the challenge data, not an average slice.
3. Warm up, reset CUDA peak statistics, and repeat the probe. Record peak allocated VRAM, peak reserved VRAM, process peak RSS, system available RAM, and ms/slice.
4. Target at least 512 MiB of reserved-VRAM headroom and 2 GiB of available host-RAM headroom. A candidate below either target is not automatically rejected, but it requires a documented exception and 30/30 full-run stability on the exact server before freeze.
5. Re-run the same probe after loading trained weights; random-weight preflight is only an architecture screen.
6. Require complete acc4 and acc8 coverage, finite outputs, unchanged full/bbox metrics within the preregistered parity tolerance, and zero OOMs or unknown skips.
7. Benchmark FP16, `inference_mode`, coil chunking, or other flags independently against the FP32/no-grad control. Keep a flag only when the measured quality, memory, and total-score tradeoff wins on GTX 1080.
8. Run the final approved 30-repeat timing cohort only after model, code, checkpoint, and environment freeze.

## Deployment priority

The order is:

1. direct FP32 deployment of the largest architecture that passes;
2. output-equivalent memory engineering such as per-coil sensitivity and coil chunking;
3. selective precision only when exact-server measurement supports it;
4. a smaller fixed student with distillation only when a non-deployable teacher has a sufficient validated quality gap;
5. pruning, INT8, or compiler specialization only as separately justified last-resort research.

No result from RTX 3090, RTX 4070 Ti SUPER, or a different 8 GB GPU substitutes for the final GTX 1080 acceptance run.
