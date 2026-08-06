# Final C10 R29 ZF-context single-lineage package

This directory is the only final submission package for the 2026 SNU FastMRI
Challenge. It contains one VESSL-only routed model and the exact source,
environment, commands, contracts, and receipts needed to reproduce it. There is
no fallback model, ensemble candidate, validation-selected checkpoint, or
learned state imported from local or RunPod runs.

The reproduction target is equality with every submitted SSIM item after
rounding to four decimal places. The authoritative score evidence is
`evidence/official-evaluation-receipt.json` after the one team evaluation.

## Package layout and verification

```text
README.md
requirements.txt
recon_eval.sh
reproduce_final.sh
run_official_evaluation_once.sh
verify_package.py
package-manifest.json
best_model.pt
project/                         # exact timed-inference source snapshot
reproduction/                    # exact training source and sealed contracts
evidence/                        # lineage, policy, assembly, and evaluation receipts
```

`package-manifest.json` records SHA-256 and byte size for every submitted file.
Run the semantic verifier before inference:

```bash
python verify_package.py
```

The verifier fails closed unless all of the following are true:

- exactly one learned-state file exists: `best_model.pt`;
- `candidate_count=1`, no fallback, and at most one team official evaluation;
- the embedded E49/ACC4/ACC8/NAF_S source hashes match the VESSL receipts;
- the actual shipped `project/utils/learning/promptmr_router.py` accepts the
  checkpoint contract;
- the R29 source, deployment, CPU-preflight, and inference snapshot hashes match;
- no optimizer, scheduler, scaler, RNG, EMA, or SWA state is packaged.

`materialize_r29_evidence.py` normalizes the authoritative VESSL receipts before
`seal_package.py` creates the manifest. Both operations are one-shot and refuse
to overwrite prior evidence or a prior package.

## Pinned environment

- VESSL Ubuntu, one NVIDIA GeForce GTX 1080 with 8,192 MiB VRAM
- Python 3.10.12
- CUDA 12.1 and PyTorch 2.3.1+cu121
- seed 430, batch size 1, FP32 command-line contract

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_package.py
```

Do not enable TF32, EMA, SWA, a separate GradScaler, a second GPU, or external
learned state. The training implementation uses FP32 master parameters with its
sealed activation/checkpointing and CPU-offloaded AdamW behavior.

## Exact training lineage

| Component | Sealed R29 contract |
|---|---|
| Generalist | compact PromptMR+ R2/C10/H0, fresh VESSL initialization, seed 430 |
| Generalist scheduler | warmup E1, cosine horizon E51 / 238,272 optimizer steps |
| Generalist handoff | atomic E49 checkpoint at optimizer step 228,928 |
| ACC4 specialist | 4,672 steps, LR horizon 35,040, peak LR 2.5e-5 |
| ACC8 specialist | 1,158 steps, LR horizon 2,315, peak LR 5e-5 |
| Shared post-refiner | fresh NAF_S, 72,625 parameters, main C10 frozen |
| NAF_S schedule | 91,231 steps on a fixed 93,567-step LR horizon |
| NAF_S objective | foreground SSIM + L1 + official-384 bbox SSIM, bbox coefficient 0.5 |
| Final artifact | one routed checkpoint; candidate count 1; no fallback |

The active generalist is launched with `--num-epochs 51`; this is the unchanged
cosine scheduler horizon, not the handoff epoch. The controller waits for and
hash-seals `checkpoint-last-000228928.pt`, verifies epoch 49, and only then
stops the generalist. It never rewrites optimizer, scheduler, sampler, or RNG
state. Both specialists start from that exact E49 model, and NAF_S is then
trained with every C10 parameter frozen.

The generalist and specialists use organizer train data. Organizer validation is
appended only as ordinary training data for the terminal NAF_S full push. There
is no validation loop, early stopping, checkpoint selection, or leaderboard/test
target access.

## R29 zero-filled context

NAF_S receives three image-domain channels with unchanged architecture and
parameter count:

1. routed PromptMR reconstruction;
2. zero-filled RSS image computed from the input masked k-space;
3. reconstruction minus zero-filled image.

The zero-filled definition is
`rss(fftshift(ifft2(ifftshift(masked_kspace), norm=ortho)))`. It is center
cropped or zero padded to the reconstruction frame. All three channels share the
detached reconstruction `amax` normalization. No target, bbox, filename, or
leaderboard information enters this input. The local score-free matched probe
was positive for every ACC4/ACC8 full/bbox cell and every tested seed while
retaining exactly 72,625 NAF_S parameters.

## Mask, augmentation, and dispatch policy

Training uses legal native-width Cartesian ACC4/ACC8 masks with ACS width
`round(native_width * 0.08)`. The ACS region is retained. For each legal ACC4
example, acquired non-ACS lines are divided into two complementary virtual ACC8
masks while the original ACC4 example remains. The epoch sampler and optimizer
step budget are defined before augmentation.

Augmentation is never applied to source-target pairing, inference, official
masks, or evaluation. Routing uses only the input k-space mask:

- exact legal ACC4 selects the ACC4 specialist and identity outer view;
- exact legal ACC8 selects the ACC8 specialist and identity plus left-right
  full-pipeline views;
- unknown or mismatched masks select the E49 generalist and identity only.

Mask classification occurs inside the official timed `recon_slice()` call.
Filename, image, target, bbox annotation, and leaderboard result are forbidden
routing inputs. `prep_volume()` only loads input. Training-frame alignment,
all PromptMR forwards, ZF construction, NAF_S, TTA restoration, and averaging
also occur inside `recon_slice()`.

At most one PromptMR expert is resident on CUDA. The active expert remains
resident across consecutive same-route volumes and is offloaded only on an
actual route change. Sensitivity estimation uses coil micro-batches of eight.

## Reproduce training

Mount organizer data at `/root/Data` with train and validation `kspace/` and
`image/` directories. Start with clean result paths and do not copy a previous
checkpoint into the lineage.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=430
export DATA_ROOT=/root/Data
export PROJECT_ROOT=/root/2026-FastMRI-SNU-promptmr-plus
bash reproduce_final.sh
```

The script performs the complete fixed sequence:

1. fresh C10 training on the E51 cosine horizon;
2. exact E49/228,928 hash-sealed handoff;
3. ACC4-4,672 and ACC8-1,158 specialist training from E49;
4. frozen-C10 NAF_S ZF-context training for 91,231 steps;
5. construction and verification of exactly one routed `best_model.pt`.

It fails if a result path exists, a sealed source hash differs, the E49 boundary
is absent, a component budget differs, or another candidate appears.

## Inference and official evaluation

The organizer-compatible repeatable entry point is:

```bash
bash recon_eval.sh
```

It verifies the package, places `best_model.pt` at the organizer's fixed model
path, and executes the unmodified packaged `recon_eval.py`. It remains
repeatable so the organizer may run inference 30 times and use the fastest run.

Our team evaluation is executed exactly once, between 2026-08-20 22:00 and
23:00 KST when training and packaging have completed:

```bash
bash run_official_evaluation_once.sh
```

The wrapper atomically records attempt 1, model and log hashes, per-item SSIM,
reconstruction time, timestamps, and completion before the hard deadline of
2026-08-20 23:59 KST. It prevents a second team evaluation but does not prevent
the organizer from rerunning `recon_eval.sh` after submission. Reconstruction
time is used only under the organizer's identical-SSIM tie rule.
