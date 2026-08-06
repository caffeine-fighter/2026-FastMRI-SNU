# Final C10 R25 E49 single-lineage package

This directory is the only final submission package for the 2026 SNU FastMRI
Challenge. It contains one VESSL-only routed model and the source, environment,
training receipts, and commands required to reproduce it. There is no fallback
model, ensemble candidate, or checkpoint selected from validation or
leaderboard results.

The reproduction target is equality with every submitted SSIM item after
rounding to four decimal places. The official evaluation receipt in
`evidence/official-evaluation-receipt.json` is authoritative once the final
evaluation has completed.

## Package contents

The sealed package has the following layout:

```text
README.md
requirements.txt
recon_eval.sh
record_official_evaluation.py
reproduce_final.sh
run_official_evaluation_once.sh
materialize_r23_evidence.py
seal_package.py
verify_package.py
package-manifest.json
best_model.pt
project/                         # exact inference project snapshot
reproduction/                    # exact R23 parent, R24 boundary, and R25 sources/contracts
evidence/                        # lineage, policy, package, and evaluation receipts
```

`package-manifest.json` records SHA-256 and byte size for every submitted file.
After all VESSL artifacts and receipts have been copied into this directory,
`python seal_package.py` creates that manifest exactly once and immediately
runs the complete semantic verifier. It refuses pre-existing evaluation output,
symlinks, or more than one learned-state file.

Before sealing, `materialize_r23_evidence.py` (a legacy filename retained for
tooling compatibility) converts the raw VESSL controller,
E49, specialist-terminal, and NAF_S receipts into the normalized evidence files.
It cross-checks every source hash, the R24 parent boundary, the R25 deployment,
and the R25 post-E49 command/parser preflight
against `best_model.pt`, and refuses an admission fallback, a non-E49 parent,
or an incomplete specialist budget. When explicit component receipt paths are
omitted, their content-addressed run directories are read from the sealed final
controller receipt.

```bash
python materialize_r23_evidence.py \
  --controller-receipt \
    /root/result/VESSL_G10_G11_TERMINAL_SUCCESSOR_AMENDMENT_R25_ACC4_E2_NAF_TAIL_R1/receipt.json \
  --scheduler-deployment-receipt \
    /root/result/VESSL_G10_G11_TERMINAL_SUCCESSOR_AMENDMENT_R24_SCHEDULER_FIX_R1/r24-deployment-receipt.json \
  --r25-deployment-receipt \
    /root/result/VESSL_G10_G11_TERMINAL_SUCCESSOR_AMENDMENT_R25_ACC4_E2_NAF_TAIL_R1/r25-deployment-receipt.json
```

The command is intentionally one-shot: existing normalized evidence is never
overwritten. Run it only after `best_model.pt` and the raw receipts have been
copied into their final package locations, and before `python seal_package.py`.
`verify_package.py` rejects an unsealed manifest, a second learned-state file,
`candidate_count != 1`, a fallback, an E51 parent, or a hash mismatch. It also
loads `best_model.pt` with PyTorch's safe weights-only loader and verifies the
exact E49/ACC4/ACC8/NAF_S embedded lineage before inference. Generalist,
specialist, refiner, policy, and GTX1080 admission receipts are then
cross-checked against the hashes and contracts embedded in that model.

## Environment

- VESSL Ubuntu instance
- one `NVIDIA GeForce GTX 1080` with 8,192 MiB VRAM
- Python 3.10.12
- CUDA 12.1 and PyTorch 2.3.1+cu121
- seed 430, batch size 1, FP32 CLI contract

Install the pinned environment and verify the package before inference:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_package.py
```

The training runtime uses FP32 master parameters with its pinned activation
autocast, activation checkpointing, and CPU-offloaded AdamW implementation.
Do not enable TF32, EMA, SWA, a separate GradScaler, external learned state, or
another GPU.

## Fixed R25 lineage

| Component | Sealed contract |
|---|---|
| Generalist | compact PromptMR+ R2/C10/H0, fresh VESSL initialization, seed 430 |
| Scheduler | one-epoch warmup and cosine horizon E51/238,272 steps |
| Parent contracts | R23 E49 handoff plus the R24 equal-boundary scheduler fix |
| Generalist handoff | exact atomic E49 checkpoint, optimizer step 228,928 |
| Generalist data | organizer train only; validation forwards 0 |
| ACC4 specialist | E2 prefix, 4,672 steps, 35,040-step LR horizon, peak LR 2.5e-5, exact upstream SSIM |
| ACC8 specialist | first 1,158/2,315 cosine-horizon steps, peak LR 5e-5 |
| ACC8 specialist data | organizer real ACC8 only, no MR augmentation |
| NAF_S | fresh 72,625-parameter refiner, 91,231 steps on a fixed 93,567-step LR horizon (19.527 epoch-equivalent) |
| NAF_S data | organizer train plus organizer validation used only as training data |
| NAF_S objective | foreground SSIM + L1 + official-384 bbox SSIM, bbox coefficient 0.5 |
| NAF_S input | outputs of the actual routed ACC4/ACC8 specialists |
| Final model | one routed checkpoint, candidate count 1, no fallback |
| Team official evaluation | one `bash run_official_evaluation_once.sh` invocation |

The generalist process is launched with `--num-epochs 51`; this defines the
unchanged scheduler horizon. The R23 boundary controller waits for the atomic
E49 file `checkpoint-last-000228928.pt`, verifies and hash-seals it, and only
then stops the trainer. No optimizer, scheduler, sampler, RNG, EMA, or SWA
state is rewritten. The R25 CPU controller preserves that live trainer and all
downstream components bind to the exact E49 hash.

Only organizer train data updates the C10 generalist and both specialists.
Organizer validation is appended as training data only for the terminal NAF_S
stage. There is no validation loop, early stopping, checkpoint selection, or
leaderboard/test-target access at any stage.

## Mask and augmentation policy

Training uses legal native-width Cartesian ACC4/ACC8 masks with ACS width
`round(native_width * 0.08)`. The ACS region is always retained. For each legal
ACC4 example, non-ACS acquired lines are divided into two complementary virtual
ACC8 masks; the original ACC4 example is retained. The sampler keeps the
pre-augmentation optimizer-step budget.

Augmentation is never applied to organizer validation during evaluation,
inference, official masks, or source-target pairing. Routing never reads a
filename, image field, target, bbox annotation, public/private score, or
leaderboard result.

## Routed inference

The final checkpoint contains the E49 generalist, one ACC4 specialist, one
ACC8 specialist, and one shared NAF_S refiner.

- Exact legal ACC4 masks select the ACC4 specialist.
- Exact legal ACC8 masks select the ACC8 specialist.
- Unknown or mismatched masks select the E49 generalist in the same package.
- ACC4 and unknown routes use one identity pipeline.
- ACC8 uses identity and left-right-flipped full-pipeline views, restores both
  outputs, and averages them.

The input k-space is aligned to the 384-row training frame inside timed
`recon_slice()` using IFFT, center crop, FFT, and official-mask reapplication.
Sensitivity estimation uses coil micro-batches of eight. Routing, alignment,
all PromptMR forwards, NAF_S, TTA, restoration, and averaging occur inside the
official timed `recon_slice()` call. `prep_volume()` only loads input.

At most one PromptMR expert is resident on CUDA. The selected expert remains
resident across consecutive volumes on the same route and is offloaded only
when the exact input mask changes route. The final GTX1080 admission receipt
records max-shape VRAM, finite output, save/reload parity, and official-wrapper
compatibility.

## Reproduce training

Mount organizer data at `/root/Data` with train and validation `kspace/` and
`image/` directories. Start from a clean result directory; do not copy a local,
RunPod, leaderboard, or previous VESSL checkpoint into it.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=430
export DATA_ROOT=/root/Data
export PROJECT_ROOT=/root/2026-FastMRI-SNU-promptmr-training
bash reproduce_final.sh
```

`reproduce_final.sh` performs the complete fixed sequence:

1. fresh C10 training on the E51 cosine horizon;
2. exact E49/228,928 boundary hash-seal;
3. fresh ACC4-4,672 and ACC8-1,158 specialist runs from the E49 model;
4. C10 freeze verification and fresh NAF_S-91,231 training on its sealed
   93,567-step LR horizon;
5. construction of exactly one routed `best_model.pt`;
6. package and policy verification.

The script fails closed if its result directory is non-empty, an exact parent
hash differs, the E49 checkpoint is absent, a second candidate appears, or any
registered source differs from `reproduction/source-sha256sums.txt`.

## Official evaluation

The repeatable inference entry point required by the organizer is one command:

```bash
bash recon_eval.sh
```

This wrapper verifies the sealed manifest and unique checkpoint, places that
checkpoint at the organizer's fixed result path, and invokes the unmodified
`project/recon_eval.py`. It remains repeatable so the organizer can execute it
30 times and use the fastest timing result.

Our one allowed official evaluation is run with:

```bash
bash run_official_evaluation_once.sh
```

That outer wrapper has a fail-closed start record preventing a second team
evaluation. It atomically seals the output metrics, log hash, model hash,
timestamps, and attempt count into the package manifest and official
evaluation receipt. It does not prevent the organizer from rerunning
`recon_eval.sh` after submission.

During the one team evaluation, `recon_eval.sh` invokes the verifier in its
strict `--evaluation-in-progress` mode. That mode permits only the exact
attempt-1 start marker and the live log in addition to the pre-evaluation
manifest; all sealed files, model semantics, and lineage receipts are still
verified. Normal organizer runs before or after that interval use the ordinary
sealed-package verifier.

SSIM is the ranking value. Reconstruction time is relevant only under the
organizer's exact-SSIM tie rule.

## Evidence

The final package is complete only when all of the following sealed receipts
are present and referenced by `package-manifest.json`:

- E49 generalist checkpoint path, SHA-256, epoch, and optimizer step;
- ACC4 and ACC8 specialist parent hashes and terminal-step receipts;
- frozen-C10 NAF_S training receipt and routed-parent hashes;
- augmentation, legal-mask, unknown-route, and dispatch-parity receipt;
- candidate count 1, no fallback, no external learned state, and no
  leaderboard/test training influence receipt;
- final package hash, inference admission, and source snapshot manifest;
- official evaluation or submission receipt dated before
  `2026-08-20 23:59 KST`.

Until those artifacts exist, the package is not claimed as the final evaluated
model.
