# Final C10 single-lineage reproduction

This package reproduces the single submitted model used for the 2026 SNU
FastMRI Challenge.  The model consists of one compact PromptMR+ C10
generalist followed by one mask-conditioned NAF_S image-domain post-refiner.
There is one routed package, no fallback model, and no candidate selection.

The reproduction target is equality with the submitted leaderboard result for
every reported SSIM item to four decimal places.  Exact checkpoint-byte
identity is not required.

## 1. Required environment

- VESSL instance with one `NVIDIA GeForce GTX 1080` (8,192 MiB)
- Ubuntu with Python 3.10.12
- CUDA 12.1 and cuDNN 8.9.2
- Organizer-provided train, validation, and test data
- Batch size 1 and seed 430

Install the exact Python dependencies from the package root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the main runtime versions:

```bash
python - <<'PY'
import h5py
import numpy as np
import torch

assert torch.__version__ == "2.3.1+cu121"
assert torch.version.cuda == "12.1"
assert np.__version__ == "1.24.4"
assert h5py.__version__ == "3.11.0"
assert torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080"
print("VESSL_ENVIRONMENT_OK")
PY
```

The training entrypoints set seed 430, `torch.backends.cudnn.deterministic =
True`, and `torch.backends.cudnn.benchmark = False`.  Do not enable TF32, an
external AMP/GradScaler policy, EMA, SWA, or an alternative CUDA device.  The
compact C10 source's built-in FP16 activation autocast with FP32 master
parameters is part of the fixed recipe and must remain enabled.

## 2. Package and data layout

Run all commands from a clean extracted package with this layout:

```text
README.md
requirements.txt
source/
reproduction/
  organizer-data-provenance.json
  vessl_train_post_refiner.py
  vessl_build_routed_promptmr_checkpoint.py
result/                         # created by the commands below
```

The organizer data must be mounted read-only as follows:

```text
/root/Data/train/kspace/*.h5
/root/Data/train/image/*.h5
/root/Data/val/kspace/*.h5
/root/Data/val/image/*.h5
```

Set the paths once:

```bash
export PACKAGE_ROOT="$(pwd)"
export SOURCE_ROOT="$PACKAGE_ROOT/source"
export REPRO_ROOT="$PACKAGE_ROOT/reproduction"
export DATA_ROOT=/root/Data
export TRUSTED_DATA_MANIFEST="$REPRO_ROOT/organizer-data-provenance.json"
export RUN_NAME=EXP_PROMPTMR_R2_C10_G20_FINAL_DELAY5_COS50_E40_SEED430_V1
export RUN_DIR="$PACKAGE_ROOT/result/$RUN_NAME"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=430

test -f "$TRUSTED_DATA_MANIFEST"
test -d "$DATA_ROOT/train/kspace"
test -d "$DATA_ROOT/train/image"
test -d "$DATA_ROOT/val/kspace"
test -d "$DATA_ROOT/val/image"
mkdir -p "$PACKAGE_ROOT/result"
```

Only organizer train data is used to update C10.  Organizer validation data is
never used for model selection or early stopping.  No leaderboard or test
payload is read during training.

## 3. C10 phase A: reproduce steps 1-31,500

The submitted lineage began with a 50-epoch cosine horizon.  Reproduce this
state exactly and stop only after the atomic step-31,500 checkpoint appears.
The validation passes completed before this transition are diagnostic only;
they do not select a checkpoint or update any learned state.

```bash
cd "$SOURCE_ROOT"

python -u train.py \
  --model-family promptmr-plus \
  --promptmr-production \
  --promptmr-rung R2 \
  --promptmr-num-cascades 10 \
  --promptmr-n-history 0 \
  --promptmr-compact-fallback \
  --promptmr-train-acceleration all \
  --promptmr-mraugment conservative_delay5 \
  --promptmr-legal-mask-family \
  --promptmr-lr-schedule cos50 \
  --precision fp32 \
  --require-cuda-device-name "NVIDIA GeForce GTX 1080" \
  --GPU-NUM 0 \
  --batch-size 1 \
  --num-epochs 50 \
  --lr 0.0001 \
  --seed 430 \
  --net-name "$RUN_NAME" \
  --data-path-train "$DATA_ROOT/train" \
  --data-path-val "$DATA_ROOT/val" \
  --trusted-data-manifest "$TRUSTED_DATA_MANIFEST" \
  --report-interval 50 \
  >"$PACKAGE_ROOT/result/c10-phase-a.log" 2>&1 &
C10_PHASE_A_PID=$!

C10_TRANSITION="$RUN_DIR/checkpoints/checkpoint-last-000031500.pt"
while kill -0 "$C10_PHASE_A_PID" 2>/dev/null; do
  if test -f "$C10_TRANSITION"; then
    kill -TERM "$C10_PHASE_A_PID"
    break
  fi
  sleep 5
done
wait "$C10_PHASE_A_PID" || true
test -f "$C10_TRANSITION"
```

The training set uses legal native-width Cartesian masks with ACS width
`round(width * 0.08)`.  Each legal ACC4 example is retained and also supplies
two complementary virtual ACC8 masks.  The pre-augmentation sampler budget is
fixed at 2,336 samples per acceleration, or 4,672 optimizer steps per epoch.

## 4. C10 phase B: no-validation full push to E51

Resume the complete optimizer, scheduler, sampler, and RNG state from step
31,500.  The scheduler object is rebuilt with the final 238,272-step horizon
before loading the saved scheduler state.  No validation dataset is
constructed and no validation forward is run in this phase.

```bash
C10_TRANSITION_SHA256="$(sha256sum "$C10_TRANSITION" | awk '{print $1}')"

cd "$SOURCE_ROOT"
python -u train.py \
  --model-family promptmr-plus \
  --promptmr-production \
  --promptmr-rung R2 \
  --promptmr-num-cascades 10 \
  --promptmr-n-history 0 \
  --promptmr-compact-fallback \
  --promptmr-train-acceleration all \
  --promptmr-mraugment conservative_delay5 \
  --promptmr-legal-mask-family \
  --promptmr-lr-schedule cos50 \
  --promptmr-skip-validation \
  --precision fp32 \
  --require-cuda-device-name "NVIDIA GeForce GTX 1080" \
  --GPU-NUM 0 \
  --batch-size 1 \
  --num-epochs 51 \
  --lr 0.0001 \
  --seed 430 \
  --net-name "$RUN_NAME" \
  --data-path-train "$DATA_ROOT/train" \
  --trusted-data-manifest "$TRUSTED_DATA_MANIFEST" \
  --report-interval 50 \
  --resume-checkpoint "$C10_TRANSITION" \
  --resume-checkpoint-sha256 "$C10_TRANSITION_SHA256" \
  >"$PACKAGE_ROOT/result/c10-phase-b.log" 2>&1

C10_FINAL="$RUN_DIR/checkpoints/checkpoint-last-000238272.pt"
test -f "$C10_FINAL"
C10_FINAL_SHA256="$(sha256sum "$C10_FINAL" | awk '{print $1}')"
```

The expected C10 terminal state is epoch 51 and optimizer step 238,272.  The
C10 parameters are frozen after this point.

## 5. Mask-conditioned NAF_S post-refiner

Train one fresh NAF_S post-refiner for 91,141 optimizer steps.  Organizer train
and organizer validation are both used as final training data in this phase;
there is no validation loop, early stopping, or checkpoint selection.  The
four listed views are registered post-refiner views used during training and
reproduced during inference in one batched refiner application.  The final
routed package uses one C10 forward and one batched post-refiner application
per slice.

```bash
NAF_DIR="$PACKAGE_ROOT/result/VESSL_POST_REFINER_R2_C10_NAF_S_E20_WINNER_FULLDATA_SEED430_R3"

cd "$SOURCE_ROOT"
python -u "$REPRO_ROOT/vessl_train_post_refiner.py" \
  --base-checkpoint "$C10_FINAL" \
  --base-checkpoint-sha256 "$C10_FINAL_SHA256" \
  --variant NAF_S \
  --views identity flip_lr flip_ud rot180 \
  --epochs 20 \
  --output-dir "$NAF_DIR" \
  --train-root "$DATA_ROOT/train" \
  --trusted-data-manifest "$TRUSTED_DATA_MANIFEST" \
  --extra-train-root "$DATA_ROOT/val" \
  --extra-trusted-data-manifest "$TRUSTED_DATA_MANIFEST" \
  --loss-family winner_foreground_ssim_l1_sqrt_area_v1 \
  --peak-lr 0.0001 \
  --weight-decay 0.0001 \
  --seed 430 \
  --optimizer-steps 91141 \
  --mask-conditioned \
  >"$PACKAGE_ROOT/result/naf-s.log" 2>&1

NAF_CKPT="$(python - "$NAF_DIR/receipt.json" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text())["checkpoint"])
PY
)"
NAF_SHA256="$(sha256sum "$NAF_CKPT" | awk '{print $1}')"
test -f "$NAF_CKPT"
```

The post-refiner has 74,065 trainable parameters, including the 1,440-parameter
mask conditioner.  Acceleration is inferred only from the input k-space mask.
Exact legal ACC4 and ACC8 masks use their conditioned route; unknown or
mismatched masks use the generalist condition.  Filenames, targets, image
fields, bounding boxes, and leaderboard results are not routing inputs.

## 6. Build the one final routed checkpoint

The same C10 checkpoint is used for both acceleration routes.  There is no
second candidate or fallback checkpoint.

```bash
FINAL_DIR="$PACKAGE_ROOT/result/final-routed-package"
mkdir -p "$FINAL_DIR"

cd "$SOURCE_ROOT"
python -u "$REPRO_ROOT/vessl_build_routed_promptmr_checkpoint.py" \
  --acc4-checkpoint "$C10_FINAL" \
  --acc4-sha256 "$C10_FINAL_SHA256" \
  --acc8-checkpoint "$C10_FINAL" \
  --acc8-sha256 "$C10_FINAL_SHA256" \
  --tta-views identity \
  --post-refiner-checkpoint "$NAF_CKPT" \
  --post-refiner-sha256 "$NAF_SHA256" \
  --output "$FINAL_DIR/best_model.pt"

sha256sum "$FINAL_DIR/best_model.pt"
```

All dispatch and model forwards occur inside the organizer-timed
`recon_slice()` implementation.  `prep_volume()` only prepares the input.

## 7. Official reconstruction and score comparison

Use the organizer-provided, unmodified `recon_eval.py` and wrapper.  Place the
single routed checkpoint at the path expected by that wrapper:

```bash
mkdir -p "$PACKAGE_ROOT/result/test_Varnet/checkpoints"
cp "$FINAL_DIR/best_model.pt" \
  "$PACKAGE_ROOT/result/test_Varnet/checkpoints/best_model.pt"

cd "$SOURCE_ROOT"
bash recon_eval.sh
```

Do not train, validate, route, or select using leaderboard/test targets or
scores.  Run official evaluation once after the final package is complete.
The reproduction is accepted when every leaderboard SSIM item equals the
submitted result after rounding to four decimal places.

## 8. Fixed lineage summary

| Component | Fixed contract |
|---|---|
| C10 | compact PromptMR+ R2/C10/H0, seed 430, batch 1 |
| C10 steps 1-31,500 | cosine-50 state, diagnostic validation allowed |
| C10 steps 31,501-238,272 | cosine-51 full push, validation forwards 0 |
| C10 data | organizer train only |
| NAF_S | 20 epochs, 91,141 steps, mask-conditioned, fresh initialization |
| NAF_S data | organizer train plus organizer validation as training data |
| Final package | one routed checkpoint, candidate count 1, no fallback |
| Official evaluation | one run; SSIM is the primary ranking value |
