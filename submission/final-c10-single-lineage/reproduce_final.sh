#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGED_PROJECT="$PACKAGE_ROOT/project"
PROJECT_ROOT="${PROJECT_ROOT:-/root/2026-FastMRI-SNU-promptmr-training}"
REPRO_ROOT="$PACKAGE_ROOT/reproduction"
DATA_ROOT="${DATA_ROOT:-/root/Data}"
RESULT_ROOT=/root/result
MANIFEST="$REPRO_ROOT/organizer-data-provenance.json"
R24_AMENDMENT="$REPRO_ROOT/FINAL_C10_SINGLE_LINEAGE_R24_SCHEDULER_BOUNDARY.json"
R24_POST_E49_PREFLIGHT="$REPRO_ROOT/R24_POST_E49_COMMAND_PARSER_PREFLIGHT.json"
R24_SPECIALIST_PRODUCTION_SHA256="ea4695f5fada7c417323d9efad495544d0743ad1d35b3023c1a645a421d8688b"
R24_POST_E49_PREFLIGHT_SHA256="5d323e4894b53cbc9b77b5e46c570580bee60baeb1444b68cb6e62008ba0a6cd"
RUN_NAME="EXP_PROMPTMR_R2_C10_G20_FINAL_DELAY5_COS50_E40_SEED430_R23_REPRO"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"
E49_STEP=228928
E49_CHECKPOINT="$RUN_DIR/checkpoints/checkpoint-last-000228928.pt"

python "$PACKAGE_ROOT/verify_package.py" --structure-only
test -d "$PACKAGED_PROJECT"
if ! test -d "$PROJECT_ROOT"; then
  cp -a -- "$PACKAGED_PROJECT" "$PROJECT_ROOT"
fi
test -f "$PROJECT_ROOT/recon_eval.py"
test -f "$REPRO_ROOT/source-sha256sums.txt"
test -f "$MANIFEST"
test -f "$R24_AMENDMENT"
test -f "$R24_POST_E49_PREFLIGHT"
test -d "$DATA_ROOT/train/kspace"
test -d "$DATA_ROOT/train/image"
test -d "$DATA_ROOT/val/kspace"
test -d "$DATA_ROOT/val/image"
mkdir -p "$RESULT_ROOT"
for path in \
  "$RUN_DIR" \
  "$RESULT_ROOT/EXP_PROMPTMR_R2_C10_ACC4_G10_FINAL_E1_S2336_SEED430_R23_REPRO" \
  "$RESULT_ROOT/EXP_PROMPTMR_R2_C10_ACC8_G10_FINAL_E1_S1158_SEED430_R23_REPRO" \
  "$RESULT_ROOT/VESSL_POST_REFINER_R2_C10_NAF_S_E21_BBOX05_R23_REPRO" \
  "$RESULT_ROOT/final-r23-single-package"; do
  if test -e "$path"; then
    echo "refusing to overwrite existing reproduction lineage: $path" >&2
    exit 1
  fi
done

cd "$REPRO_ROOT"
sha256sum -c source-sha256sums.txt

python - "$R24_AMENDMENT" "$REPRO_ROOT/specialist/promptmr_production.py" \
  "$R24_SPECIALIST_PRODUCTION_SHA256" "$R24_POST_E49_PREFLIGHT" \
  "$R24_POST_E49_PREFLIGHT_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

amendment_path = Path(sys.argv[1])
production_path = Path(sys.argv[2])
expected_production = sys.argv[3]
preflight_path = Path(sys.argv[4])
expected_preflight = sys.argv[5]
amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
assert amendment["schema"] == (
    "final-c10-single-lineage-r24-scheduler-boundary-amendment-v1"
)
assert amendment["state"] == "SEALED"
assert amendment["amendment"]["equal_boundary_semantics"] == (
    "ONE_EPOCH_LINEAR_WARMUP_WITH_ZERO_COSINE_TAIL"
)
assert amendment["amendment"]["r23_recipe_changed"] is False
assert amendment["source_hashes"]["promptmr_production.py"] == expected_production
assert hashlib.sha256(production_path.read_bytes()).hexdigest() == expected_production
assert hashlib.sha256(preflight_path.read_bytes()).hexdigest() == expected_preflight
assert preflight["schema"] == "vessl-r24-post-e49-command-parser-preflight-v1"
assert preflight["state"] == "PASS"
assert preflight["cpu_only"] is True
assert preflight["cuda_initialized"] is False
assert preflight["remote_process_changed"] is False
assert preflight["active_generalist_process_touched"] is False
assert preflight["recipe_changed"] is False
assert preflight["candidate_count"] == 1
assert preflight["fallback_registered"] is False
assert preflight["handoff"] == {
    "epoch": 49,
    "optimizer_step": 228928,
    "scheduler_horizon_epoch": 51,
    "scheduler_horizon_optimizer_step": 238272,
}
assert all(
    entry["parser"] == "PASS"
    for entry in preflight["specialists"].values()
)
assert preflight["post_refiner"]["parser"] == "PASS"
assert preflight["post_refiner"]["main_c10_frozen"] is True
assert preflight["post_refiner"]["optimizer_scope"] == "naf_s_only"
assert preflight["post_refiner"]["bbox_loss_coefficient"] == 0.5
assert preflight["final_builder"]["parser"] == "PASS"
assert preflight["final_builder"]["candidate_count"] == 1
print("R24_SCHEDULER_REPRODUCTION_SOURCE_OK")
print("R24_POST_E49_COMMAND_PARSER_PREFLIGHT_OK")
PY

install -m 0644 "$REPRO_ROOT/generalist/train.py" "$PROJECT_ROOT/train.py"
install -m 0644 "$REPRO_ROOT/generalist/promptmr_production.py" \
  "$PROJECT_ROOT/utils/learning/promptmr_production.py"

cd "$PROJECT_ROOT"
setsid python -u train.py \
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
  --trusted-data-manifest "$MANIFEST" \
  --report-interval 50 \
  >"$RESULT_ROOT/generalist.log" 2>&1 &
GENERALIST_PID=$!

while kill -0 "$GENERALIST_PID" 2>/dev/null; do
  if test -f "$E49_CHECKPOINT"; then
    python - "$E49_CHECKPOINT" "$E49_STEP" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
expected_step = int(sys.argv[2])
state = torch.load(path, map_location="cpu", weights_only=True)
assert int(state["epoch"]) == 49
assert int(state["global_optimizer_step"]) == expected_step
print("EXACT_E49_CHECKPOINT_OK")
PY
    kill -TERM -- "-$GENERALIST_PID"
    break
  fi
  sleep 15
done
wait "$GENERALIST_PID" || GENERALIST_RETURN=$?
if ! test -f "$E49_CHECKPOINT"; then
  echo "generalist exited before exact E49 checkpoint" >&2
  exit 1
fi
C10_SHA256="$(sha256sum "$E49_CHECKPOINT" | awk '{print $1}')"

install -m 0644 "$REPRO_ROOT/specialist/train.py" "$PROJECT_ROOT/train.py"
install -m 0644 "$REPRO_ROOT/specialist/promptmr_production.py" \
  "$PROJECT_ROOT/utils/learning/promptmr_production.py"

ACC4_RUN="EXP_PROMPTMR_R2_C10_ACC4_G10_FINAL_E1_S2336_SEED430_R23_REPRO"
ACC4_DIR="$RESULT_ROOT/$ACC4_RUN"
python -u train.py \
  --model-family promptmr-plus --promptmr-production --promptmr-rung R2 \
  --promptmr-num-cascades 10 --promptmr-n-history 0 --promptmr-compact-fallback \
  --promptmr-train-acceleration acc4 \
  --promptmr-mraugment conservative_immediate --promptmr-legal-mask-family \
  --promptmr-lr-schedule specialist_warmup1_cosine --promptmr-skip-validation \
  --precision fp32 --require-cuda-device-name "NVIDIA GeForce GTX 1080" \
  --GPU-NUM 0 --batch-size 1 --num-epochs 1 \
  --promptmr-stop-after-optimizer-steps 2336 \
  --promptmr-specialist-lr-horizon-optimizer-steps 2336 \
  --promptmr-mraugment-seed 433 --promptmr-legal-mask-seed 438 \
  --promptmr-specialist-loss-family exact_upstream_ssim \
  --promptmr-specialist-trainable-scope all --lr 0.00001 --seed 430 \
  --net-name "$ACC4_RUN" --data-path-train "$DATA_ROOT/train" \
  --data-path-val "$DATA_ROOT/val" --trusted-data-manifest "$MANIFEST" \
  --report-interval 50 --promptmr-vessl-model-only-import "$E49_CHECKPOINT" \
  --promptmr-vessl-model-only-import-sha256 "$C10_SHA256" \
  >"$RESULT_ROOT/acc4-specialist.log" 2>&1

ACC8_RUN="EXP_PROMPTMR_R2_C10_ACC8_G10_FINAL_E1_S1158_SEED430_R23_REPRO"
ACC8_DIR="$RESULT_ROOT/$ACC8_RUN"
python -u train.py \
  --model-family promptmr-plus --promptmr-production --promptmr-rung R2 \
  --promptmr-num-cascades 10 --promptmr-n-history 0 --promptmr-compact-fallback \
  --promptmr-train-acceleration acc8 --promptmr-mraugment off \
  --promptmr-legal-mask-family --promptmr-lr-schedule specialist_warmup1_cosine \
  --promptmr-skip-validation --precision fp32 \
  --require-cuda-device-name "NVIDIA GeForce GTX 1080" --GPU-NUM 0 \
  --batch-size 1 --num-epochs 1 --promptmr-stop-after-optimizer-steps 1158 \
  --promptmr-specialist-lr-horizon-optimizer-steps 2315 \
  --promptmr-mraugment-seed 433 --promptmr-legal-mask-seed 438 \
  --promptmr-specialist-loss-family r10_image_masked_ssim_valid_windows_mean \
  --promptmr-specialist-trainable-scope all --lr 0.00005 --seed 430 \
  --net-name "$ACC8_RUN" --data-path-train "$DATA_ROOT/train" \
  --data-path-val "$DATA_ROOT/val" --trusted-data-manifest "$MANIFEST" \
  --report-interval 50 --promptmr-vessl-model-only-import "$E49_CHECKPOINT" \
  --promptmr-vessl-model-only-import-sha256 "$C10_SHA256" \
  >"$RESULT_ROOT/acc8-specialist.log" 2>&1

checkpoint_from_terminal() {
  python - "$1" <<'PY'
import json
from pathlib import Path
import sys
print(Path(json.loads(Path(sys.argv[1]).read_text())["checkpoint"]).resolve())
PY
}
ACC4_CHECKPOINT="$(checkpoint_from_terminal "$ACC4_DIR/terminal.json")"
ACC8_CHECKPOINT="$(checkpoint_from_terminal "$ACC8_DIR/terminal.json")"
ACC4_SHA256="$(sha256sum "$ACC4_CHECKPOINT" | awk '{print $1}')"
ACC8_SHA256="$(sha256sum "$ACC8_CHECKPOINT" | awk '{print $1}')"

install -m 0644 "$REPRO_ROOT/generalist/train.py" "$PROJECT_ROOT/train.py"
install -m 0644 "$REPRO_ROOT/generalist/promptmr_production.py" \
  "$PROJECT_ROOT/utils/learning/promptmr_production.py"

NAF_DIR="$RESULT_ROOT/VESSL_POST_REFINER_R2_C10_NAF_S_E21_BBOX05_R23_REPRO"
python -u "$REPRO_ROOT/vessl_train_post_refiner.py" \
  --base-checkpoint "$E49_CHECKPOINT" --base-checkpoint-sha256 "$C10_SHA256" \
  --acc4-checkpoint "$ACC4_CHECKPOINT" --acc4-checkpoint-sha256 "$ACC4_SHA256" \
  --acc8-checkpoint "$ACC8_CHECKPOINT" --acc8-checkpoint-sha256 "$ACC8_SHA256" \
  --variant NAF_S --views identity flip_lr --epochs 21 \
  --optimizer-steps 93567 --output-dir "$NAF_DIR" \
  --train-root "$DATA_ROOT/train" --trusted-data-manifest "$MANIFEST" \
  --extra-train-root "$DATA_ROOT/val" --extra-trusted-data-manifest "$MANIFEST" \
  --loss-family winner_foreground_ssim_l1_sqrt_area_plus_official384_bbox05_v2 \
  --peak-lr 0.0001 --weight-decay 0.0001 --seed 430 \
  >"$RESULT_ROOT/naf-s.log" 2>&1

NAF_CHECKPOINT="$(python - "$NAF_DIR/receipt.json" <<'PY'
import json
from pathlib import Path
import sys
print(Path(json.loads(Path(sys.argv[1]).read_text())["checkpoint"]).resolve())
PY
)"
NAF_SHA256="$(sha256sum "$NAF_CHECKPOINT" | awk '{print $1}')"
FINAL_DIR="$RESULT_ROOT/final-r23-single-package"
mkdir -p "$FINAL_DIR"
python -u "$REPRO_ROOT/vessl_build_routed_promptmr_checkpoint.py" \
  --generalist-checkpoint "$E49_CHECKPOINT" --generalist-sha256 "$C10_SHA256" \
  --acc4-checkpoint "$ACC4_CHECKPOINT" --acc4-sha256 "$ACC4_SHA256" \
  --acc8-checkpoint "$ACC8_CHECKPOINT" --acc8-sha256 "$ACC8_SHA256" \
  --tta-views acc8_identity_flip_lr \
  --post-refiner-checkpoint "$NAF_CHECKPOINT" \
  --post-refiner-sha256 "$NAF_SHA256" \
  --output "$FINAL_DIR/best_model.pt"

test -f "$FINAL_DIR/best_model.pt"
echo "R23_REPRODUCTION_COMPLETE $FINAL_DIR/best_model.pt"
