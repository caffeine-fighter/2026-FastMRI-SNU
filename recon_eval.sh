#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Final submitted EXP030 architecture. The ignored candidate file is only an
# optional local override; clean submission checkouts use these defaults.
CASCADE="${CASCADE:-4}"
CHANS="${CHANS:-12}"
SENS_CHANS="${SENS_CHANS:-8}"
CANDIDATE_ENV="$REPO_DIR/reports/phase2/current_candidate.env"

if [ -f "$CANDIDATE_ENV" ]; then
  while IFS='=' read -r key value; do
    case "$key" in
      CHECKPOINT) CHECKPOINT="$value" ;;
      CASCADE) CASCADE="$value" ;;
      CHANS) CHANS="$value" ;;
      SENS_CHANS) SENS_CHANS="$value" ;;
    esac
  done < "$CANDIDATE_ENV"
fi

for name in CASCADE CHANS SENS_CHANS; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $name must be a positive integer, got: $value" >&2
    exit 1
  fi
done

MODEL_PATH="$REPO_DIR/../result/test_Varnet/checkpoints/best_model.pt"
if [ -n "${CHECKPOINT:-}" ]; then
  if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: candidate checkpoint not found: $CHECKPOINT" >&2
    exit 1
  fi
  CHECKPOINT="$(realpath -- "$CHECKPOINT")"
  mkdir -p "$(dirname -- "$MODEL_PATH")"
  ln -sfn "$CHECKPOINT" "$MODEL_PATH"
elif [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: model weight not found: $MODEL_PATH" >&2
  echo "Place the submitted best_model.pt at that path before evaluation." >&2
  exit 1
fi

python recon_eval.py \
  -n 'test_Varnet' \
  -p '/root/Data/leaderboard' \
  --cascade "$CASCADE" \
  --chans "$CHANS" \
  --sens_chans "$SENS_CHANS"
