#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: bash scripts/set_phase2_candidate.sh <exp_id> <checkpoint_path> <cascade> <chans> <sens_chans> [notes]"
  echo "Example:"
  echo "  bash scripts/set_phase2_candidate.sh EXP012 ../result/EXP012_varnet_c4_ch12_s4_e10/checkpoints/best_model.pt 4 12 4"
  exit 2
fi

EXP_ID="$1"
CKPT="$2"
CASCADE="$3"
CHANS="$4"
SENS_CHANS="$5"
NOTES="${6:-}"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

mkdir -p checkpoints_phase2 reports/phase2

# Link the selected checkpoint; do not copy model weights into the repo.
ln -sfn "$CKPT" checkpoints_phase2/best_model.pt

cat > reports/phase2/current_candidate.env <<EOC
EXP_ID=$EXP_ID
CHECKPOINT=$CKPT
CHECKPOINT_SYMLINK=checkpoints_phase2/best_model.pt
CASCADE=$CASCADE
CHANS=$CHANS
SENS_CHANS=$SENS_CHANS
NOTES=$NOTES
EOC

echo "Set Phase 2 candidate:"
cat reports/phase2/current_candidate.env
echo ""
echo "Symlink:"
ls -lah checkpoints_phase2/best_model.pt
