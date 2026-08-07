#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SOURCE="$PACKAGE_ROOT/project"
EVIDENCE_ROOT="$PACKAGE_ROOT/evidence"
MODEL_SOURCE="$PACKAGE_ROOT/best_model.pt"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${1:-${DATA_ROOT:-$PACKAGE_ROOT/data/leaderboard}}"
OUT_DIR="${2:-${OUT_DIR:-$PACKAGE_ROOT/../fastmri_eval_output}}"
RUNTIME_ROOT="$OUT_DIR/runtime"
PROJECT_ROOT="$RUNTIME_ROOT/project"
RESULT_ROOT="$RUNTIME_ROOT/result"
MODEL_DESTINATION="$RESULT_ROOT/test_Varnet/checkpoints/best_model.pt"

if test -f "$EVIDENCE_ROOT/official-evaluation-start.json" && \
  ! test -f "$EVIDENCE_ROOT/official-evaluation-receipt.json"; then
  "$PYTHON_BIN" "$PACKAGE_ROOT/verify_package.py" --evaluation-in-progress
else
  "$PYTHON_BIN" "$PACKAGE_ROOT/verify_package.py"
fi
test -f "$PROJECT_SOURCE/recon_eval.py"
if ! test -d "$DATA_ROOT/acc4" || ! test -d "$DATA_ROOT/acc8"; then
  echo "usage: bash recon_eval.sh DATA_ROOT [OUT_DIR]" >&2
  echo "DATA_ROOT must contain acc4/ and acc8/." >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT"
cp -a -- "$PROJECT_SOURCE/." "$PROJECT_ROOT/"
mkdir -p "$(dirname -- "$MODEL_DESTINATION")"
if ! ln -f -- "$MODEL_SOURCE" "$MODEL_DESTINATION" 2>/dev/null; then
  cp -f -- "$MODEL_SOURCE" "$MODEL_DESTINATION"
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" recon_eval.py \
  -n test_Varnet \
  -p "$DATA_ROOT" \
  --cascade 10 \
  --chans 48 \
  --sens_chans 24
