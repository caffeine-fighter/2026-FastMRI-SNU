#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$PACKAGE_ROOT/project"
MODEL_SOURCE="$PACKAGE_ROOT/best_model.pt"
RESULT_ROOT="$PACKAGE_ROOT/result"
MODEL_DESTINATION="$RESULT_ROOT/test_Varnet/checkpoints/best_model.pt"
LEADERBOARD_DATA_ROOT="${LEADERBOARD_DATA_ROOT:-/root/Data/leaderboard}"

python "$PACKAGE_ROOT/verify_package.py"
test -f "$PROJECT_ROOT/recon_eval.py"
test -d "$LEADERBOARD_DATA_ROOT/acc4"
test -d "$LEADERBOARD_DATA_ROOT/acc8"

mkdir -p "$(dirname -- "$MODEL_DESTINATION")"
if ! ln -f -- "$MODEL_SOURCE" "$MODEL_DESTINATION" 2>/dev/null; then
  cp -f -- "$MODEL_SOURCE" "$MODEL_DESTINATION"
fi

cd "$PROJECT_ROOT"
exec python recon_eval.py \
  -n test_Varnet \
  -p "$LEADERBOARD_DATA_ROOT" \
  --cascade 10 \
  --chans 48 \
  --sens_chans 24
