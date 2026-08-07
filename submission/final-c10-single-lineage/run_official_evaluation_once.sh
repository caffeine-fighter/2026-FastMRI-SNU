#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE_ROOT="$PACKAGE_ROOT/evidence"
EVALUATION_LOG="$EVIDENCE_ROOT/official-evaluation.log"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${1:-${DATA_ROOT:-$PACKAGE_ROOT/data/leaderboard}}"
OUT_DIR="${2:-${OUT_DIR:-$PACKAGE_ROOT/../fastmri_eval_output}}"

if ! test -d "$DATA_ROOT/acc4" || ! test -d "$DATA_ROOT/acc8"; then
  echo "usage: bash run_official_evaluation_once.sh DATA_ROOT [OUT_DIR]" >&2
  exit 2
fi

cd "$PACKAGE_ROOT"
"$PYTHON_BIN" verify_package.py
"$PYTHON_BIN" record_official_evaluation.py start

set +e
bash recon_eval.sh "$DATA_ROOT" "$OUT_DIR" 2>&1 | tee "$EVALUATION_LOG"
evaluation_return_code="${PIPESTATUS[0]}"
set -e

"$PYTHON_BIN" record_official_evaluation.py finish \
  --return-code "$evaluation_return_code"
"$PYTHON_BIN" verify_package.py --submission-ready
