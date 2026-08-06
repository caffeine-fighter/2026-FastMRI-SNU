#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE_ROOT="$PACKAGE_ROOT/evidence"
EVALUATION_LOG="$EVIDENCE_ROOT/official-evaluation.log"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PACKAGE_ROOT"
"$PYTHON_BIN" verify_package.py
"$PYTHON_BIN" record_official_evaluation.py start

set +e
bash recon_eval.sh 2>&1 | tee "$EVALUATION_LOG"
evaluation_return_code="${PIPESTATUS[0]}"
set -e

"$PYTHON_BIN" record_official_evaluation.py finish \
  --return-code "$evaluation_return_code"
"$PYTHON_BIN" verify_package.py --submission-ready
