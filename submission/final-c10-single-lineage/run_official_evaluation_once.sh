#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE_ROOT="$PACKAGE_ROOT/evidence"
EVALUATION_LOG="$EVIDENCE_ROOT/official-evaluation.log"

cd "$PACKAGE_ROOT"
python verify_package.py
python record_official_evaluation.py start

set +e
bash recon_eval.sh 2>&1 | tee "$EVALUATION_LOG"
evaluation_return_code="${PIPESTATUS[0]}"
set -e

python record_official_evaluation.py finish \
  --return-code "$evaluation_return_code"
python verify_package.py --submission-ready
