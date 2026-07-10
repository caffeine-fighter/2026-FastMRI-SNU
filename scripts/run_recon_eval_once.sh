#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-phase2_eval}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="reports/phase2/${TAG}_${TS}"
LOG="$OUT_DIR/eval_run.log"
SCORE_JSON="$OUT_DIR/score.json"
SCORE_CSV="reports/phase2/scoreboard.csv"

mkdir -p "$OUT_DIR"

echo "=== output dir ==="
echo "$OUT_DIR"

echo "=== candidate ===" | tee "$OUT_DIR/candidate.txt"
cat reports/phase2/current_candidate.env 2>/dev/null | tee -a "$OUT_DIR/candidate.txt" || true

echo ""
echo "=== preflight ==="
bash scripts/phase2_preflight.sh | tee "$OUT_DIR/preflight.log"

echo ""
echo "=== running official recon_eval.sh ==="
set -o pipefail

# This script intentionally wraps only the official bash recon_eval.sh entrypoint.
# stdbuf keeps logs streaming under tee and does not change model behavior.
stdbuf -oL -eL bash recon_eval.sh 2>&1 | tee "$LOG"

echo ""
echo "=== parsing score ==="
python scripts/phase2_score.py \
  --log "$LOG" \
  --tag "$TAG" \
  --out-json "$SCORE_JSON" \
  --out-csv "$SCORE_CSV" | tee "$OUT_DIR/score.txt"

echo ""
echo "done: $OUT_DIR"
