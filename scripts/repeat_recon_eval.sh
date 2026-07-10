#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-phase2_repeat}"
N="${2:-3}"

ROOT_DIR="reports/phase2/repeat_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT_DIR"

echo "Repeat recon_eval.sh"
echo "TAG=$TAG"
echo "N=$N"
echo "ROOT_DIR=$ROOT_DIR"

for i in $(seq 1 "$N"); do
  RUN_TAG="${TAG}_run$(printf "%02d" "$i")"
  echo ""
  echo "=============================="
  echo "Run $i / $N : $RUN_TAG"
  echo "=============================="

  bash scripts/run_recon_eval_once.sh "$RUN_TAG"

  LATEST=$(ls -1dt reports/phase2/${RUN_TAG}_* | head -1)
  cp -r "$LATEST" "$ROOT_DIR/"
done

python - <<PY
import csv
import json
from pathlib import Path

root = Path("$ROOT_DIR")
score_files = sorted(root.glob("*/score.json"))

if not score_files:
    raise SystemExit("No score.json files found")

rows = []
for p in score_files:
    data = json.loads(p.read_text(encoding="utf-8"))
    data["run_dir"] = str(p.parent)
    rows.append(data)

best_by_time = min(rows, key=lambda x: (float(x["time_ms_per_slice"]), -float(x["total_score"])))
best_by_total_score = max(rows, key=lambda x: (float(x["total_score"]), -float(x["time_ms_per_slice"])))

fields = sorted(set().union(*(r.keys() for r in rows)))

csv_path = root / "repeat_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "num_runs": len(rows),
    "best_by_minimum_ms_per_slice": best_by_time,
    "best_by_total_score": best_by_total_score,
}

json_path = root / "repeat_summary.json"
json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=== repeat summary ===")
print(f"runs: {len(rows)}")
print("")
print("best_by_minimum_ms_per_slice:")
for k in ["tag", "ssim_full", "ssim_bbox", "quality_score", "time_ms_per_slice", "time_score", "total_score", "run_dir"]:
    print(f"  {k}: {best_by_time.get(k)}")
print("")
print("best_by_total_score:")
for k in ["tag", "ssim_full", "ssim_bbox", "quality_score", "time_ms_per_slice", "time_score", "total_score", "run_dir"]:
    print(f"  {k}: {best_by_total_score.get(k)}")
print("")
print(f"saved: {csv_path}")
print(f"saved: {json_path}")
PY
