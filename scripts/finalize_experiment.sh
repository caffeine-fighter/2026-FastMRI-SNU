#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/finalize_experiment.sh [--allow-skipped] exp_id exp_name seed config notes

Example:
  bash scripts/finalize_experiment.sh EXP001 EXP001_baseline_varnet_c1_ch9_s4_e5 430 none "notes"
EOF
}

allow_skipped=0
if [[ "$#" -gt 0 && "$1" == "--allow-skipped" ]]; then
  allow_skipped=1
  shift
fi

if [[ "$#" -gt 0 && "${!#}" == "--allow-skipped" ]]; then
  allow_skipped=1
  set -- "${@:1:$(($# - 1))}"
fi

if [[ "$#" -ne 5 ]]; then
  usage >&2
  exit 2
fi

exp_id="$1"
exp_name="$2"
seed="$3"
config="$4"
notes="$5"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

result_dir="../result/${exp_name}"
metrics_dir="${result_dir}/metrics"
metrics_csv="${metrics_dir}/metrics.csv"
skipped_json="${metrics_dir}/skipped.json"
evaluate_stdout="${metrics_dir}/evaluate_stdout.txt"

mkdir -p "${metrics_dir}"

evaluate_cmd=(
  python scripts/evaluate_val.py
  --exp-name "${exp_name}"
  --target-dir /root/Data/val/image
  --recon-dir "${result_dir}/reconstructions_val"
  --out-dir "${metrics_dir}"
)

printf 'Running evaluation command:\n  '
printf '%q ' "${evaluate_cmd[@]}"
printf '\n'
"${evaluate_cmd[@]}" | tee "${evaluate_stdout}"

if [[ ! -f "${metrics_csv}" ]]; then
  echo "ERROR: metrics.csv not found after evaluation: ${metrics_csv}" >&2
  exit 1
fi

if [[ ! -f "${skipped_json}" ]]; then
  echo "ERROR: skipped.json not found after evaluation: ${skipped_json}" >&2
  exit 1
fi

printf '\n===== %s =====\n' "${metrics_csv}"
cat "${metrics_csv}"
printf '\n===== %s =====\n' "${skipped_json}"
cat "${skipped_json}"
printf '\n'

if [[ "${allow_skipped}" -ne 1 ]]; then
  python -c 'import json, sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data:
    print(f"ERROR: skipped.json is not empty: {path}", file=sys.stderr)
    print("Pass --allow-skipped to finalize anyway.", file=sys.stderr)
    raise SystemExit(1)
' "${skipped_json}"
fi

plot_cmd=(
  python scripts/plot_loss.py
  --exp-name "${exp_name}"
)

printf 'Running loss plot command:\n  '
printf '%q ' "${plot_cmd[@]}"
printf '\n'
"${plot_cmd[@]}"

recorded_command="$(python -c 'import csv, sys
from pathlib import Path

exp_id, exp_name = sys.argv[1], sys.argv[2]
queue_path = Path("experiments/experiment_queue.csv")
if not queue_path.exists():
    raise SystemExit(0)

with queue_path.open(newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row.get("exp_id") == exp_id or row.get("exp_name") == exp_name:
            print(row.get("command", ""))
            break
' "${exp_id}" "${exp_name}")"

record_cmd=(
  python scripts/record_experiment.py
  --exp-id "${exp_id}"
  --exp-name "${exp_name}"
  --status done
  --seed "${seed}"
  --config "${config}"
  --command "${recorded_command}"
  --notes "${notes}"
)

printf 'Running record command:\n  '
printf '%q ' "${record_cmd[@]}"
printf '\n'
"${record_cmd[@]}"
