#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_vessl_experiment.sh exp_name epochs lr cascade chans sens_chans seed

Example:
  bash scripts/run_vessl_experiment.sh EXP010_varnet_c2_ch9_s4_e10 10 0.001 2 9 4 430
EOF
}

if [[ "$#" -ne 7 ]]; then
  usage >&2
  exit 2
fi

exp_name="$1"
epochs="$2"
lr="$3"
cascade="$4"
chans="$5"
sens_chans="$6"
seed="$7"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

train_path="/root/Data/train/"
val_path="/root/Data/val/"
result_dir="../result/${exp_name}"
log_dir="${result_dir}/logs"

mkdir -p "${log_dir}"

python scripts/print_run_context.py > "${log_dir}/run_context.txt"

cmd=(
  python train.py
  -b 1
  -e "${epochs}"
  -l "${lr}"
  -r 10
  -n "${exp_name}"
  -t "${train_path}"
  -v "${val_path}"
  --cascade "${cascade}"
  --chans "${chans}"
  --sens_chans "${sens_chans}"
  --seed "${seed}"
)

printf 'Running command:\n  '
printf '%q ' "${cmd[@]}"
printf '\n'

"${cmd[@]}" | tee "${log_dir}/train_stdout.log"
