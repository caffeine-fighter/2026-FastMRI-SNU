#!/usr/bin/env bash
set -u -o pipefail
DRY_RUN="${DRY_RUN:-1}"

# Generated from docs/github_issue_plan.md. Default is dry-run.
quote_cmd() { printf "%q " "$@"; printf "\n"; }
run_or_echo() {
  if [ "$DRY_RUN" = "1" ]; then
    printf "+ "; quote_cmd "$@"
  else
    "$@"
  fi
}

create_or_update_label() {
  local name="$1" color="$2" description="$3"
  if [ "$DRY_RUN" = "1" ]; then
    printf "# label: %s\n" "$name"
    printf "+ "; quote_cmd gh label create "$name" --color "$color" --description "$description"
    printf "+ # if already exists, run: "; quote_cmd gh label edit "$name" --color "$color" --description "$description"
    return 0
  fi
  if gh label view "$name" >/dev/null 2>&1; then
    gh label edit "$name" --color "$color" --description "$description"
  else
    gh label create "$name" --color "$color" --description "$description" || gh label edit "$name" --color "$color" --description "$description"
  fi
}

LABELS=(
  'type:experiment|5319e7|Experiment tracking and model-run decisions'
  'type:automation|1d76db|Automation, scripts, wrappers, and tooling'
  'type:bug|d73a4a|Bug, failure, crash, or unexpected behavior'
  'type:docs|0075ca|Documentation and communication work'
  'type:submission|fbca04|Final/official submission preparation'
  'area:vessl|0e8a16|VESSL official training/evaluation environment'
  'area:desktop|a2eeef|Desktop WSL local-probe environment'
  'area:phase2|5319e7|Phase 2 official scoring/evaluation'
  'area:metrics|c5def5|Metrics, scoring, plots, and summaries'
  'area:recon_eval|bfdadc|Official recon_eval wrapper/evaluator'
  'priority:P0|b60205|Critical path / must-do'
  'priority:P1|d93f0b|High priority follow-up'
  'priority:P2|fbca04|Useful but not blocking'
  'status:blocked|d73a4a|Blocked by dependency or approval'
  'status:running|0e8a16|Currently running or in progress'
  'status:needs-review|fbca04|Needs review before action/merge'
  'status:done|ededed|Completed or resolved'
  'risk:oom|b60205|Memory / OOM risk'
  'risk:timing|d93f0b|Runtime or ms/slice risk'
  'risk:rules|b60205|Competition rule / safety risk'
)

for spec in "${LABELS[@]}"; do
  IFS='|' read -r name color description <<<"$spec"
  create_or_update_label "$name" "$color" "$description"
done
