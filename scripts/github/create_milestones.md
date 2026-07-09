# GitHub Milestone Creation Notes

GitHub CLI milestone support varies by version. Do not assume `gh` has a first-class `milestone create` command.

## Milestones to create

### Phase 2 official evaluation

Wrapper/preflight readiness, official one-shot runs, 30-repeat timing, and total-score review.

### Model selection

EXP012/EXP030 comparison, local-probe interpretation, and any follow-up candidate decisions.

### Final submission package

Final README, checklist, score summary, allowed artifacts, and explanation materials.

## Option A — GitHub web UI

1. Open the repository on GitHub.
2. Go to **Issues → Milestones → New milestone**.
3. Create each milestone above with the listed description.
4. Leave due dates blank unless the submission schedule is known.

## Option B — `gh api` if authenticated

Check authentication first:

```bash
gh auth status
```

Then create milestones idempotently by checking existing milestones first:

```bash
for title in "Phase 2 official evaluation" "Model selection" "Final submission package"; do
  gh api repos/:owner/:repo/milestones --jq ".[ ] | select(.title == "$title") | .number" | grep -q . && {
    echo "milestone exists: $title"
    continue
  }
  case "$title" in
    "Phase 2 official evaluation") desc="Wrapper/preflight readiness, official one-shot runs, 30-repeat timing, and total-score review." ;;
    "Model selection") desc="EXP012/EXP030 comparison, local-probe interpretation, and follow-up candidate decisions." ;;
    "Final submission package") desc="Final README, checklist, score summary, allowed artifacts, and explanation materials." ;;
  esac
  gh api repos/:owner/:repo/milestones -f title="$title" -f description="$desc"
done
```

## Safety

- Milestone creation has no code/data side effects.
- Create labels before issues so issue creation can attach the planned labels cleanly.
- Run issue scripts with `DRY_RUN=1` first and inspect output before `DRY_RUN=0`.
