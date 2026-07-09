---
name: Task
description: Track documentation, automation, review, or submission work
title: "[Task] <short description>"
labels: ["type:automation", "status:needs-review"]
assignees: []
---

## Goal

- What should be accomplished?

## Files / scope

- Files expected to change:
- Files that must not change:

## Validation commands

```bash
python scripts/check_submission.py
# add any task-specific validation commands
```

## Done criteria

- [ ] Scope completed.
- [ ] Validation commands run and outputs recorded.
- [ ] Diff reviewed.
- [ ] No forbidden artifacts staged.

## Blocking dependencies

- Dependency 1:
- Dependency 2:

## Safety constraints

- Do not stage/commit `Data/`, `data/`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`, result folders, `.env`, or secrets.
- Do not run training or `recon_eval.sh` unless explicitly approved.
- Do not modify protected model/evaluator files unless the issue explicitly requires it and approval is recorded.
