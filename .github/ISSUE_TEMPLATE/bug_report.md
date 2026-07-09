---
name: Bug report
description: Report a failure, crash, metric issue, or automation problem
title: "[Bug] <short description>"
labels: ["type:bug", "status:needs-review"]
assignees: []
---

## Environment

- Machine: VESSL / Desktop WSL / Laptop
- Branch:
- Commit:
- Python/CUDA/PyTorch versions if relevant:
- GPU if relevant:

## Command

```bash
# paste exact command that failed
```

## Error / actual behavior

```text
paste error output here
```

## Logs

- Relevant log path:
- Last relevant lines:

```text
paste log excerpt here
```

## Expected behavior

- What should have happened?

## Suspected cause

- Initial hypothesis:
- Related files:

## Acceptance criteria

- [ ] Root cause identified.
- [ ] Fix or workaround documented.
- [ ] Validation command passes.
- [ ] No protected files modified without approval.

## Safety constraints

- Do not commit data/checkpoints/results/secrets.
- Do not modify `recon_eval.py` unless explicitly approved and verified byte-identical to upstream.
- Do not run `recon_eval.sh` or training as part of bug triage unless separately approved.
