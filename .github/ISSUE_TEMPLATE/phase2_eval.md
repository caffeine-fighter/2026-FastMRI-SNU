---
name: Phase 2 official eval
description: Track an official Phase 2 recon_eval wrapper run
title: "[Phase2 Eval] <candidate>"
labels: ["type:submission", "area:phase2", "area:recon_eval", "status:needs-review"]
assignees: []
---

## Candidate

- Candidate name:
- Source experiment:
- Config:

## Checkpoint

- Checkpoint path:
- Exists? yes/no
- Symlink/candidate setup command:

```bash
bash scripts/set_phase2_candidate.sh <candidate> <checkpoint> <cascade> <chans> <sens_chans> '<notes>'
```

## Preflight

```bash
bash scripts/phase2_preflight.sh
```

- Preflight result:

## Official recon_eval log

- Run tag:
- Log path:
- Score JSON path:

```bash
bash scripts/run_recon_eval_once.sh <candidate_tag>
```

## Scores

- `SSIM_full`:
- `SSIM_bbox`:
- `quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox`:
- `ms/slice`:
- `time_score`:
- `total_score`:

## 30-repeat status

- One-shot valid? yes/no
- Repeat-30 required? yes/no
- Repeat command:

```bash
bash scripts/repeat_recon_eval.sh <candidate_tag> 30
```

- Minimum valid `ms/slice`:
- Best total score:
- Summary path:

## Acceptance criteria

- [ ] Official entrypoint was `bash recon_eval.sh` via wrapper.
- [ ] `recon_eval.py` was unmodified.
- [ ] Mounted `Data` was read-only.
- [ ] Score fields were parsed and reviewed.
- [ ] Repeat-30 completed if this is the final selected candidate.

## Safety constraints

- Do not use image fields, bbox annotations, or given GRAPPA during inference.
- Do not modify mounted `Data`.
- Do not modify `recon_eval.py`.
- Do not stage checkpoints, `.h5`, results, `.env`, or secrets.
