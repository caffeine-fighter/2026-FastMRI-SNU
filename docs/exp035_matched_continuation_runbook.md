# EXP035 matched continuation runbook

> Prepared on 2026-07-15 without starting or probing a GPU. Training remains separately approval-gated.

## Question

EXP035 epoch 30 is the global retained-epoch winner, and its last five epochs improved overall despite non-monotonic noise. The next bounded experiment must separate an **extra-epoch effect** from a **lower-learning-rate effect**.

Start both LOCAL arms from the same immutable VESSL checkpoint:

- generation: `3e8af14268a64d67a308ebe30484ddf2`
- SHA-256: `dc6e034f18df2a7872c416d4dccb4bb00e6e5b41fb89e438a86682db3097ffb7`
- stored epoch: `30`
- protected LOCAL quality: `0.9199788092310326`

## Preregistered arms

| Field | Fixed-LR control | Lower-LR candidate |
|---|---|---|
| Name | `LOCAL_EXP035_E30_TO_E35_ADAM_LR1E3_SEED430` | `LOCAL_EXP035_E30_TO_E35_ADAM_LR3E4_SEED430` |
| Optimizer | Adam | Adam |
| Resume LR | `0.001` | `0.0003` |
| Total epoch | 35 | 35 |
| New retained epochs | 31–35 | 31–35 |

Architecture `c8/ch12/s8`, objective, data, batch 1, seed 430, checkpoint, RNG restoration, evaluator, and retention policy remain fixed. Do not use `--allow-inexact-resume` unless a separate review proves exact resume impossible and explicitly changes the experiment question.

## CPU-only preparation

When the immutable checkpoint has been transferred to the LOCAL machine, run the dry-run planner before checking GPU availability:

```bash
python scripts/plan_exp035_continuations.py \
  --checkpoint /absolute/path/to/.checkpoint-generation-3e8af14268a64d67a308ebe30484ddf2-model.pt \
  --result-root ../result \
  --train-dir /root/Data/train \
  --val-dir /root/Data/val \
  --require-data
```

The planner verifies the regular-file path, immutable generation, SHA-256, and collision-free arm output paths. It prints commands only and never imports torch, initializes CUDA, or starts training.

Before a later launch, separately record `nvidia-smi`, running GPU processes, repository commit, source checkpoint hash, data mount counts, and free result paths. Run the two arms sequentially, not concurrently.

## Selection

Independently score epochs 31–35 for both arms with exact `30 volumes / 791 slices / 161 boxes`, `skipped=[]`, unknown 0, and finite outputs.

- Promote the lower-LR direction only if it beats the fixed-LR control by at least `0.0005` equal-acc quality and all four protected cells remain healthy.
- If both arms improve similarly, attribute the result to extra epochs rather than LR. Consider at most one additional five-epoch block on the better arm; do not start it automatically.
- If both plateau, stop vanilla continuation and move the GPU budget to the upstream model-family feasibility winner.
- A gain below `0.0005` is marginal and does not authorize an official run.

No official evaluation or VESSL reproduction is automatic.
