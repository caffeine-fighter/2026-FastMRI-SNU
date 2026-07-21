# FI-VarNet acc8 epochs 1–30 full-training handoff

## Authority and scope

> **LAUNCH IS NOT AUTHORIZED.**

This handoff describes the separately gated, checkpointed FI-VarNet acc8 epochs 1–30 runner. It does not authorize a CUDA launch. Authorization remains pending an exact review of the final source bytes and command, followed by a reviewed CUDA split/resume test that proves an interrupted run resumes to the same exact state as an uninterrupted run. Do not run this lane, query/initialize CUDA for it, or touch organizer H5 payloads while performing CPU review.

The executable gate is distinct from the one-step smoke gate:

- full lane: `--model-family fi-varnet-acc8 --fi-acc8-full-training`
- smoke lane: `--model-family fi-varnet-acc8 --fi-acc8-one-step-smoke`
- exactly one lane must be selected; the parser rejects neither or both
- the full namespace/output is fixed to `EXP_FI_ACC8_CKPT_BASE_E30_R1/fi-acc8-full-training`
- full checkpoints are `FULL_TRAINING_ONLY`; they set `evaluation_authorized: false` and `submission_authorized: false`

No validation, evaluation, inference, score comparison, ensemble, or submission is part of this runner. A strict-hybrid Q5 review is required later before any such work.

## Q2 feasibility PASS anchor (not full-launch authority)

The prerequisite checkpointed one-step feasibility evidence is:

- review record: `/root/result/FI_ACC8_TRAINING_REVIEWS/FI_ACC8_CKPT_ONE_STEP_ACTUAL_R7_PASS.json`
- record schema/status: `fi_acc8_checkpointed_one_step_actual_pass_v1` / `PASS`
- scope: `SMOKE_ONLY_NONRESUMABLE`
- smoke checkpoint: `/root/result/LOCAL_FI_ACC8_CKPT_SMOKE_R1/fi-acc8-training-fit-smoke/checkpoint-step-000001.pt`
- smoke checkpoint SHA-256: `12d57839f2e34039f0e292d09fad810d1193a7c6aa82d33f5e9177cf495dc9f3`
- report SHA-256: `b817f1be50085ddf07a8caf8ecca4ec14b70050c989ae4949601ab88966a6a5b`
- selected GPU: `GPU-3073d3e5-383c-775f-faca-904c38057c94`, `NVIDIA GeForce GTX 1080`, 8192 MiB
- observed one-step GPU use: 6157 MiB
- finite loss: `0.2901897430419922`
- gradient-bearing/changed parameters: `744` / `744`

That PASS establishes only one-step checkpointed FP32 training fit. Its own record says `training_authorized: false`; it is not a resumable full checkpoint and must never be supplied to this runner.

## Frozen full recipe

| Field | Exact value |
|---|---:|
| recipe schema | `fi-varnet-acc8-checkpointed-full-training-v1` |
| namespace | `EXP_FI_ACC8_CKPT_BASE_E30_R1` |
| scope | `FULL_TRAINING_ONLY` |
| initialization | scratch; no external learned state |
| seed | `431` |
| batch size | `1` |
| precision/autocast | FP32 / disabled |
| optimizer | `torch.optim.AdamW` |
| base LR | `3e-4` |
| weight decay | `0.0` |
| loss | pinned upstream `fastmri.losses.SSIMLoss` |
| gradient clipping | disabled |
| feature/image cascades | `12` / `12` |
| cascade channels/pools | `18` / `4` |
| sensitivity channels/pools | `8` / `4` |
| activation checkpointing | all 12 + 12 cascades; non-reentrant; preserve RNG; state dict unchanged |
| acceleration | acc8 only |
| selected inventory | exactly 85 files / 2315 slices per epoch |
| base run | epochs 1–30 / exactly 69,450 optimizer and scheduler steps |
| schedule horizon | 40 epochs / 92,600 steps |
| ramp | 3,704 steps |
| cosine-decay start | step 46,300 |
| checkpoint cadence | every accepted file transaction |
| status interval | no more than 300 seconds, plus every checkpoint and completion boundary |

The nominal schedule is not smoke-primed. Constructing the frozen `LambdaLR` sets LambdaLR step 0 to multiplier `0.0`, so the first full-training optimizer update applies LR `0.0`. Each optimizer update is followed by exactly one scheduler step. At global step `n`, the optimizer LR must exactly equal `3e-4 * fi_lr_multiplier(n)` before the update and the step-`n+1` value after it.

## Deterministic execution contract

All pre-CUDA gates complete before the run root is reserved and before device selection. The runner then enforces this exact contract before any CUDA selection:

```json
{"schema":"fi-acc8-determinism-v1","cublas_workspace_config":":4096:8","deterministic_algorithms":true,"cudnn_deterministic":true,"cudnn_benchmark":false}
```

`CUBLAS_WORKSPACE_CONFIG` must be absent (the runner sets `:4096:8`) or already exactly `:4096:8`; any other value is a blocker. PyTorch deterministic algorithms are enabled, cuDNN deterministic mode is enabled, and cuDNN benchmark mode is disabled. File and slice permutations are deterministic functions of seed, epoch, and basename. Python, NumPy, CPU Torch, and selected-device CUDA RNG state are checkpointed and restored.

## Same-FD data transaction and durable rollback boundary

Each selected file is one transaction. The root chain, `kspace/`, `image/`, and both H5 leaves are opened nofollow and descriptor-relatively. The runner verifies inventory identity and SHA, opens H5 from those same retained file descriptors, consumes every slice exactly once in deterministic order, closes H5, then re-hashes and re-stats those same retained file descriptors. A transaction is accepted only after the complete pair passes post-consumption verification.

Model, optimizer, scheduler, sampler, metrics, RNG, bindings, provenance, and the accepted transaction list are published only after that acceptance. If a process, CUDA operation, H5 verification, serialization, pointer switch, or status write fails mid-file, that file has no resume authority. Recovery starts at the prior durable verified-file boundary; the file is replayed from its beginning. This deliberately rolls back any non-durable in-memory work and prevents skipped or partially accepted files.

## Resource preflight

Resource checks use live measurements; no current free-memory or free-space value is hardcoded.

RAM is conservatively required to cover:

- the maximum selected k-space/image pair bytes in RAM;
- one full-checkpoint-sized CPU model allowance;
- one full-checkpoint-sized CPU optimizer allowance;
- one full-checkpoint-sized detached checkpoint snapshot allowance;
- one full-checkpoint-sized serialization allowance; and
- a fixed 512 MiB margin.

The checkpoint-size bound is 1,479,000,000 bytes. Thus:

```text
required RAM = max pair bytes
             + 4 * 1,479,000,000
             + 512 MiB
```

Disk preflight is measured on the nearest existing ancestor of the **output filesystem**, never on `/root/Data/train`. It covers 32 retained checkpoint generations, one additional staging checkpoint, and a 5% margin over those 33 checkpoint-size bounds:

```text
retained = 32 * 1,479,000,000
staging  =  1 * 1,479,000,000
margin   = ceil(0.05 * (retained + staging))
required disk = retained + staging + margin
```

Falling one byte below either computed threshold fails before output reservation and CUDA.

## Checkpoint publication, retention, and cleanup

Every accepted file boundary serializes an immutable generation containing exactly regular `checkpoint.pt` and `metadata.json`. The generation is fsynced and published before `checkpoint-current.json` atomically switches authority. Pointer failure leaves the prior pointer authoritative; a later successful publication prunes the orphan.

Retention is at most 30 unique immutable epoch-end generations plus latest and previous, bounded by the 32 retained checkpoint generations used in resource preflight. The one pointer validator runs both when `checkpoint-current.json` is loaded and on every newly constructed pointer before atomic replacement. It requires the exact versioned top-level, entry, and verified-file sampler schemas; lowercase `generation-` plus 32-hex names and 64-hex checkpoint digests; distinct latest/previous names; and at most 30 epoch entries. Epoch entries must be strictly increasing unique cursors for next epochs 2–31 with `file_cursor: 0` and `global_step: (epoch - 1) * 2315`, with no duplicate epoch-generation names. Latest or previous may be the same exact entry as an epoch generation (the normal epoch-end case), but the distinct union across all three pointer fields must never exceed 32. Thus malformed, oversized, duplicate, out-of-order, or impossible preexisting lineage fails before resume and before a new generation is created; newly constructed lineage fails before pointer publication. Publishing a second epoch-end generation for an already-retained completed-epoch cursor is rejected before generation creation or pointer mutation. Pruning is rename-first: an unretained `generation-*` directory is moved to a random private `.retired-*` name under the checkpoint lock. Only then are the two known regular files opened with `O_NOFOLLOW`, retained by FD/inode identity, renamed individually to private `.delete-*` aliases, rebound to the same retained inode, unlinked alias-relatively, and directory-fsynced. Symlinks, subdirectories, missing files, or unknown entries fail closed and are preserved for review.

Atomic status/pointer replacement also retains the staged file FD. On failure, cleanup renames the staged name to a private cleanup alias only while it is still the retained inode, reopens and verifies the alias is that inode, then unlinks it. Any racing replacement or unverifiable alias is preserved and the original exception is annotated; cleanup never unlinks a stale pathname.

## Exact CPU-staged resume

Resume is allowed only into an existing run root containing `INCOMPLETE` and not `COMPLETE`. Fresh launch treats every existing root—including a stranded `INCOMPLETE` root with no checkpoint—as a collision and never deletes or recovers it automatically.

Before source inventory, GPU probing, resource checks, output reservation, or CUDA, the resume path must be exactly:

```text
<same absolute run root>/checkpoint-generations/generation-*/checkpoint.pt
```

The generation path, required SHA-256, and generation-metadata sampler must match **only** `latest` or `previous` in that root's durable `checkpoint-current.json`. Entries retained in `epoch_generations` are immutable Q5 evaluation/forensic inputs and are never Q3 resume sources; arbitrary direct generation paths are also rejected. Both `--resume-checkpoint` and `--resume-checkpoint-sha256` are mandatory. The reference check occurs before source inventory, GPU probing, output mutation, or CUDA. The restricted checkpoint load and all schema, source/data/recipe/GPU binding, cursor, step, LR, transaction, and finite-tensor checks occur in fresh CPU model/optimizer/scheduler objects. Only after successful CPU validation does the runner move the model and every optimizer tensor to the already selected device and restore RNG state. Inexact resume, LR override, external state, cross-root checkpoints, and rollback branching/truncation are forbidden.

## Non-authoritative production status

`status.json` is validated before atomic publication and always contains exactly:

- `authoritative: false`, schema, phase, PID, and update Unix timestamp;
- selected GPU UUID, name, and index;
- current allocated, reserved, and peak VRAM bytes read from the explicitly selected CUDA device (never from a pre-selection `torch.cuda.is_available()` probe);
- epoch, file cursor, file, slice, and global step;
- current nominal LR, finite exponential moving loss, session throughput in steps/second, and finite ETA;
- last checkpoint absolute path and SHA-256 as a paired value;
- exact command argv; and
- the exact deterministic contract above.

It is refreshed at least every 300 seconds during a long file, after every checkpoint boundary, and at completion. It is operational telemetry only. It cannot authorize resume, completion, evaluation, or submission; checkpoint pointer + generation SHA establish resume authority, and `COMPLETE`/summary establish training completion.

## Proposed commands — review only, do not execute

Fresh epochs 1–30 command from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python train.py \
  --model-family fi-varnet-acc8 \
  --fi-acc8-full-training \
  --expected-gpu-uuid GPU-3073d3e5-383c-775f-faca-904c38057c94 \
  --GPU-NUM 0 \
  --net-name EXP_FI_ACC8_CKPT_BASE_E30_R1 \
  --data-path-train /root/Data/train
```

Exact resume command (the reviewer must substitute the pointer `latest` or `previous` absolute generation path and its exact lowercase SHA-256 from the same run root):

```bash
PYTHONDONTWRITEBYTECODE=1 python train.py \
  --model-family fi-varnet-acc8 \
  --fi-acc8-full-training \
  --expected-gpu-uuid GPU-3073d3e5-383c-775f-faca-904c38057c94 \
  --GPU-NUM 0 \
  --net-name EXP_FI_ACC8_CKPT_BASE_E30_R1 \
  --data-path-train /root/Data/train \
  --resume-checkpoint /root/result/EXP_FI_ACC8_CKPT_BASE_E30_R1/fi-acc8-full-training/checkpoint-generations/generation-REVIEWED/checkpoint.pt \
  --resume-checkpoint-sha256 REVIEWED_64_LOWERCASE_HEX
```

Do not issue either command yet. Exact final-byte review and the separately authorized CUDA split/resume equivalence exercise must pass first. Even a completed epochs 1–30 run authorizes no validation/evaluation/submission; strict-hybrid Q5 remains a later, separate gate.

## CPU-only closeout

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_fi_acc8_full_training.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_fi_acc8_training.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
PYTHONPYCACHEPREFIX="$(mktemp -d /tmp/fi-acc8-pycache-XXXXXX)" \
  python -m py_compile train.py \
  utils/learning/fi_acc8_training.py \
  utils/learning/fi_acc8_full_training.py \
  utils/model/fi_varnet_adapter.py \
  tests/test_fi_acc8_training.py \
  tests/test_fi_acc8_full_training.py
PYTHONDONTWRITEBYTECODE=1 python scripts/check_submission.py
git diff --check
```

These checks are CPU-only and provide no live CUDA, VRAM, organizer-data, launch, evaluation, or submission evidence.
