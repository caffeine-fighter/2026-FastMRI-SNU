# FI-VarNet acc8 one-step training-fit smoke handoff

## Status and scope

This integration is a **one-optimizer-step training-fit smoke only**. It is not a full-training launcher, evaluator, submission path, checkpoint-resume path, or authorization artifact. The emitted report and checkpoint are explicitly marked `SMOKE_ONLY`, `resumable: false`, `nominal_resumable_step: false`, `optimizer_update_semantics: SMOKE_ONLY_LR_PRIMED_FINITE_UPDATE_PROBE`, `full_training_authorized: false`, `evaluation_authorized: false`, and `submission_authorized: false`.

The integrated bytes are on branch `agent/fi-acc8-training-r1`, based directly on `f6216164267e401653b56913c46d0f8e62a84140`. This checkpointed recipe is the single-variable adaptation after the documented exact FP32 batch-1 maximum-shape attempt failed with `torch.cuda.OutOfMemoryError` while requesting 18 MiB at `feature_varnet.py:1520 -> apply_model_with_crop -> Unet2d InstanceNorm`; that attempt is preserved in `/root/result/FI_ACC8_TRAINING_REVIEWS/FI_ACC8_ONE_STEP_ACTUAL_R5_OOM_BLOCKED.json`.

The subsequently authorized checkpointed real one-step run successfully traversed forward, backward, and optimizer update with approximately `6152 MiB` observed, then failed closed only at final directory publication with `ValueError: FI-VarNet smoke staging and destination must share a parent`. The caller had retained the lexical relative destination from `../result`, while staging had correctly canonicalized its own path beneath `/root/result`. This revision canonicalizes the final output once, before staging, and carries that same absolute path through publication and return. No GPU, CUDA, training, organizer H5 access, output publication, or orphan recovery was performed while making or CPU-verifying this publication-only fix.

The failed run's sealed hidden `.fi-acc8-training-fit-smoke-unpublished-orphan-59cd…` forensic tree remains preserved under `/root/result/LOCAL_FI_ACC8_CKPT_SMOKE_R1/`. It must not be deleted, modified, salvaged, scored, or published. The authoritative final path remains absent.

**Full FI-VarNet training remains prohibited.** The CLI rejects FI-VarNet without `--fi-acc8-one-step-smoke`. A successful one-step smoke still does not authorize full training; a separate review, explicit launch authority, and a separately implemented/reviewed bounded full-training path are required.

## Frozen recipe

| Field | Frozen value |
|---|---:|
| schema | `fi-varnet-acc8-training-recipe-v1` |
| model family | `fi-varnet-acc8` |
| initialization | scratch only; no external learned state or resume |
| seed | `431` |
| batch size | `1` |
| precision | FP32; no autocast |
| optimizer | `torch.optim.AdamW` |
| nominal full-training learning rate | base LR `3e-4`; LambdaLR step 0 is `0.0` |
| smoke-only applied learning rate | exactly `3e-4` for the sole optimizer update |
| weight decay | `0.0` |
| loss | pinned upstream `fastmri.losses.SSIMLoss` |
| gradient clipping | disabled |
| cascades | `12` feature cascades + `12` image cascades |
| activation checkpointing | every feature and image cascade; `torch.utils.checkpoint.checkpoint`; `use_reentrant=False`; `preserve_rng_state=True`; training/grad-enabled path only; exact `state_dict` keys and tensor values unchanged |
| cascade U-Net channels / pools | `18` / `4` |
| sensitivity channels / pools | `8` / `4` |
| acceleration | `8` only |
| organizer training inventory | exactly `170` paired entries: `85` selected acc8 files / `2315` selected acc8 slices, plus `85` recognized ignored acc4 files |
| maximum accepted k-space item shape | `(15, 640, 386)` |
| nominal full recipe | `40` epochs / `92600` steps |
| LR ramp | `3704` steps |
| cosine-decay start | step `46300` |
| nominal retention | checkpoint and reconstruction every epoch |
| smoke execution bound | exactly one selected maximum-shape item and one optimizer step |

The nominal 40-epoch fields are provenance only in this integration. They are not executable full-training authorization. The upstream schedule definition remains exactly the frozen LambdaLR recipe: ramp `3704`, cosine start `46300`, and maximum step `92600`, with multiplier/LR `0.0` at nominal step 0. Constructing that scheduler therefore sets optimizer LR to zero.

For this one non-resumable finite-update feasibility probe only, the smoke records that nominal step-0 multiplier/LR, then explicitly primes the sole optimizer param group to the frozen base LR `3e-4` immediately before its one `optimizer.step()`. It rejects a zero or non-finite applied LR. After the update it calls exactly one `scheduler.step()`, which restores the untouched nominal schedule at step 1 (multiplier `1/3704`, LR `8.099352051835852e-08`). This transient smoke-only priming does **not** alter `fi_lr_multiplier`, the LambdaLR base LR, or any nominal 40-epoch scalar. Consequently, the emitted state is deliberately not represented as an exact or resumable first step of full training.

Activation checkpointing is the **only** recipe adaptation from the failed exact FP32 attempt: architecture, maximum input, batch size, FP32 precision, optimizer, scheduler, and loss remain frozen. The adapter wraps all 12 feature cascades and all 12 image cascades only while training with gradients enabled. Every wrapper calls `torch.utils.checkpoint.checkpoint(..., use_reentrant=False, preserve_rng_state=True)`. Enabling it is idempotent and verifies that the complete ordered `state_dict` key set and every tensor value are unchanged; therefore checkpoints remain state-dict-transparent. The exact seven-field activation-checkpointing contract is repeated at checkpoint/report top level and in provenance. `save_smoke_checkpoint` copies caller provenance before inserting that contract, so the serialized snapshot gains mandatory evidence while the caller-owned mapping remains unchanged.

## Pinned upstream provenance

The thin adapter verifies these immutable upstream bytes before GPU selection or CUDA use:

- checkout: `/root/upstream-fastMRI-91f2df47`
- commit: `91f2df4711adbb6d643df1810f234e4abcf5881b`
- `fastmri_examples/feature_varnet/feature_varnet.py` SHA-256: `810bf9c18b6e81b38bfc7b3732a26e2b87dc146c907a9b8bbc2d63428ea45d99`
- `fastmri/losses.py` SHA-256: `73ebfe3bc2d9c72b04250cc5a8dc35f31b283496a9411ab92fb422eca59f57ad`
- `fastmri/data/transforms.py` SHA-256: `0eedd9b6762ea720bd8014a8cd0365a022e1e16de96293b609e45fad96fb65c2`
- `LICENSE.md` SHA-256: `52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85`

The CPU suite imports and executes `SSIMLoss` from those pinned `losses.py` bytes. It also verifies the legacy repository VarNet factory remains on its original path.

## Data and integrity contract

The smoke accepts only the exact production root `/root/Data/train`, containing paired `kspace/` and `image/` directories. Every absolute root component and both data subdirectories are opened descriptor-relatively with `O_DIRECTORY | O_NOFOLLOW`. The complete paired basename sets must match exactly. Every lowercase `.h5` basename is classified by exactly one whole underscore-delimited `acc8` or `acc4` token; malformed, unknown, or ambiguous names are rejected. Production counts are frozen separately to `organizer_total_entries=170`, `selected_acc8_count=85`, and `ignored_acc4_count=85`. Symlinked directories, symlinked/non-regular leaves, mismatched filename sets or counts, malformed selected-acc8 shapes, and masks that are not binary stored `float32` width vectors are rejected. The input, target, and max keys are frozen to `kspace`, `image_label`, and `max`.

Only the 85 selected acc8 pairs are opened as payloads. Inventory inspection opens each selected acc8 H5 leaf relative to a retained directory FD with `O_NOFOLLOW`, records directory and leaf device/inode/size identities, and hashes and passes that same opened FD to h5py. It hashes the same FD again after inspection. Selected-sample loading reopens the entire directory chain nofollow, requires directory and selected-leaf identity plus digest equality with inventory, and performs pre-read hash, H5 read, and post-read hash on the same FD. The stored mask remains a binary `float32` width vector while it is multiplied into k-space. At the pinned official model boundary it is reshaped to `[B,1,1,W,1]` and converted to `torch.bool`, matching the attested upstream `fastmri/data/transforms.py` contract. The smoke requires masked-out k-space to be zero and records stored/model dtypes and shapes plus the applied stored-mask SHA-256 in report, checkpoint, and data provenance.

The 85 co-resident acc4 pairs are recognized but ignored by this acc8 lane. Each ignored leaf is checked with no-follow metadata only, must be regular, and must retain the same device/inode/size/mtime/ctime identity across selected-acc8 inspection. Ignored acc4 payloads are never opened, hashed, or H5-read. Data provenance in both `report.json` and the checkpoint records the three scoped counts, every ignored-acc4 stat-only identity record, a SHA-256 over those canonical records, selected-acc8 payload identities/digests, directory identities, and the complete manifest SHA-256. The ignored-acc4 identity digest is explicitly a digest of metadata records, not an acc4 payload digest.

## Publication and checkpoint contract

The final output is canonicalized exactly once with lexical `abspath` before staging. For the command below, the returned and published final directory is the canonical absolute path `/root/result/LOCAL_FI_ACC8_CKPT_SMOKE_R1/fi-acc8-training-fit-smoke`. That same absolute `Path` is used through staging, descriptor-bound publication, checkpoint reload, and return; later caller working-directory changes cannot reinterpret it. This does not call `resolve()` or weaken symlink, parent-identity, descriptor-identity, or `RENAME_NOREPLACE` checks.

The output is prepared as a private, same-parent staged directory with a durable `INCOMPLETE` marker. Checkpoint serialization uses a same-directory anonymous `O_TMPFILE` descriptor when supported. On a trusted/cooperatively locked directory where that primitive is unavailable, it uses a retained random named descriptor. Both paths validate a restricted CPU reload before descriptor-based atomic no-overwrite publication and directory `fsync`.

The named fallback is explicitly limited to trusted writers that cooperate on the directory lock. Failed publication never deletes a possibly replaced staging pathname; it may retain one bounded hidden forensic orphan. Successful publication removes its owned temporary alias. An existing final checkpoint or run directory is never overwritten.

The checkpoint contains detached, cloned CPU snapshots of model, nested optimizer state, scheduler state, RNG state, sampler identity, recipe, and copied provenance. Its top-level and provenance `activation_checkpointing` mappings must both equal the exact all-12-feature/all-12-image nonreentrant, RNG-preserving, state-dict-unchanged contract; missing, malformed, or disagreeing evidence is rejected. Adding this mandatory mapping to serialized provenance does not mutate the caller's provenance object. The checkpoint is immutable (`0444`), restricted-loader compatible, exactly round-tripped on CPU, and explicitly non-resumable. Its smoke result records the nominal step-0 multiplier/LR, smoke-applied LR, post-step nominal multiplier/LR, trainable and changed parameter counts, and distinct SHA-256 digests of deterministic detached pre/post trainable-parameter snapshots. Checkpoint validation rejects zero/non-finite or non-frozen applied LR, absent parameter changes, malformed/equal digests, and any claim that this is a nominal resumable step. The reconstruction is also CPU-reloaded and validated. `report.json` repeats the finite-update evidence, exact activation-checkpointing evidence, and smoke-only semantics, is fsynced, `COMPLETE` replaces `INCOMPLETE` last, the full tree is sealed read-only and fsynced, and the final run directory is published with `RENAME_NOREPLACE` followed by parent-directory `fsync`.

A successful fresh run publishes beneath:

```text
/root/result/LOCAL_FI_ACC8_CKPT_SMOKE_R1/fi-acc8-training-fit-smoke/
├── COMPLETE
├── checkpoint-step-000001.pt
├── reconstruction-step-000001.pt
└── report.json
```

The failed hidden orphan is forensic evidence, not the authoritative final output, and must remain untouched. The command below targets the still-absent canonical final path; any existing final path or a concurrent winner remains a hard no-overwrite collision.

## Required first-CUDA review and evidence

Before issuing the command, an independent reviewer/launch authority must approve the exact current source hashes and this command. The first CUDA run must then retain evidence that:

1. source commit, FI source, pinned SSIM loss, and MIT license hashes passed **before** GPU probing;
2. `nvidia-smi` identified the selected physical index and exact expected UUID;
3. the device name was `GeForce GTX 1080` or `NVIDIA GeForce GTX 1080`, with reported memory `8192 MiB`;
4. the selected UUID had zero compute-owner PIDs immediately before launch;
5. the final report records the selected index/UUID/name/memory, runtime versions, the exact `170` total / `85` selected acc8 / `85` ignored acc4 counts, ignored-acc4 stat-only records and metadata digest, sampled selected-acc8 H5 identity/digests, the exact activation-checkpointing contract for all `12 + 12` cascades, finite loss, positive gradient-bearing parameter count, the nominal step-0 LR `0.0`, smoke-applied LR `3e-4`, post-step nominal LR, positive changed-parameter count, distinct pre/post parameter snapshot digests, and `global_step == 1`;
6. checkpoint optimizer state independently confirms exactly one optimizer step, checkpoint scheduler state confirms exactly one scheduler step, and checkpoint/report top-level activation-checkpointing evidence exactly matches their provenance copies; the output contains only the immutable smoke artifacts with `COMPLETE` published last;
7. any CUDA OOM, telemetry failure, digest change, collision, non-finite value, or publication failure is a smoke failure—not permission to retry blindly or continue to full training.

CPU tests and mocked dispatch prove ordering, mixed-cohort classification, stat-only ignored-entry handling, and artifact contracts against synthetic data. They do not prove real CUDA feasibility, VRAM fit, cleanup-to-idle, or the live organizer inventory. Those claims require evidence from the separately authorized first CUDA smoke; this handoff preparation did not open any organizer H5 payload.

## Proposed real one-step smoke command — do not run without authority

From the repository root, with `EXPECTED_GPU_UUID` set to the independently verified physical GTX 1080 UUID and with a fresh output name:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python train.py \
  --model-family fi-varnet-acc8 \
  --fi-acc8-one-step-smoke \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --GPU-NUM 0 \
  --net-name LOCAL_FI_ACC8_CKPT_SMOKE_R1 \
  --data-path-train /root/Data/train
```

Do not add resume, learned-state, alternate loss, alternate H5 key, precision, batch-size, architecture, acceleration, optimizer, schedule, or full-training options. Do not run evaluation or submission after the smoke without a separate review and authorization.

## CPU-only verification recipe

These commands do not query a GPU, initialize CUDA, or open organizer H5 files:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_fi_acc8_training.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
PYTHONPYCACHEPREFIX="$(mktemp -d /tmp/fi-acc8-pycache-XXXXXX)" \
  python -m py_compile \
  train.py \
  utils/learning/train_part.py \
  utils/learning/fi_acc8_training.py \
  utils/model/fi_varnet_adapter.py \
  tests/test_fi_acc8_training.py
PYTHONDONTWRITEBYTECODE=1 python scripts/check_submission.py
git diff --check
```

The repository-wide `python -m pytest -q` command is baseline-comparable but is expected to encounter the pre-existing `utils/learning/test_part.py::test` pytest collection error; report it separately rather than treating it as a new FI integration regression. The canonical `tests/` suite is the broad regression gate.
