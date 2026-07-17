# PromptMR+ pinned training runbook

Status: `RESERVED_NOT_LAUNCHED`

Reserved candidate name: `EXP036_promptmr_plus_default_e5_seed430`

Objective: Maximize validated expected official total score while preserving evidence
validity, component robustness, reproducibility, and final 8GB deployment feasibility.
No fixed score threshold is an automatic failure or stopping rule. Local metrics must
not be converted into an estimated official total, and official evaluation always
requires separate explicit approval.

This document prepares a run; it does not authorize or start one. Do not add EXP036
to `experiments/experiment_log.csv`, create its result directory, or run training until
the license and smoke-test gates below are satisfied.

## Integration boundary

This change is a CPU-validated **training-path integration only**. The current official
reconstruction path (`reconstruct.py` and `utils/learning/test_part.py`) still constructs
only `VarNet` and does not consume checkpoint `model_contract` or PromptMR+ recipe/source
metadata to select an inference factory. A PromptMR+ state dict therefore cannot silently
load as a working VarNet (its parameters are incompatible), but it also cannot yet run
through the official reconstruction path. Training checkpoints preserve family, recipe,
source, scheduler, and scaler provenance for that future adapter. Official-path inference,
GPU smoke/reload validation, and competition readiness remain separate pending gates.

## Source and license gate

- Upstream: `https://github.com/hellopipu/PromptMR-plus`
- Commit: `934eeda6d4d18cd39e406fa1eee9e1f70603cb5e`
- Canonical source/config/license hashes: `vendor/promptmr_plus/SOURCE_MANIFEST.json`
- Local adapter boundary: `vendor/promptmr_plus/ADAPTERS.md`
- License: Rutgers PromptMR+ Non-commercial Research License (`LICENSE.md`)

The bundled license permits source/binary redistribution and modification only for
noncommercial purposes, requires preservation of its copyright notice, conditions,
and disclaimer, and does not grant commercial-use permission. Before adding the
confirmation flag or launching, retain written confirmation that this competition,
training, packaging, and submission use is noncommercial. Rutgers and contributor
names must not be used for endorsement.

## Pinned training contract

- family: `promptmr_plus`; VarNet remains the CLI default
- PromptMR+ v2, 12 cascades, 5 adjacent slices, clamp/repeat at volume edges
- central adjacent slice is the reconstruction output
- train preprocessing: image-domain center crop/reflect pad to 384 x 384
- validation input: full resolution; only output/target are center-aligned for loss
- loss: SSIM, window 7, k1 0.01, k2 0.03
- optimizer: AdamW, lr 1e-4, weight decay 1e-2
- scheduler: StepLR, step 35, gamma 0.1
- clipping: global gradient norm 0.01
- batch size 1; seed 430; FP32
- architecture checkpointing and per-coil sensitivity computation enabled
- checkpoint payload includes PromptMR+ recipe identity/hash, optimizer, scheduler,
  disabled FP32 scaler state, RNG state, epoch, and best validation loss
- every completed validation epoch is retained without overwrite

`configs/promptmr_plus_training.json` is authoritative for this local integration.
Architecture/config changes require a new recipe ID and checkpoint contract.

## CPU-only preflight

Use the Python environment that contains project dependencies. These commands do not
instantiate or execute the model:

```bash
python -m unittest \
  tests.test_promptmr_training_contract \
  tests.test_promptmr_source_pin \
  tests.test_promptmr_checkpoint_contract \
  tests.test_promptmr_dataset \
  tests.test_promptmr_planner_cli \
  tests.test_promptmr_training_routing \
  tests.test_promptmr_reconciliation
python -m py_compile train.py utils/promptmr/*.py scripts/plan_promptmr_run.py
```

Run the metadata-only planner against the real train/validation roots. Omit
`--license-confirmed` until written confirmation exists; its absence keeps the printed
candidate command fail-closed at the training parser.

```bash
python scripts/plan_promptmr_run.py \
  --train-data-path /Data/train \
  --val-data-path /Data/val \
  --output-parent ../result \
  --epochs 5 \
  --retain-val-epochs
```

The planner reads HDF5 shapes/attrs only. It reports exact 4x/8x volume and slice
counts, steps, dependency presence, an explicit host-RAM estimate, a disk estimate,
free disk, and complete control/candidate commands. It must report
`creates_output_directories: false` and must not reserve an experiment directory.
The RAM estimate covers input/target staging only; it is not a GPU/model-memory claim.
The disk estimate covers both emitted arms: one immutable checkpoint generation per epoch,
two stable checkpoint aliases per arm, retained validation payloads, and 1 MiB of metadata
reserve per epoch per arm. `--checkpoint-reserve-gib` is the assumed size of one full
checkpoint, not the whole run.

## Resume contract

PromptMR+ resume requires an exact `model_family`, recipe ID, and recipe SHA-256 match,
plus optimizer, scheduler, scaler, and RNG state. A missing or mismatched field fails
before live model/optimizer state is changed. `--resume-lr` is not allowed for the
pinned PromptMR+ recipe. Legacy metadata-free VarNet checkpoints remain accepted only
through the VarNet family path.

## Local 24 GB smoke gate (not approved by this runbook)

Do not execute this gate until the user explicitly approves local GPU use. First rerun
the planner with real paths and legal confirmation, then take its exact candidate
command and bound it to one training step/one validation sample using a disposable
smoke-only dataset or test hook. Required acceptance evidence:

1. exact device identity and 24 GB capacity;
2. one PromptMR+ forward, SSIM backward, unscale, clip, AdamW step, and StepLR state;
3. peak allocated/reserved memory;
4. checkpoint save and reload with exact family/recipe/scheduler/scaler metadata;
5. one full-resolution validation reconstruction with shape-safe alignment;
6. no EXP036 registry row or production output directory from the disposable smoke.

If no bounded hook exists, add and review one rather than launching a full epoch.

## Candidate launch and retained validation

After written license confirmation, CPU checks, explicit GPU-smoke approval, and a
passing 24 GB smoke:

1. rerun the planner with `--license-confirmed` and the real roots;
2. review its 4x/8x counts, disk/RAM assumptions, and parser-complete commands;
3. register the experiment once with immutable source/recipe/command provenance;
4. atomically reserve a new result directory;
5. launch exactly one approved command;
6. retain each completed validation epoch under
   `reconstructions_val_epochs/epoch_XXXX`;
7. verify checkpoint manifest, history, best alias, and retained-tree consistency;
8. select a candidate by the declared validation metric, not by test-server probing;
9. materialize the candidate checkpoint with SHA-256 and exact reconstruction command;
10. run the official harness once under the final-evaluation runbook.

Each retained tree remains reconstruction-only (`*.h5`). After immutable model and
history artifacts are durable, the checkpoint directory receives an immutable,
no-replace generation publication record containing parent-manifest, artifact-hash,
epoch, and retained-tree-digest provenance. The retained tree is then published with
`RENAME_NOREPLACE`, followed by the atomic checkpoint-manifest commit and alias refresh.
On restart, one retained epoch ahead of the manifest is adopted only when exactly one
publication record matches every parent, generation, artifact, epoch, and tree digest.
An unprovenanced final output, missing/changed history, multiple candidates, duplicate
or non-next epoch, existing destination, or unknown entry fails closed before any
device query. Hidden `*-unpublished-orphan-*` staging trees are never adopted or
deleted, but are ignored as non-final so the old manifest epoch can be rerun. Nothing
is overwritten or automatically deleted.
