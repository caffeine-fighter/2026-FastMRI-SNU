# 2026 SNU FastMRI final package

This package contains one VESSL-trained, acceleration-routed PromptMR+ C10
checkpoint and the code and evidence needed to reproduce it. The only learned
state in the package is `best_model.pt`; there is no fallback checkpoint,
ensemble, model soup, optimizer state, EMA, SWA, or external learned state.

The organizer inference entry point is `recon_eval.sh`. The end-to-end training
entry point is `reproduce_final.sh`. Both accept data and output locations at
runtime and require no source-code edits.

## Quick verification

Use Python 3.10 in a clean Linux/VESSL environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sha256sum -c SHA256SUMS
python verify_package.py
python submission_audit.py
```

`SHA256SUMS` covers every sealed package file except itself and
`package-manifest.json`. The manifest independently records the SHA-256 and byte
length of every sealed file, including `SHA256SUMS`. The final archive has a
separate `.tar.gz.sha256` sidecar.

## Inference and scoring

The exact organizer-compatible command is:

```bash
bash recon_eval.sh "$DATA_ROOT" "$OUT_DIR"
```

`DATA_ROOT` must contain `acc4/` and `acc8/`. `OUT_DIR` is writable scratch
space. If arguments are omitted, the script uses the relative locations
`data/leaderboard` and `../fastmri_eval_output`. The script verifies the package,
creates a disposable runtime tree under `OUT_DIR`, places the single checkpoint
at the fixed location expected by the unmodified organizer harness, and runs
the packaged `project/recon_eval.py`.

For the team's one recorded final evaluation, use:

```bash
bash run_official_evaluation_once.sh "$DATA_ROOT" "$OUT_DIR"
```

That wrapper is intentionally one-shot. It records the model hash, complete
evaluation log, per-acceleration SSIM values, reconstruction time, timestamps,
and return code in `evidence/`. The organizer may independently run
`recon_eval.sh` any number of times. Training and mask routing do not read
leaderboard data; the evaluation harness receives it only for scoring.

## Routing table

All routing is performed inside the officially timed `recon_slice()` call and
uses only the mask derived from the input k-space.

| Input mask | PromptMR component | Outer views | Post-refiner |
|---|---|---|---|
| Exact legal ACC4 | ACC4 specialist | identity | shared neighbor-ZF NAF_S |
| Exact legal ACC8 | ACC8 specialist | identity, left-right flip | shared neighbor-ZF NAF_S |
| Unknown or mismatched | E49 generalist | identity | shared neighbor-ZF NAF_S |

Mask density, ACS width, period, residue, and offset determine the exact route.
Filename, image field, target, bbox annotation, and score are not routing inputs.
`prep_volume()` loads input only. PromptMR, zero-filled context construction,
NAF_S, view restoration, and averaging all execute inside `recon_slice()`.

## End-to-end training

Organizer data must have this layout:

```text
DATA_ROOT/
  train/
    kspace/
    image/
  val/
    kspace/
    image/
```

Start from an empty writable output directory on one NVIDIA GeForce GTX 1080:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=430
bash reproduce_final.sh "$DATA_ROOT" "$OUT_DIR"
```

The runner refuses to overwrite an existing lineage. It performs this fixed
sequence:

1. Fresh FP32 PromptMR+ R2/C10/H0 training with seed 430 and batch size 1.
2. Atomic E49 handoff at optimizer step 228,928 on the unchanged E51 cosine
   horizon.
3. ACC4 specialist training for 7,008 optimizer steps with a 35,040-step LR
   horizon and peak LR 2.5e-5.
4. ACC8 specialist training for 1,158 optimizer steps with a 2,315-step LR
   horizon and peak LR 5e-5.
5. Frozen-C10, neighbor-ZF NAF_S training for 88,895 optimizer steps on a
   93,567-step LR horizon.
6. Construction of exactly one routed `best_model.pt`.

The generalist and specialists use organizer train data. Organizer validation
is appended as ordinary training data only for the terminal NAF_S full push.
There is no validation loop, early stopping, or validation-based checkpoint
choice in this lineage. Training code does not access test targets.

The legal-mask augmentation retains the original ACC4 example and partitions
its acquired non-ACS lines into two complementary virtual ACC8 examples while
preserving ACS. The sampler and optimizer-step budget are defined before this
augmentation. No augmentation is applied during inference or official scoring.

The NAF_S input channels are the routed reconstruction, current zero-filled RSS,
their residual, previous-slice zero-filled RSS, and next-slice zero-filled RSS.
Neighbor slices are from the same masked-k-space volume, with nearest-slice
replication at volume boundaries. Every C10 parameter remains frozen while the
73,489-parameter refiner is trained.

## Reproducibility evidence

The final assembly refuses to run unless all four real VESSL logs and the real
VESSL environment snapshot are supplied:

- `evidence/training_logs/generalist.log`
- `evidence/training_logs/acc4_specialist.log`
- `evidence/training_logs/acc8_specialist.log`
- `evidence/training_logs/naf_s.log`
- `evidence/environment/pip_freeze.txt`

The package also includes hash-bound E49, specialist, NAF_S, policy, assembly,
and inference-admission receipts. `package-manifest.json` binds all evidence to
the single final checkpoint.

`evidence/raw/` preserves the exact VESSL contracts and source hashes, including
their historical server locations. Those files are immutable evidence and are
never executed by either entry point. The runnable copies under `reproduction/`
replace only server-location literals with `DATA_ROOT`, `OUT_DIR`,
`FASTMRI_TRAIN_ROOT`, and `FASTMRI_RESULT_ROOT`; the path-only transformation is
hash-bound in `reproduction/portability-receipt.json` and changes no numeric
recipe or architecture.

Random initialization, sampler order, legal-mask generation, augmentation,
Python hashing, and cuDNN deterministic mode are fixed by seed 430. The expected
result is reproduction to the organizer's four displayed decimal places on the
pinned VESSL GTX1080 environment. Bit identity is not promised across different
GPU architectures, CUDA versions, or PyTorch builds.

## Pinned environment

- Ubuntu on VESSL
- NVIDIA GeForce GTX 1080, 8,192 MiB
- Python 3.10.12
- CUDA 12.1
- PyTorch 2.3.1+cu121
- NumPy 1.24.4 (NumPy major version 1)
- seed 430, batch size 1, FP32

`requirements.txt` is the install specification. The actual final server state
is preserved separately in `evidence/environment/pip_freeze.txt`.

## Directory tree

```text
final_package/
  README.md
  requirements.txt
  recon_eval.sh
  reproduce_final.sh
  run_official_evaluation_once.sh
  best_model.pt
  SHA256SUMS
  package-manifest.json
  project/
    recon_eval.py
    reconstruct.py
    utils/
    third_party/
  reproduction/
    generalist/
    specialist/
    organizer-data-provenance.json
    source-sha256sums.txt
    vessl_train_post_refiner.py
    vessl_build_routed_promptmr_checkpoint.py
  evidence/
    training_logs/
    environment/
    raw/
      contracts/
      training_source/
    official-evaluation-receipt.json
  verify_package.py
  submission_audit.py
  build_submission_archive.py
```

Packaging helpers and receipts are included because they verify lineage and
integrity; generated reconstruction outputs, caches, demo files, local research
artifacts, and extra checkpoints are excluded.

## Linux-safe archive

After the one official evaluation has passed, build the upload archive with an
ASCII team slug:

```bash
python build_submission_archive.py --team-slug TEAM_NAME --output-dir ..
```

This creates `2026_FastMRI_TEAM_NAME.tar.gz` and
`2026_FastMRI_TEAM_NAME.tar.gz.sha256`. The builder uses POSIX member names,
forbids symlinks and unsafe filenames, extracts the archive into a fresh
temporary directory, and reruns the submission-ready audit there.

The explanation video and presentation are separate attachments named
`2026_FastMRI_TEAM_NAME.mp4` and `2026_FastMRI_TEAM_NAME.pdf` (or an accepted
presentation extension); they are not mixed into the model archive.

## Submission handoff

Before sending the email, copy the score and time from
`evidence/official-evaluation-receipt.json`, verify the archive sidecar, and use
the organizer's exact title format:

```text
[최종제출] TEAM_NAME – iabengXXX
```

The email body must state the public score, Full, Bbox, ms/slice, certificate
preference, and whether the model is provided by server, attachment, or both.
When leaving the package on VESSL, state its exact server artifact path and the
archive SHA-256. Replace every placeholder and omit braces in the actual email.
