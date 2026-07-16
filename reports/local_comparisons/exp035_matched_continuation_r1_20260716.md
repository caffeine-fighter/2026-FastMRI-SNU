# EXP035 matched continuation R1

- Date: 2026-07-16
- Environment: VESSL, NVIDIA GeForce GTX 1080 (8,192 MiB)
- Repository commit: `317bf27c36ea3db340982d6022537940f0e2ce48`

## Question

Does reducing Adam learning rate from `0.001` to `0.0003` improve an exact five-epoch continuation of the protected EXP035 epoch-30 state, beyond the effect of simply training for five more epochs?

The competition target is total `0.94`, not `0.93`. EXP035 currently scores `0.92146109375`; at the same time score the remaining gap is `+0.01853890625`. This makes a marginal vanilla continuation insufficient as the primary route to the target.

Both arms independently resumed the same immutable state:

- generation: `3e8af14268a64d67a308ebe30484ddf2`
- checkpoint SHA-256: `dc6e034f18df2a7872c416d4dccb4bb00e6e5b41fb89e438a86682db3097ffb7`
- validation-history SHA-256: `e976a7f5fa2d700c7dbfca93bbb5a1853d49cbd4070d6a5cfe55f8c39512f23f`
- stored epoch: `30`
- architecture: c8/ch12/s8
- batch size: `1`
- seed: `430`
- total epoch: `35`

The Control and Candidate used separate regular-file source copies with distinct inodes and identical protected hashes. Exact resume restored model, optimizer, and RNG state. The only scientific variable was the resumed Adam learning rate.

## Arms

| Arm | Output identity | Adam LR | Completion |
|---|---|---:|---|
| Control | `LOCAL_EXP035_E30_TO_E35_ADAM_LR1E3_SEED430_R1` | `0.001` | PASS |
| Candidate | `LOCAL_EXP035_E30_TO_E35_ADAM_LR3E4_SEED430_R1` | `0.0003` | PASS |

### Verified checkpoint evidence

| Arm | Best epoch | Optimizer | LR | Generation | Checkpoint SHA-256 |
|---|---:|---|---:|---|---|
| Control | 34 | Adam | `0.001` | `760a3925404f4192b13ab9b3249a5409` | `48a5ca2f1d3b701e67b95e33b62227aa10bac5373e38bd5d930c265dec7cc2cd` |
| Candidate | 34 | Adam | `0.0003` | `4d9295fbfd18492286c107eae0774620` | `96d3c89225647871e588b26be39c8c6279f94ccd17ddcf56b399f61acff6f081` |

Both checkpoints use format version 1 and contain 224 Adam states with `exp_avg` and `exp_avg_sq`. The production constructor is `torch.optim.Adam`; the serialized parameter-group learning rates match the two preregistered arms.

Both VESSL runs completed epochs 31–35 without OOM, NaN, non-finite history, or training errors. The GPU returned idle after completion.

## Strict validation

Each retained epoch passed:

- overall: 30 volumes / 791 slices / 161 boxes
- acc4: 15 volumes / 407 slices / 107 boxes
- acc8: 15 volumes / 384 slices / 54 boxes
- skipped: `[]`
- unknown: `0`
- non-finite: `0`
- regular, non-symlink reconstruction H5 files with the expected schema and filename set

The first canonical sweep invocation pointed at the validation parent directory and failed closed before publishing because the H5 targets live in its image child. The failed staging orphan was preserved. Fresh, collision-free sweeps against the correct target directory passed all gates for both arms.

## Equal-acceleration trajectory

| Epoch | Control quality | Candidate quality |
|---:|---:|---:|
| 30 source | 0.9199788092 | 0.9199788092 |
| 31 | 0.9191209472 | 0.9204724846 |
| 32 | 0.9193542230 | 0.9202920177 |
| 33 | 0.9194130287 | 0.9205140402 |
| 34 | **0.9202459833** | **0.9205928470** |
| 35 | 0.9199206931 | 0.9204108386 |

Epoch 34 was the global best for both arms across source epoch 30 and continuation epochs 31–35.

## Matched best-epoch comparison

| Metric | Control epoch 34 | Candidate epoch 34 | Candidate - Control |
|---|---:|---:|---:|
| acc4 full | 0.9212700175 | 0.9216055381 | +0.0003355205 |
| acc4 bbox | 0.9409147429 | 0.9419705495 | +0.0010558066 |
| acc8 full | 0.8918866431 | 0.8922531530 | +0.0003665099 |
| acc8 bbox | 0.9269125296 | 0.9265421474 | **-0.0003703822** |
| equal-acc full | 0.9065783303 | 0.9069293456 | +0.0003510152 |
| equal-acc bbox | 0.9339136362 | 0.9342563484 | +0.0003427122 |
| quality | 0.9202459833 | 0.9205928470 | **+0.0003468637** |

The Candidate improved over source epoch 30 by `+0.0006140378`, but the preregistered comparison is Candidate versus the independently continued Control. Its matched gain was only `+0.0003468637`, short of the required `+0.0005` by `0.0001531363`. The protected acc8 bbox component also decreased by `0.0003703822`.

## Decision

**DO NOT PROMOTE. The vanilla capacity/continuation track is closed.**

- Lower-LR Candidate: rejected.
- Official evaluation: not authorized.
- Second seed: do not run.
- Epoch-40 continuation: do not run.
- c9/c10/c12 vanilla capacity expansion: do not run.
- EXP035 epoch 30: protected official leader.
- Candidate epoch-34 checkpoint: research artifact only.
- Repeated timing: not run; remains reserved for a separately approved final freeze.

The Candidate is `+0.0006140378` above source epoch 30, but that is not promotion evidence: the matched improvement over the independently continued Control is only `+0.0003468637`, below the required `+0.0005`, and the protected acc8 bbox component decreased by `0.0003703822`.

The lower-LR direction does not authorize another continuation block, an official one-shot, or final timing. GPU budget moves to a separately source- and license-gated model-family feasibility probe rather than more vanilla epochs.

## Evidence

Local fail-closed evidence is preserved outside Git:

- Control strict marker: `CONTROL_STRICT_PASS.json`
- Candidate strict marker: `CANDIDATE_STRICT_PASS.json`
- matched comparison schema: `exp035_matched_continuation_r1_comparison_v1`
- matched comparison status: `COMPLETE`
- official evaluation flag: `false`
- sanitized machine-readable record: [`exp035_matched_continuation_r1_20260716.json`](exp035_matched_continuation_r1_20260716.json)

No checkpoint, reconstruction, H5, mounted data, credential, or result directory is included in this repository update.
