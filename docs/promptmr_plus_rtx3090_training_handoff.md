# PromptMR+ local RTX 3090 training handoff

Status: DESIGN-ONLY HANDOFF — NOT LAUNCHED

This handoff separates the approved environments:

- VESSL GTX 1080 8 GB: telemetry canary and full-resolution inference feasibility only.
- Local RTX 3090 24 GB: one-step training smoke and, only after that smoke passes, one EXP036 five-epoch architecture screen.
- Official evaluation: not authorized.

No PromptMR+ training, backward pass, optimizer step, checkpoint creation, EXP036 directory creation, or experiment-registry update was performed on VESSL.

## Reserved identity

- Reserved name: `EXP036_promptmr_plus_default_e5_seed430`
- Registry status: not registered
- Output directory status: not created
- Checkpoint status: not created

## Pinned source and license

- Repository: <https://github.com/hellopipu/PromptMR-plus>
- Upstream commit: `934eeda6d4d18cd39e406fa1eee9e1f70603cb5e`
- License: Rutgers Non-commercial Research License
- Permitted status: `NONCOMMERCIAL_COMPETITION_USE_ALLOWED`
- Commercial-use permission is not claimed.
- Copyright, license conditions, and disclaimer must remain intact.

SHA-256:

- Vendored source manifest: `3d77c331b3d756ea855c12c08efe82d34755b30035eaef1c933c053bfa128876`
- License: `c0c4c7d85180b493cd7a213d4509155d3734de26562e4490589963e1c356db21`
- Model config: `ac883402084ccdf4d2790fd944c6e7c972b4d2c1a6d6b094007c8eb03e1d08d6`
- Knee training config: `5241adee5cb34846a6569d37fbd4078877bb90ee3d00442e9175173bccf1be41`
- Upstream base config: `cbdbebc95f6f0e508fdaacc0703d03f2bfa9b568b4301c2f8cd9b4d9e2fdd7d7`
- Existing thin inference adapter: `e7ea03ff4beb19f5b2bf8a36a4dbacfe84d168a16f877b08909c824d1ac2f843`
- Environment requirements: `e6fcfe6a9040d90fcaa4ad18d4bfbae501f7477e1844d6399d52dbc111460f21`

## Exact architecture contract

Use the pinned upstream PromptMR+ v2 architecture without width/depth reduction:

- 12 cascades
- 5 adjacent slices
- `n_feat0=48`
- `feature_dim=[72,96,120]`
- `prompt_dim=[24,48,72]`
- sensitivity `n_feat0=24`
- sensitivity `feature_dim=[36,48,60]`
- sensitivity `prompt_dim=[12,24,36]`
- `len_prompt=[5,5,5]`
- `prompt_size=[64,32,16]`
- encoder CABs `[2,3,3]`
- decoder CABs `[2,2,3]`
- skip CABs `[1,1,1]`
- bottleneck CABs `3`
- adaptive input enabled
- `n_buffer=4`
- `n_history=11`
- adjacent-slice sensitivity enabled
- channel attention enabled
- fixed/non-learnable prompts

The pinned upstream implementation reads `PromptMRBlock.n_buffer` without initializing it; preserve the audited thin adapter's compatibility assignment rather than forking or reimplementing the upstream algorithm.

## Exact learning recipe

- Seed: `430`
- Batch size: `1`
- Precision: FP32
- BF16: forbidden
- FP16 fallback: forbidden
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-2`
- Other AdamW parameters: ordinary pinned PyTorch defaults
- Scheduler: StepLR
- Step size: `35`
- Gamma: `0.1`
- Loss: upstream SSIMLoss
- SSIM window: `7`
- `k1=0.01`
- `k2=0.03`
- Gradient clipping: norm `0.01`
- `use_checkpoint=true`
- `compute_sens_per_coil=true`
- Gradient accumulation: not part of the upstream recipe
- Hidden mixed precision: forbidden

Upstream knee training uses a 384×384 spatial preprocessing contract. That crop is part of the pinned upstream recipe and is not an architecture width/depth reduction. It must remain explicitly separated from the VESSL full-resolution inference contract at 640×480/400.

## Dataset wrapper contract

The minimum local wrapper must adapt the SNU challenge data without reimplementing PromptMR+ internals:

1. Pair each `<split>/kspace/<filename>.h5` with exactly one `<split>/image/<filename>.h5`.
2. Treat exactly one underscore-delimited `acc4` or `acc8` filename token as authoritative. Unknown, duplicate, or mismatched acceleration tokens fail closed; do not estimate acceleration from mask density.
3. Read five adjacent slices from the same volume only. At the first/last slice, edge-replicate within that volume; never cross a filename/volume boundary.
4. Concatenate the five adjacent multicoil slices to `[5*C,H,W,2]`, then batch to `[B,5*C,H,W,2]` with `B=1`.
5. Convert the stored byte mask to the upstream boolean broadcast layout `[B,1,1,W,1]`.
6. Preserve paired target, `max`, filename, slice index, crop metadata, and source-file provenance.
7. Keep annotations evaluator-only; they must not enter optimization or model selection.
8. Use one shared model/checkpoint for acc4 and acc8. Routing is a data/mask contract, not an architecture/config switch. This expectation still requires actual inference-preflight evidence before it can be claimed as deployed compatibility.
9. Preserve the full-resolution inference adapter separately from the upstream 384×384 training preprocessing.

## Local one-step smoke gate

Implement the thinnest training integration around pinned upstream components before issuing a runnable command. The current VESSL repository's `train.py` is VarNet-oriented and does not yet provide the required PromptMR+ CLI contract.

The first local RTX 3090 action must be exactly one bounded real-batch optimizer step with a fresh non-EXP output:

- validate pinned source/config/license hashes before data access
- verify RTX 3090 24 GB identity and idle state
- source H5 opened read-only
- one acc4/acc8-routed training item at batch size 1
- forward
- finite upstream SSIM loss
- backward
- all participating gradients finite
- norm clip `0.01`
- one AdamW step
- `zero_grad`
- StepLR state present but no recipe change
- checkpoint save/load round trip
- source/config/split/seed provenance
- history schema validation
- allocated/reserved/process VRAM and CPU RSS
- cleanup to idle
- no output replacement

Even 24 GB may OOM at challenge resolution. An OOM must stop the handoff; do not auto-crop beyond the upstream 384×384 recipe, shrink the architecture, or enable FP16/BF16.

## Conditional EXP036 e5 gate

Only a fully passing local one-step smoke may authorize one five-epoch screen. Before launch:

- verify `EXP036` remains collision-free in the canonical registry and filesystem
- create the registry row and output directory exactly once
- retain immutable checkpoints for epochs 1–5
- retain validation reconstructions for epochs 1–5
- bind every retained reconstruction generation to its checkpoint SHA-256
- preserve scheduler, optimizer, RNG, source/config, split, and resume provenance
- never use train+validation pretrained leakage

After exit code 0, run local strict validation only across all five epochs:

- 30 volumes / 791 slices / 161 boxes
- acc4: 15 volumes / 407 slices / 107 boxes
- acc8: 15 volumes / 384 slices / 54 boxes
- skipped `[]`, unknown `0`, non-finite `0`
- global best by equal-acc local quality descending, ties to earlier epoch

Do not run official evaluation, leaderboard submission, repeated timing, e15/e30 extension, second seed, optimizer/loss/scheduler sweep, or automatic merge.

## VESSL inference dependency

The VESSL launch gate is currently `BLOCKED_BY_PID_DOMAIN_ATTRIBUTION`. See:

`reports/promptmr_plus/PROMPTMR_PLUS_INFERENCE_PREFLIGHT_8GB_V2_BLOCKED_20260717.json`

This handoff is architecture/training design evidence only. It does not establish GTX 1080 inference feasibility, training feasibility, quality, competition readiness, official-evaluation approval, or progress toward total score 0.94.
