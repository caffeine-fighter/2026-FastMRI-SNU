# EXP035 epoch-30 validation and official result

## Candidate

- Run: `EXP035_varnet_c8_ch12_s8_e30`
- Architecture: c8/ch12/s8
- Training source: from scratch
- Epochs: 30
- Optimizer: Adam
- Learning rate: 0.001
- Seed: 430
- Source commit: `a3d3fc28d2817c17eb39bdb0c3864730042247c6`
- Selected immutable generation: `3e8af14268a64d67a308ebe30484ddf2`
- Checkpoint SHA-256: `dc6e034f18df2a7872c416d4dccb4bb00e6e5b41fb89e438a86682db3097ffb7`

The tracked training command exited normally with code 0. The final checkpoint stores epoch 30. The complete log had zero requested fatal/error-pattern matches.

## Strict retained-epoch validation

All 30 retained epochs were evaluated independently. Every epoch had exact coverage of 30 volumes, 791 slices, and 161 boxes, with `skipped=[]`, unknown count 0, and finite outputs. Selection used leaderboard-faithful equal-acceleration quality in descending order, with lower epoch breaking ties. `best_model.pt` was not rewritten.

Epoch 30 ranked first:

- Equal-acc full: `0.90666693657748`
- Equal-acc bbox: `0.9332906818845851`
- Quality: `0.9199788092310326`
- Delta versus EXP033R LOCAL `0.9156824558941089`: `+0.004296353336923686`
- acc4 full/bbox: `0.9212797105458796` / `0.9408938840170887`
- acc8 full/bbox: `0.8920541626090804` / `0.9256874797520814`

The last five quality values were:

| Epoch | Quality | Equal-acc full | Equal-acc bbox |
|---:|---:|---:|---:|
| 26 | 0.9196235036450945 | 0.9064050757421149 | 0.9328419315480740 |
| 27 | 0.9195447032863875 | 0.9063709833534140 | 0.9327184232193608 |
| 28 | 0.9190089002690043 | 0.9058430022807955 | 0.9321747982572130 |
| 29 | 0.9197685778715851 | 0.9064166067962525 | 0.9331205489469176 |
| 30 | 0.9199788092310326 | 0.9066669365774800 | 0.9332906818845851 |

The trajectory is positive but not strictly monotonic. Epoch 30 is the global best.

## Authorized official one-shot

Exactly one official run completed through `scripts/run_recon_eval_once.sh`, which wraps the required `bash recon_eval.sh` entrypoint. The tracked wrapper exited with code 0.

- SSIM full: `0.9234`
- SSIM bbox: `0.9177`
- Quality: `0.92055`
- Reconstruction time: `554.85 s`
- Time: `250.7 ms/slice`
- Time score: `0.00091109375`
- Total score: `0.92146109375`
- acc4 full/bbox: `0.9442` / `0.9424`
- acc8 full/bbox: `0.9025` / `0.8930`

Versus the prior EXP033R one-shot leader:

- Full: `+0.0035`
- Bbox: `+0.0057`
- Quality: `+0.0046`
- Time: `+77.1 ms/slice` (slower)
- Time score: `-0.00004015625`
- Total: `+0.00455984375`

The quality gain materially exceeds the timing penalty, making EXP035 epoch 30 the new one-shot official leader. The required repeated timing cohort remains separately approval-gated and has not been run.
