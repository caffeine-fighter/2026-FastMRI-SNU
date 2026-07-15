# LOCAL Adam-vs-AdamW matched E5 V10

**Completed:** 2026-07-14 16:39 KST
**Scope:** independently verified desktop RTX 3090 evidence only. This result is not an official score, candidate checkpoint, or authorization for another run.

## Question

Does changing only the optimizer from fixed-LR Adam to AdamW with weight decay `1e-6` improve the established `c8/ch12/s8` five-epoch LOCAL trajectory?

Both arms used the same frozen source, data, evaluator, seed, learning rate, batch size, architecture, and retained-epoch selection procedure.

| Field | Control | Candidate |
|---|---|---|
| Experiment | `LOCAL_EXP067_RETRY8_varnet_c8_ch12_s8_e5_adam_seed430` | `LOCAL_EXP068_RETRY8_varnet_c8_ch12_s8_e5_adamw_wd1e6_seed430` |
| Optimizer | Adam | AdamW |
| Weight decay | `0` | `1e-6` |
| Architecture | `c8/ch12/s8` | `c8/ch12/s8` |
| Epochs / seed / LR / batch | `5 / 430 / 0.001 / 1` | `5 / 430 / 0.001 / 1` |
| Selected retained epoch | 4 | 4 |

## Result

The authoritative equal-acceleration metrics tied exactly.

| Metric | Adam | AdamW | AdamW - Adam |
|---|---:|---:|---:|
| Equal-acc quality | 0.9079597004022214 | 0.9079597004022214 | 0 |
| Equal-acc full SSIM | 0.8990602053686241 | 0.8990602053686241 | 0 |
| Equal-acc bbox SSIM | 0.9168591954358187 | 0.9168591954358187 | 0 |
| acc4 full | 0.9145860890205720 | 0.9145860890205720 | 0 |
| acc4 bbox | 0.9310585795161880 | 0.9310585795161880 | 0 |
| acc8 full | 0.8835343217166761 | 0.8835343217166761 | 0 |
| acc8 bbox | 0.9026598113554495 | 0.9026598113554495 | 0 |

Pooled diagnostics also tied: full `0.8995116533129918`, bbox `0.9215335268411577`.

| Runtime | Adam | AdamW | Difference |
|---|---:|---:|---:|
| End-to-end arm elapsed time | 2:11:21.736 | 2:21:12.972 | AdamW +9:51.236 (`+7.50%`) |

The selected checkpoints are distinct: Adam SHA-256 `8a7582058080bf2cf1f663c2d66ebcbaffe96aa22afb8b4de6e21073846828b7`; AdamW SHA-256 `d296eeecf2d03d4dbff097cfa8dd82de9237dcb6db6675e773f45a81c957bd6d`. The exact metric tie is therefore reported as a neutral optimizer result, not as evidence that the two arms accidentally reused one checkpoint.

## Independent verification

- Terminal status: `local_adamw_matched_e5_done`.
- Terminal SHA-256: `37ef86092ca0052a0148e547210e7edcf25245340db02aaa0070ebb1b49c48cf`.
- Runner SHA-256: `29615568674f9e164b372bea12418a4bd53a6cf7096dad465df1f91228c279a5`.
- Frozen source commit: `332f2ac4e4c9dac5634f4e8b8cf4bed86604ac57`.
- Both 228-entry publication manifests were independently rehashed:
  - Adam: `1e6a1f39760803c4902080485f2cf34b87850259a00b3db49d051a2d954284e7`
  - AdamW: `752e96621256ba3cfae5f0b7b0aaa5c931e1bab8360604f30908f43a16c14024`
- Each selected reconstruction cohort contained 30 ordinary, non-virtual H5 files and 791 finite slices.
- Evaluation coverage was exactly 30 volumes, 791 slices, and 161 boxes, with `unknown=0` and `skipped=[]`.
- Published trees were read-only and contained no symlinks.

The per-epoch metric JSON retains a pre-publication private staging path in its diagnostic `recon_dir` field. The terminal records separate producer and published paths, and this audit reopened and revalidated the published files; the stale diagnostic field is a provenance-reporting defect, not a metric difference.

## Decision

`reject_adamw_only_no_automatic_scheduler_followup`

- Retain Adam as the default for this matched configuration.
- Do not launch an AdamW second seed, longer AdamW run, AdamW scheduler rescue, VESSL promotion, or official evaluation from this result.
- EXP035 epoch 30 is now the protected vanilla baseline.
- Any scheduler probe must be an independently preregistered Adam-only comparison rather than an automatic continuation of this rejected AdamW branch.
