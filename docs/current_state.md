# Current State — Phase 2 Handoff

_Last updated: 2026-07-11 KST during the VESSL EXP031 continuation and desktop adaptive LOCAL campaign._

## Machine roles

- **VESSL**: official `EXP###` training and official `recon_eval` / leaderboard evaluation.
- **Desktop WSL**: local probes, automation, documentation, and Phase 2 wrapper preparation.
- **Laptop**: control plane / VS Code SSH client / assistant interface.

## GitHub delivery status

- Repository: `https://github.com/caffeine-fighter/2026-FastMRI-SNU`
- Default branch: `baseline/2026-baby-varnet` at `1a38022`
- VESSL feature branch: `phase2/eval-wrapper-vessl` at `1a38022`
- Submission implementation commit: `fbbddf6700cd65b1e2b52c1c6418f48a5eef9b82`
- Desktop LOCAL branch: `local/probe-sweep-20260710-desktop4070ti`
- Current LOCAL branch scope: first sweep, Round 2, adaptive plan, and status documentation
- Collision policy: keep desktop progress isolated; do not merge into default while EXP031 is active

The verified EXP030 submission implementation remains delivered on the default branch. The isolated LOCAL branch is the only path used to prepare and publish new desktop reports and documentation until the VESSL EXP031 handoff is complete. The remaining external action for the existing EXP030 package is the organizer upload.

## VESSL training status

`EXP031` is continuing the c4/ch12/s8 model from epochs 21 through 30 on VESSL. This is operator-reported active training; no EXP031 metric or checkpoint handoff is present in Git yet. A read-only watcher is monitoring the default and VESSL feature refs for movement.

The existing official 30-repeat result still refers to the `EXP030_varnet_c4_ch12_s8_e20` checkpoint. Thirty repeats are timing repetitions, not 30 training epochs.

## Current best validation candidate

`EXP030_varnet_c4_ch12_s8_e20` remains the current file-backed official candidate until EXP031 completes and passes the same VESSL validation checks.

| Experiment | Config | val_loss | SSIM_full | SSIM_bbox | quality_score |
|---|---|---:|---:|---:|---:|
| `EXP012_varnet_c4_ch12_s4_e10` | cascade=4, chans=12, sens_chans=4, epochs=10 | 3.2876096990602717 | 0.8994141339351495 | 0.9187541341189271 | 0.9090841340270383 |
| `EXP030_varnet_c4_ch12_s8_e20` | cascade=4, chans=12, sens_chans=8, epochs=20 | 3.202955630212294 | 0.90337035478141 | 0.9259878171156652 | 0.9146790859485376 |
| `EXP031` | cascade=4, chans=12, sens_chans=8, epochs=30 continuation | pending | pending | pending | pending |

EXP030 beats EXP012 by `+0.0055949519214994` validation quality.

Quality formula:

```text
quality_score = 0.5 * SSIM_full + 0.5 * SSIM_bbox
```

EXP030 validation details:

- best_epoch: 19
- acc4 SSIM_full: 0.918414056886912
- acc4 SSIM_bbox: 0.9352672735107279
- acc8 SSIM_full: 0.8874255976018807
- acc8 SSIM_bbox: 0.9076007461106336
- skipped validation files: []

## Desktop LOCAL probe status

All `LOCAL_` results are exploratory desktop probes only. They are not official VESSL scores, must not be submitted, and must not be treated as final candidates.

### Completed one-epoch campaign

Seventeen one-epoch probes are complete (`LOCAL_EXP012` through `LOCAL_EXP028`). The current top five are:

| Rank | Probe | Config | SSIM_full | SSIM_bbox | quality_score |
|---:|---|---|---:|---:|---:|
| 1 | `LOCAL_EXP018_varnet_c4_ch16_s8_e1` | c4/ch16/s8/e1 | 0.8854211528 | 0.9004961319 | 0.8929586423 |
| 2 | `LOCAL_EXP019_varnet_c4_ch12_s12_e1` | c4/ch12/s12/e1 | 0.8844853268 | 0.8948144457 | 0.8896498863 |
| 3 | `LOCAL_EXP028_varnet_c7_ch12_s8_e1` | c7/ch12/s8/e1 | 0.8843235364 | 0.8948626322 | 0.8895930843 |
| 4 | `LOCAL_EXP021_varnet_c6_ch12_s12_e1` | c6/ch12/s12/e1 | 0.8844415618 | 0.8939631755 | 0.8892023687 |
| 5 | `LOCAL_EXP027_varnet_c6_ch14_s8_e1` | c6/ch14/s8/e1 | 0.8841284133 | 0.8931268794 | 0.8886276464 |

Round 2 (`LOCAL_EXP024`–`LOCAL_EXP028`) completed 5/5 with no failed runs and `skipped=[]` for every probe. The main findings are:

- c7/ch12/s8 led Round 2 but did not beat c4/ch16/s8 overall.
- c4/ch12/s10 improved over the one-epoch c4/ch12/s8 baseline but remained below c4/ch12/s12.
- c4/ch16/s10 and c4/ch16/s12 both underperformed c4/ch16/s8.
- c4/ch14/s8 did not interpolate between ch12 and ch16.

### Adaptive longer-run campaign

The desktop GPU is running a serial matched-comparison campaign:

1. `LOCAL_EXP029 c4/ch12/s8/e5` — active baseline.
2. `LOCAL_EXP032 c4/ch16/s8/e5` — matched candidate.
3. `LOCAL_EXP033 c4/ch12/s8/e1`, seed 431.
4. `LOCAL_EXP034 c4/ch16/s8/e1`, seed 431.
5. `LOCAL_EXP035 c4/ch16/s8/e10` — longer candidate trajectory.

No adaptive metric is reported until its source evaluation finishes. A fail-closed report builder requires all LOCAL sources plus the official EXP031 row before writing a final decision report.

See:

- `reports/local_comparisons/local_probe_summary.md`
- `reports/local_comparisons/local_probe_sweep_20260711_round2.md`
- `reports/local_comparisons/local_probe_adaptive_followup_plan_20260711.json`
- `docs/exp031_post_training_handoff.md`

## Official Phase 2 result

The approved official 30-repeat evaluation completed successfully for
`EXP030_official`. All 30 runs reported the same quality metrics, and run 01
provided both the minimum valid timing and maximum total score.

| Metric | Final value |
|---|---:|
| completed runs | 30 |
| best run | `EXP030_official_run01` |
| SSIM_full | 0.9178 |
| SSIM_bbox | 0.9108 |
| quality_score | 0.9143 |
| minimum time_ms_per_slice | 173.4 |
| time_score | 0.0009513541666666666 |
| total_score | 0.9152513541666667 |

EXP030 remains the final candidate. Its official one-shot total-score advantage
over EXP012 was `0.0062859895833333`, despite EXP030's slower reconstruction.

## Next actions

1. Let VESSL EXP031 finish without Git or GPU interference from the desktop branch.
2. Capture the EXP031 checkpoint identity, best epoch, validation metrics, subgroup counts, and skipped list using `docs/exp031_post_training_handoff.md`.
3. Let the adaptive LOCAL campaign finish its matched e5, seed-431, and e10 comparisons.
4. Run `python scripts/build_exp031_decision_report.py --check`; generate the final decision report only when all sources are present and valid.
5. If c4/ch16/s8 passes the LOCAL gate, propose—but do not automatically launch—a separately approved VESSL experiment.
6. Keep the verified EXP030 organizer package available until an officially validated replacement is selected.

Existing EXP030 organizer artifacts:

- GitHub repository: `https://github.com/caffeine-fighter/2026-FastMRI-SNU`
- Loss graph: `reports/figures/EXP030_varnet_c4_ch12_s8_e20_val_loss.png`
- Model weight: `/root/result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt` (submit separately; never commit it)
- Model description: `reports/phase2/EXP030_model_description.pptx`
- Prepared bundle: `/root/submissions/EXP030_final_fbbddf6.zip`
- Bundle SHA-256: `65b150fb749b772db99e4fde77a636ed58eb19f215e859dbc77cf60ea3aeb18f`

No organizer upload URL or authenticated submission CLI is stored in this repository.

## Safety rules

Do not run these unless explicitly approved for the current step:

- `recon_eval.sh`
- `scripts/run_recon_eval_once.sh`
- `scripts/repeat_recon_eval.sh`
- any training command
- any command that modifies mounted `Data` directories

Do not modify without explicit approval:

- `recon_eval.py`
- `train.py`
- `reconstruct.py`
- model code
- loss code
- transforms/data pipeline code
- official metric functions

Never stage or commit:

- `Data/`, `data/`
- `*.h5`
- `result/`, `results/`, `runs/`, `checkpoints/`, `checkpoints_phase2/`
- `*.pt`, `*.pth`, `*.ckpt`
- `.env`, `.env.local`
- secrets or credentials
