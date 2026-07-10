# Archived EXP030 VESSL runbook

> Historical execution record. EXP030 training, validation, official evaluation, and GitHub delivery are complete. Use [`../vessl_workflow.md`](../vessl_workflow.md) for current commands.

This runbook was used after `EXP030_varnet_c4_ch12_s8_e20` finished training on VESSL.

Do **not** run these VESSL commands while `EXP030` is still training. If any `EXP030` training process is still present, stop and wait.

## 0. Enter the VESSL workspace

```bash
cd /root/FastMRI_challenge  # adjust only if the VESSL repo path differs
pwd
hostname
git branch --show-current
git status -sb
```

## 1. Check the `EXP030` process

Copy-paste:

```bash
ps -ef | grep -E 'EXP030|train.py|python' | grep -v grep || true
nvidia-smi
```

Stop here if any `EXP030` training process is still running.

## 2. Check `EXP030` artifacts

Copy-paste:

```bash
EXP=EXP030_varnet_c4_ch12_s8_e20
ls -lah ../result/$EXP
ls -lah ../result/$EXP/checkpoints
ls -lah ../result/$EXP/checkpoints/best_model.pt
ls -lah ../result/$EXP/val_loss_log.npy || true
ls -lah ../result/$EXP/reconstructions_val 2>/dev/null | sed -n '1,120p' || true
find ../result/$EXP -maxdepth 3 -type f | sort | sed -n '1,200p'
```

Do not copy checkpoints into git. Do not stage weights, reconstructions, or mounted data.

## 3. Run validation evaluation

First inspect the current CLI:

```bash
python scripts/evaluate_val.py --help
```

Then run validation evaluation. Use the repository's actual `evaluate_val.py` flags if the help text differs from this template:

```bash
python scripts/evaluate_val.py \
  --recon-dir ../result/EXP030_varnet_c4_ch12_s8_e20/reconstructions_val \
  --target-dir /root/Data/val/image \
  --out-json ../result/EXP030_varnet_c4_ch12_s8_e20/metrics/metrics.json \
  --out-csv ../result/EXP030_varnet_c4_ch12_s8_e20/metrics/metrics.csv
```

Record:

- `val_loss`
- `SSIM_full`
- `SSIM_bbox`
- `SSIM_full_acc4`
- `SSIM_bbox_acc4`
- `SSIM_full_acc8`
- `SSIM_bbox_acc8`
- `skipped.json` status
- checkpoint path

## 4. Run validation loss plot

First inspect the CLI:

```bash
python scripts/plot_loss.py --help
```

Template:

```bash
python scripts/plot_loss.py \
  --loss ../result/EXP030_varnet_c4_ch12_s8_e20/val_loss_log.npy \
  --out reports/figures/EXP030_varnet_c4_ch12_s8_e20_val_loss.png
```

If `plot_loss.py --help` shows different flags, use that help text and keep the output under `reports/figures/`.

## 5. Print `metrics.csv` and `skipped.json`

Copy-paste:

```bash
EXP=EXP030_varnet_c4_ch12_s8_e20
printf '\n=== metrics.csv ===\n'
sed -n '1,120p' ../result/$EXP/metrics/metrics.csv
printf '\n=== skipped.json ===\n'
if [ -f ../result/$EXP/skipped.json ]; then
  python -m json.tool ../result/$EXP/skipped.json
elif [ -f ../result/$EXP/metrics/skipped.json ]; then
  python -m json.tool ../result/$EXP/metrics/skipped.json
else
  echo 'skipped.json not found'
fi
```

## 6. Compare `EXP030` against `EXP012`

Known `EXP012` reference:

```text
EXP012_varnet_c4_ch12_s4_e10
val_loss      = 3.2876096990602717
SSIM_full     = 0.8994141339351495
SSIM_bbox     = 0.9187541341189271
quality_score = 0.9090841340270383
```

Copy-paste this comparison helper after `EXP030` metrics are available:

```bash
python - <<'PY'
import csv
from pathlib import Path

exp012_quality = 0.9090841340270383
exp012_full = 0.8994141339351495
exp012_bbox = 0.9187541341189271
exp012_loss = 3.2876096990602717

metrics_path = Path('../result/EXP030_varnet_c4_ch12_s8_e20/metrics/metrics.csv')
rows = list(csv.DictReader(metrics_path.open()))

def pick(row, *names):
    by_lower = {k.lower(): k for k in row}
    for name in names:
        key = by_lower.get(name.lower())
        if key is not None and row[key] != '':
            return float(row[key])
    raise KeyError(names)

# Prefer an overall row if metrics.csv is row-oriented.
overall = None
for row in rows:
    label = ' '.join(str(v).lower() for v in row.values())
    if 'overall' in label or 'all' in label:
        overall = row
        break
if overall is None:
    overall = rows[0]

full = pick(overall, 'ssim_full_mean', 'ssim_full', 'full')
bbox = pick(overall, 'ssim_bbox_mean', 'ssim_bbox', 'bbox')
quality = 0.5 * full + 0.5 * bbox

print(f'EXP030 SSIM_full={full:.16f}')
print(f'EXP030 SSIM_bbox={bbox:.16f}')
print(f'EXP030 quality_score={quality:.16f}')
print(f'EXP012 SSIM_full={exp012_full:.16f}')
print(f'EXP012 SSIM_bbox={exp012_bbox:.16f}')
print(f'EXP012 quality_score={exp012_quality:.16f}')
print(f'delta_quality_EXP030_minus_EXP012={quality - exp012_quality:.16f}')
print(f'EXP012 val_loss={exp012_loss:.16f}')
PY
```

Decision rule:

- Prefer higher validation quality unless Phase 2 timing is materially worse.
- If validation quality is effectively tied, compare official Phase 2 `ms/slice` after repeated wrapper runs.
- Never select a `LOCAL_` checkpoint as final.

## 7. Record `EXP030` in `experiments/experiment_log.csv`

Append one reviewed row with exact metrics.

CSV header:

```text
exp_id,date,machine,branch,commit,config,seed,command,status,val_loss,ssim_full,ssim_bbox,ssim_full_acc4,ssim_bbox_acc4,ssim_full_acc8,ssim_bbox_acc8,checkpoint_path,notes
```

Use:

```text
exp_id: EXP030
config: EXP030_varnet_c4_ch12_s8_e20
checkpoint_path: ../result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt
notes: include skipped.json status and whether this beats EXP012
```

Before staging, check forbidden paths:

```bash
git status -sb
git diff -- experiments/experiment_log.csv docs/current_state.md docs/archive/vessl_after_exp030_runbook.md | sed -n '1,240p'
git diff --cached --name-only | grep -E '(^|/)Data/|(^|/)data/|\.h5$|(^|/)result/|(^|/)results/|(^|/)runs/|(^|/)checkpoints/|(^|/)checkpoints_phase2/|\.pt$|\.pth$|\.ckpt$|(^|/)\.env$|(^|/)\.env\.local$' && echo 'FORBIDDEN STAGED FILE' && exit 1 || true
```

## 8. Commit and push validation metrics

Only after reviewing exact diffs:

```bash
git add experiments/experiment_log.csv docs/current_state.md docs/archive/vessl_after_exp030_runbook.md reports/figures/EXP030_varnet_c4_ch12_s8_e20_val_loss.png
python scripts/check_submission.py
git status -sb
git commit -m "record EXP030 validation metrics"
git push
```

Do not stage or commit checkpoints, `.h5`, mounted data, `result/`, `runs/`, `checkpoints/`, `checkpoints_phase2/`, `.env`, or secrets.

## 9. Only then merge `phase2/eval-wrapper`

Only after EXP030 metrics are safely committed/pushed and the VESSL workspace is clean:

```bash
git status -sb
git switch phase2/eval-wrapper
git merge <metrics-branch-or-main>
python scripts/check_submission.py
git status -sb
```

Do not merge while `EXP030` training is active.

## 10. Only then run official `recon_eval`

Run official Phase 2 evaluation only after:

- `EXP030` training has finished.
- Validation metrics are recorded.
- The candidate checkpoint exists.
- `scripts/phase2_preflight.sh` passes.
- Mounted leaderboard `Data` exists.
- The user approves running official evaluation.

### Candidate `EXP012`

```bash
bash scripts/set_phase2_candidate.sh \
  EXP012 \
  ../result/EXP012_varnet_c4_ch12_s4_e10/checkpoints/best_model.pt \
  4 12 4 \
  'EXP012 completed reference: quality_score=0.9090841340270383'

bash scripts/phase2_preflight.sh
bash scripts/run_recon_eval_once.sh EXP012_phase2_once
bash scripts/repeat_recon_eval.sh EXP012_phase2 30
```

### Candidate `EXP030`

```bash
bash scripts/set_phase2_candidate.sh \
  EXP030 \
  ../result/EXP030_varnet_c4_ch12_s8_e20/checkpoints/best_model.pt \
  4 12 8 \
  'EXP030 candidate after completed VESSL training'

bash scripts/phase2_preflight.sh
bash scripts/run_recon_eval_once.sh EXP030_phase2_once
bash scripts/repeat_recon_eval.sh EXP030_phase2 30
```

Review `reports/phase2/repeat_*` and compare `EXP012` vs `EXP030` by:

- `SSIM_full`
- `SSIM_bbox`
- `quality_score`
- `ms/slice`
- `time_score`
- `total_score`
