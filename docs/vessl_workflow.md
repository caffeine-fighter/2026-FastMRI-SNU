# VESSL workflow

## Paths

| Item | Path |
|---|---|
| Repository | `/root/2026-FastMRI-SNU` |
| Data | `/root/Data` |
| Training data | `/root/Data/train` |
| Validation data | `/root/Data/val` |
| Results | `/root/result/<EXP_NAME>` |

Mounted data is read-only. Results and checkpoints stay outside Git.

## Before a run

```bash
cd /root/2026-FastMRI-SNU
git status -sb
nvidia-smi
python scripts/check_submission.py
```

Confirm that no other training or evaluation process is using the GPU.

## Training

```bash
python -u train.py \
  -b 1 \
  -e <EPOCHS> \
  -l 0.001 \
  -r 10 \
  -n <EXP_NAME> \
  -t /root/Data/train/ \
  -v /root/Data/val/ \
  --cascade <CASCADES> \
  --chans <CHANNELS> \
  --sens_chans <SENS_CHANNELS> \
  --seed 430
```

Use an `EXP###_...` name on VESSL. Record the command and result in `experiments/experiment_log.csv` after review.

## Validation

Training writes the best validation reconstructions to `../result/<EXP_NAME>/reconstructions_val`.

```bash
python scripts/evaluate_val.py \
  --exp-name <EXP_NAME> \
  --target-dir /root/Data/val/image \
  --recon-dir ../result/<EXP_NAME>/reconstructions_val \
  --out-dir ../result/<EXP_NAME>/metrics

python scripts/plot_loss.py \
  --loss-log ../result/<EXP_NAME>/val_loss_log.npy \
  --out ../result/<EXP_NAME>/metrics/val_loss.png
```

Review `metrics.json`, `metrics.csv`, and `skipped.json`. Compare candidates with:

```text
quality = 0.5 * SSIM_full + 0.5 * SSIM_bbox
```

## Official evaluation

Run only when training is finished and the candidate is approved:

```bash
bash scripts/set_phase2_candidate.sh \
  <TAG> <CHECKPOINT> <CASCADES> <CHANNELS> <SENS_CHANNELS> '<NOTE>'

bash scripts/phase2_preflight.sh
bash scripts/run_recon_eval_once.sh <RUN_TAG>
```

The official entrypoint remains:

```bash
bash recon_eval.sh
```

Do not modify `recon_eval.py`, mounted data, or official metric code.
