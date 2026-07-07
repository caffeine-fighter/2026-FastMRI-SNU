# VESSL workflow

## Core rule

Final candidate models must be trained end-to-end on VESSL.

## Default paths

- Repo: /root/2026-FastMRI-SNU
- Data: /root/Data
- Train data: /root/Data/train
- Validation data: /root/Data/val
- Results: ../result/<exp_name>

## System dependencies

The metric code imports OpenCV through `cv2`. On a clean Linux image, install the
OpenCV shared-library dependencies before running evaluation if they are missing:

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

## EXP000 smoke command

```bash
python train.py \
  -b 1 \
  -e 1 \
  -l 0.001 \
  -r 10 \
  -n EXP000_smoke_varnet_c1_ch9_s4_e1 \
  -t /root/Data/train/ \
  -v /root/Data/val/ \
  --cascade 1 \
  --chans 9 \
  --sens_chans 4 \
  --seed 430
```

## EXP001 baseline command

```bash
python train.py \
  -b 1 \
  -e 5 \
  -l 0.001 \
  -r 10 \
  -n EXP001_baseline_varnet_c1_ch9_s4_e5 \
  -t /root/Data/train/ \
  -v /root/Data/val/ \
  --cascade 1 \
  --chans 9 \
  --sens_chans 4 \
  --seed 430
```

## Evaluation command

```bash
python scripts/evaluate_val.py \
  --exp-name EXP000_smoke_varnet_c1_ch9_s4_e1 \
  --target-dir /root/Data/val/image \
  --recon-dir ../result/EXP000_smoke_varnet_c1_ch9_s4_e1/reconstructions_val \
  --out-dir ../result/EXP000_smoke_varnet_c1_ch9_s4_e1/metrics
```

## If VESSL start fails with resource quota exceeded

- Do not create repeated duplicate workspaces.
- Do not delete the Data volume.
- Keep the local GitHub branch ready.
- Report the issue with workspace name, server/node, team, and screenshots.
- Continue local code preparation until VESSL is available.
