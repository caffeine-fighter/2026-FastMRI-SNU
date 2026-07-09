#!/usr/bin/env bash
set -euo pipefail

echo "=== Phase 2 preflight ==="

echo ""
echo "=== identity ==="
hostname
whoami
pwd

echo ""
echo "=== git ==="
git branch --show-current || true
git status -sb || true

echo ""
echo "=== required files ==="
for f in recon_eval.py recon_eval.sh utils/learning/test_part.py scripts/phase2_score.py; do
  if [ -f "$f" ]; then
    echo "OK: $f"
  else
    echo "MISSING: $f"
    exit 1
  fi
done

echo ""
echo "=== recon_eval.py modification check ==="
if ! git diff --quiet -- recon_eval.py; then
  echo "ERROR: recon_eval.py has unstaged local modifications. Phase 2 rules say do not modify recon_eval.py."
  git diff -- recon_eval.py | sed -n '1,120p'
  exit 1
fi
if ! git diff --cached --quiet -- recon_eval.py; then
  echo "ERROR: recon_eval.py has staged local modifications. Phase 2 rules say do not modify recon_eval.py."
  git diff --cached -- recon_eval.py | sed -n '1,120p'
  exit 1
fi
echo "OK: recon_eval.py has no local diff"

echo ""
echo "=== candidate checkpoint ==="
if [ -e checkpoints_phase2/best_model.pt ]; then
  ls -lah checkpoints_phase2/best_model.pt
else
  echo "ERROR: checkpoints_phase2/best_model.pt missing"
  echo "Run scripts/set_phase2_candidate.sh first. This script should create a symlink, not copy weights."
  exit 1
fi

echo ""
echo "=== leaderboard Data path check ==="
DATA_ROOT="${FASTMRI_DATA_ROOT:-}"

if [ -z "$DATA_ROOT" ]; then
  for p in /root/Data /home/ubuntu/Data "$HOME/fastmri_data/Data"; do
    if [ -d "$p/leaderboard" ]; then
      DATA_ROOT="$p"
      break
    fi
  done
fi

if [ -z "$DATA_ROOT" ]; then
  echo "WARNING: leaderboard Data root not found in default locations."
  echo "This is expected on some desktop-only prep sessions, but VESSL recon_eval requires mounted leaderboard data."
else
  echo "DATA_ROOT=$DATA_ROOT"
  find "$DATA_ROOT" -maxdepth 4 -type d | sort | head -120

  echo ""
  echo "leaderboard file counts:"
  printf "acc4/kspace "; find "$DATA_ROOT/leaderboard/acc4/kspace" -type f -name "*.h5" 2>/dev/null | wc -l
  printf "acc8/kspace "; find "$DATA_ROOT/leaderboard/acc8/kspace" -type f -name "*.h5" 2>/dev/null | wc -l
  printf "acc4/image  "; find "$DATA_ROOT/leaderboard/acc4/image" -type f -name "*.h5" 2>/dev/null | wc -l
  printf "acc8/image  "; find "$DATA_ROOT/leaderboard/acc8/image" -type f -name "*.h5" 2>/dev/null | wc -l
fi

echo ""
echo "=== forbidden staged files ==="
STAGED_FORBIDDEN=$(git diff --cached --name-only | grep -E '(^|/)Data/|(^|/)data/|\.h5$|\.pt$|\.pth$|\.ckpt$|(^|/)result/|(^|/)results/|(^|/)runs/|(^|/)checkpoints/|(^|/)checkpoints_phase2/|(^|/)\.env$|(^|/)\.env\.local$' || true)
if [ -n "$STAGED_FORBIDDEN" ]; then
  echo "ERROR: forbidden files staged:"
  echo "$STAGED_FORBIDDEN"
  exit 1
fi
echo "OK: no forbidden staged files"

echo ""
echo "Preflight OK."
