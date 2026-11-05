#!/usr/bin/env bash
# Full 5-fold reproduction: 5 pan-organ + 5 x 24 per-organ clocks, then the cross-fitted fusion.
# Finished runs are skipped, so the script can simply be restarted.
#
#   bash scripts/run_cv.sh <features_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES=$1
for k in 0 1 2 3 4; do
  python train.py --clock pan    --fold $k --features_dir "$FEATURES"
  python train.py --clock organs --fold $k --features_dir "$FEATURES"
done
python fuse.py --ckpt_dir checkpoints --out results
