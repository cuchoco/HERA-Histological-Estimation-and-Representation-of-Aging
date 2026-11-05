#!/usr/bin/env bash
# Tissue segmentation, 224 px patching at 20x and UNI2-h patch embeddings with TRIDENT
# (https://github.com/mahmoodlab/TRIDENT). The same settings are used for GTEx and external slides.
#
#   bash scripts/extract_features.sh <wsi_dir> <job_dir>
#
#   seg   : HEST segmenter at 10x, holes inside the tissue mask kept as tissue (no --remove_holes)
#   coords: non-overlapping 224 x 224 px patches at 20x (~0.5 um/px)
#   feat  : UNI2-h (uni_v2), TRIDENT eval transform, bfloat16
#
# Output: <job_dir>/20x_224px_0px_overlap/features_uni_v2/<slide>.h5  (dataset 'features', N x 1536)
# UNI2-h is gated on Hugging Face (MahmoodLab/UNI2-h); log in with `huggingface-cli login` first.
set -euo pipefail
WSI_DIR=$1
JOB_DIR=$2
TRIDENT=${TRIDENT:-TRIDENT}          # path to a TRIDENT checkout
GPU=${GPU:-0}

python "$TRIDENT/run_batch_of_slides.py" --task seg    --wsi_dir "$WSI_DIR" --job_dir "$JOB_DIR" \
    --gpu "$GPU" --segmenter hest --skip_errors
python "$TRIDENT/run_batch_of_slides.py" --task coords --wsi_dir "$WSI_DIR" --job_dir "$JOB_DIR" \
    --mag 20 --patch_size 224 --overlap 0 --skip_errors
python "$TRIDENT/run_batch_of_slides.py" --task feat   --wsi_dir "$WSI_DIR" --job_dir "$JOB_DIR" \
    --gpu "$GPU" --mag 20 --patch_size 224 --patch_encoder uni_v2 --skip_errors
