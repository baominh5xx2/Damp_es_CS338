#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# End-to-end pipeline for VOC12: DAMP training → CAM generation → Evaluation
# ============================================================================
# Usage:
#   bash scripts/pipeline_voc12.sh /path/to/data
#
# Expected data layout:
#   $DATA_ROOT/VOC2012/
#       JPEGImages/
#       Annotations/
#       SegmentationClass/
# ============================================================================

DATA_ROOT="${1:?Usage: $0 <DATA_ROOT>}"
OUTPUT_DIR="${2:-output}"

VOC_ROOT="${DATA_ROOT}/VOC2012"
IMG_ROOT="${VOC_ROOT}/JPEGImages"
GT_ROOT="${VOC_ROOT}/SegmentationClass"
SPLIT_DIR="datasets/splits/voc12"
TRAIN_SPLIT="${SPLIT_DIR}/train.txt"
VAL_SPLIT="${SPLIT_DIR}/val.txt"

DAMP_DIR="${OUTPUT_DIR}/damp/voc12"
CAM_DIR="${OUTPUT_DIR}/voc12/cams_damp"
MASK_DIR="${OUTPUT_DIR}/voc12/pseudo_masks"

echo "============================================"
echo " VOC12 Pipeline"
echo " DATA_ROOT:  ${DATA_ROOT}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo "============================================"

# ---- Step 1: Train DAMP prompts ----
echo ""
echo "[Step 1/4] Training DAMP prompt learner ..."
python train.py \
    --config-file configs/trainers/damp_voc12.yaml \
    DATASET.ROOT "${DATA_ROOT}" \
    OUTPUT_DIR "${DAMP_DIR}"

PROMPT_CKPT="${DAMP_DIR}/prompt_learner.pth"
if [ ! -f "${PROMPT_CKPT}" ]; then
    echo "ERROR: prompt_learner.pth not found at ${PROMPT_CKPT}"
    exit 1
fi
echo "  => Prompt checkpoint: ${PROMPT_CKPT}"

# ---- Step 2: Generate CAMs ----
echo ""
echo "[Step 2/4] Generating CAMs with DAMP prompts ..."
python generate_cams.py \
    --dataset voc12 \
    --img_root "${IMG_ROOT}" \
    --split_file "${TRAIN_SPLIT}" \
    --cam_out_dir "${CAM_DIR}" \
    --damp_prompt_ckpt "${PROMPT_CKPT}" \
    --skip_existing

# ---- Step 3: Evaluate CAMs (threshold grid search) ----
echo ""
echo "[Step 3/4] Evaluating CAMs (threshold grid search) ..."
python eval_cam.py \
    --dataset voc12 \
    --cam_out_dir "${CAM_DIR}" \
    --gt_root "${GT_ROOT}" \
    --split_file "${TRAIN_SPLIT}" \
    --cam_type attn_highres

# ---- Step 4: CRF post-processing → pseudo-masks ----
echo ""
echo "[Step 4/4] CRF post-processing → pseudo-masks ..."
python eval_cam.py \
    --dataset voc12 \
    --cam_out_dir "${CAM_DIR}" \
    --gt_root "${GT_ROOT}" \
    --split_file "${TRAIN_SPLIT}" \
    --cam_type attn_highres \
    --crf \
    --image_root "${IMG_ROOT}" \
    --mask_output_dir "${MASK_DIR}"

echo ""
echo "============================================"
echo " VOC12 Pipeline complete."
echo " Pseudo-masks: ${MASK_DIR}"
echo "============================================"
