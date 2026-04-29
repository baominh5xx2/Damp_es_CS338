#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# End-to-end pipeline for COCO14: DAMP training → CAM generation → Evaluation
# ============================================================================
# Usage:
#   bash scripts/pipeline_coco14.sh /path/to/data
#
# Expected data layout:
#   $DATA_ROOT/coco14/
#       train2014/  (COCO_train2014_*.jpg)
#       val2014/    (COCO_val2014_*.jpg)
# ============================================================================

DATA_ROOT="${1:?Usage: $0 <DATA_ROOT>}"
OUTPUT_DIR="${2:-output}"

COCO_ROOT="${DATA_ROOT}/coco14"
IMG_ROOT="${COCO_ROOT}/train2014"
SPLIT_DIR="datasets/splits/coco14"
TRAIN_SPLIT="${SPLIT_DIR}/train.txt"

DAMP_DIR="${OUTPUT_DIR}/damp/coco14"
CAM_DIR="${OUTPUT_DIR}/coco14/cams_damp"
MASK_DIR="${OUTPUT_DIR}/coco14/pseudo_masks"

# COCO14 does not ship with dense GT masks for the train split;
# skip mIoU evaluation if no gt_root is given.
GT_ROOT="${3:-}"

echo "============================================"
echo " COCO14 Pipeline"
echo " DATA_ROOT:  ${DATA_ROOT}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo "============================================"

# ---- Step 1: Train DAMP prompts ----
echo ""
echo "[Step 1/3] Training DAMP prompt learner ..."
python train.py \
    --config-file configs/trainers/damp_coco14.yaml \
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
echo "[Step 2/3] Generating CAMs with DAMP prompts ..."
python generate_cams.py \
    --dataset coco14 \
    --img_root "${IMG_ROOT}" \
    --split_file "${TRAIN_SPLIT}" \
    --cam_out_dir "${CAM_DIR}" \
    --damp_prompt_ckpt "${PROMPT_CKPT}" \
    --skip_existing

# ---- Step 3: Evaluate (if GT masks are available) ----
if [ -n "${GT_ROOT}" ] && [ -d "${GT_ROOT}" ]; then
    echo ""
    echo "[Step 3/3] Evaluating CAMs ..."
    python eval_cam.py \
        --dataset coco14 \
        --cam_out_dir "${CAM_DIR}" \
        --gt_root "${GT_ROOT}" \
        --split_file "${TRAIN_SPLIT}" \
        --cam_type attn_highres
else
    echo ""
    echo "[Step 3/3] Skipping evaluation (no --gt_root provided)."
fi

echo ""
echo "============================================"
echo " COCO14 Pipeline complete."
echo " CAMs: ${CAM_DIR}"
echo "============================================"
