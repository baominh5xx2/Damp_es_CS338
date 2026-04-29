#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# End-to-end pipeline for GTA5→Cityscapes UDA:
#   DAMP training → CAM generation → Evaluation → CRF pseudo-masks
# ============================================================================
# Usage:
#   bash scripts/pipeline_gta5.sh /path/to/data
#
# Expected data layout:
#   $DATA_ROOT/data/raw/gta5_kaggle_full/
#       images/   labels/   splits/train_full.txt
#   $DATA_ROOT/data/raw/cityscapes/
#       images/   labels/   splits/{train,val}.txt
# ============================================================================

DATA_ROOT="${1:?Usage: $0 <DATA_ROOT>}"
OUTPUT_DIR="${2:-output}"

GTA5_ROOT="${DATA_ROOT}/data/raw/gta5_kaggle_full"
CITY_ROOT="${DATA_ROOT}/data/raw/cityscapes"

GTA5_IMG="${GTA5_ROOT}/images"
GTA5_LBL="${GTA5_ROOT}/labels"
GTA5_SPLIT="${GTA5_ROOT}/splits/train_full.txt"

CITY_IMG="${CITY_ROOT}/images"
CITY_LBL="${CITY_ROOT}/labels"
CITY_VAL_SPLIT="${CITY_ROOT}/splits/val.txt"

DAMP_DIR="${OUTPUT_DIR}/damp/gta5"
CAM_DIR="${OUTPUT_DIR}/gta5/cams_damp"
MASK_DIR="${OUTPUT_DIR}/gta5/pseudo_masks"

echo "============================================"
echo " GTA5 → Cityscapes Pipeline"
echo " DATA_ROOT:  ${DATA_ROOT}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo "============================================"

# ---- Step 1: Train DAMP prompts (GTA5 → Cityscapes) ----
echo ""
echo "[Step 1/4] Training DAMP prompt learner ..."
python train.py \
    --config-file configs/trainers/damp_gta5.yaml \
    DATASET.ROOT "${DATA_ROOT}" \
    OUTPUT_DIR "${DAMP_DIR}"

PROMPT_CKPT="${DAMP_DIR}/prompt_learner.pth"
if [ ! -f "${PROMPT_CKPT}" ]; then
    echo "ERROR: prompt_learner.pth not found at ${PROMPT_CKPT}"
    exit 1
fi
echo "  => Prompt checkpoint: ${PROMPT_CKPT}"

# ---- Step 2: Generate CAMs on GTA5 train set ----
echo ""
echo "[Step 2/4] Generating CAMs on GTA5 images ..."
python generate_cams.py \
    --dataset gta5 \
    --img_root "${GTA5_IMG}" \
    --label_root "${GTA5_LBL}" \
    --split_file "${GTA5_SPLIT}" \
    --cam_out_dir "${CAM_DIR}" \
    --damp_prompt_ckpt "${PROMPT_CKPT}" \
    --max_long_side 1024 \
    --skip_existing

# ---- Step 3: Evaluate CAMs ----
echo ""
echo "[Step 3/4] Evaluating CAMs (threshold grid search) ..."
python eval_cam.py \
    --dataset gta5 \
    --cam_out_dir "${CAM_DIR}" \
    --gt_root "${GTA5_LBL}" \
    --split_file "${GTA5_SPLIT}" \
    --cam_type attn_highres

# ---- Step 4: CRF post-processing → pseudo-masks ----
echo ""
echo "[Step 4/4] CRF post-processing → pseudo-masks ..."
python eval_cam.py \
    --dataset gta5 \
    --cam_out_dir "${CAM_DIR}" \
    --gt_root "${GTA5_LBL}" \
    --split_file "${GTA5_SPLIT}" \
    --cam_type attn_highres \
    --crf \
    --image_root "${GTA5_IMG}" \
    --mask_output_dir "${MASK_DIR}"

echo ""
echo "============================================"
echo " GTA5 Pipeline complete."
echo " Pseudo-masks: ${MASK_DIR}"
echo "============================================"
