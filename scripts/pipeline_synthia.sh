#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# End-to-end pipeline for SYNTHIA→Cityscapes UDA (16-class):
#   DAMP training → CAM generation → Evaluation → CRF pseudo-masks
# ============================================================================
# Usage:
#   bash scripts/pipeline_synthia.sh /path/to/data
#
# Expected data layout:
#   $DATA_ROOT/data/raw/synthia/
#       images/   labels/   splits/train.txt
#   $DATA_ROOT/data/raw/cityscapes/
#       images/   labels/   splits/{train,val}.txt
# ============================================================================

DATA_ROOT="${1:?Usage: $0 <DATA_ROOT>}"
OUTPUT_DIR="${2:-output}"

SYNTHIA_ROOT="${DATA_ROOT}/data/raw/synthia"
CITY_ROOT="${DATA_ROOT}/data/raw/cityscapes"

SYNTHIA_IMG="${SYNTHIA_ROOT}/images"
SYNTHIA_LBL="${SYNTHIA_ROOT}/labels"
SYNTHIA_SPLIT="${SYNTHIA_ROOT}/splits/train.txt"

CITY_IMG="${CITY_ROOT}/images"
CITY_LBL="${CITY_ROOT}/labels"
CITY_VAL_SPLIT="${CITY_ROOT}/splits/val.txt"

DAMP_DIR="${OUTPUT_DIR}/damp/synthia"
CAM_DIR="${OUTPUT_DIR}/synthia/cams_damp"
MASK_DIR="${OUTPUT_DIR}/synthia/pseudo_masks"

echo "============================================"
echo " SYNTHIA → Cityscapes Pipeline (16-class)"
echo " DATA_ROOT:  ${DATA_ROOT}"
echo " OUTPUT_DIR: ${OUTPUT_DIR}"
echo "============================================"

# ---- Step 0: Build multilabel.json for SYNTHIA ----
echo ""
echo "[Step 0] Building multilabel.json for SYNTHIA ..."
MULTILABEL_DIR="${DATA_ROOT}/data/processed/synthia_multilabel"
python tools/build_synthia_multilabel.py \
    --split-file "${SYNTHIA_SPLIT}" \
    --label-dir "${SYNTHIA_LBL}" \
    --output-dir "${MULTILABEL_DIR}"

# ---- Step 1: Train DAMP prompts (SYNTHIA → Cityscapes) ----
echo ""
echo "[Step 1/4] Training DAMP prompt learner ..."
python train.py \
    --config-file configs/trainers/damp_synthia.yaml \
    DATASET.ROOT "${DATA_ROOT}" \
    OUTPUT_DIR "${DAMP_DIR}"

PROMPT_CKPT="${DAMP_DIR}/prompt_learner.pth"
if [ ! -f "${PROMPT_CKPT}" ]; then
    echo "ERROR: prompt_learner.pth not found at ${PROMPT_CKPT}"
    exit 1
fi
echo "  => Prompt checkpoint: ${PROMPT_CKPT}"

# ---- Step 2: Generate CAMs on SYNTHIA train set ----
echo ""
echo "[Step 2/4] Generating CAMs on SYNTHIA images ..."
python generate_cams.py \
    --dataset synthia \
    --img_root "${SYNTHIA_IMG}" \
    --label_root "${SYNTHIA_LBL}" \
    --split_file "${SYNTHIA_SPLIT}" \
    --cam_out_dir "${CAM_DIR}" \
    --damp_prompt_ckpt "${PROMPT_CKPT}" \
    --max_long_side 1024 \
    --skip_existing

# ---- Step 3: Evaluate CAMs ----
echo ""
echo "[Step 3/4] Evaluating CAMs (threshold grid search) ..."
python eval_cam.py \
    --dataset synthia \
    --cam_out_dir "${CAM_DIR}" \
    --gt_root "${SYNTHIA_LBL}" \
    --split_file "${SYNTHIA_SPLIT}" \
    --cam_type attn_highres

# ---- Step 4: CRF post-processing → pseudo-masks ----
echo ""
echo "[Step 4/4] CRF post-processing → pseudo-masks ..."
python eval_cam.py \
    --dataset synthia \
    --cam_out_dir "${CAM_DIR}" \
    --gt_root "${SYNTHIA_LBL}" \
    --split_file "${SYNTHIA_SPLIT}" \
    --cam_type attn_highres \
    --crf \
    --image_root "${SYNTHIA_IMG}" \
    --mask_output_dir "${MASK_DIR}"

echo ""
echo "============================================"
echo " SYNTHIA Pipeline complete."
echo " Pseudo-masks: ${MASK_DIR}"
echo "============================================"
