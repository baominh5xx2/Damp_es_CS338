# Phase 6: Evaluation & Post-Processing

## Goal
Port and unify all evaluation scripts (CAM eval, CRF post-processing) from CLIP-ES into the project, supporting all datasets.

## Source Files
- `CLIP-ES/eval_cam.py` — VOC12/COCO14 CAM evaluation
- `CLIP-ES/eval_cam_with_crf.py` — CRF post-processing + evaluation
- `CLIP-ES/eval_cam_gta5.py` — GTA5 evaluation
- `CLIP-ES/eval_cam_cityscapes.py` — Cityscapes evaluation

## Target Structure
```
eval_cam.py               # Unified evaluation CLI
cam/
├── evaluate.py           # Evaluation logic (all datasets)
├── crf.py                # DenseCRF post-processing
└── ...
```

## Tasks

### 6.1 Port DenseCRF module
- Extract `DenseCRF` class from `CLIP-ES/eval_cam_with_crf.py` → `cam/crf.py`
- Key parameters:
  - Unary energy from CAM probabilities
  - Gaussian kernel: spatial smoothness (sx=3, sy=3)
  - Bilateral kernel: color smoothness (sx=40, sy=40, sr=10, sg=10, sb=10)
  - 10 inference iterations

### 6.2 Port evaluation logic
- Merge 3 eval scripts into `cam/evaluate.py`:
  - `eval_cam.py` → VOC12/COCO14 logic
  - `eval_cam_gta5.py` → GTA5/Cityscapes logic (19 classes, ID mapping)
- Unified `evaluate_cams()` function with dataset parameter:
  ```python
  def evaluate_cams(dataset, split, cam_dir, crf=False,
                    mask_output_dir=None, threshold=None):
  ```
- Metrics for all datasets:
  - Pixel Accuracy, Mean Accuracy, Mean IoU, Class IoU
  - Confidence masking (pixels < 0.95 confidence → ignore label 255)

### 6.3 Create unified eval CLI
- `eval_cam.py` at project root:
  ```
  python eval_cam.py --dataset voc12 --split val --cam-dir output/cams/voc12_damp
  python eval_cam.py --dataset gta5 --split val --cam-dir output/cams/gta5_damp --crf
  ```
- Arguments:
  - `--dataset`: voc12 | coco14 | gta5 | cityscapes
  - `--split`: val | test
  - `--cam-dir`: directory containing .npy CAM files
  - `--crf`: enable CRF post-processing
  - `--mask-output-dir`: save pseudo-masks for segmentation training
  - `--bg-threshold`: background score threshold for grid search
  - `--confidence-threshold`: confidence masking threshold (default 0.95)

### 6.4 Background scoring
- VOC12/COCO14: `bg_score = (1 - max(cams))^threshold`
- GTA5/Cityscapes: optional background channel from text embeddings of scene elements
- Grid search over threshold values for optimal mIoU

### 6.5 ID mapping for Cityscapes/GTA5
- Port `ID_TO_TRAINID` mapping:
  ```python
  id_to_trainid = {
      7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
      19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
      25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
  }
  ```
- Apply mapping during evaluation for GTA5/Cityscapes

## Verification
- `python eval_cam.py --dataset voc12 --split val --cam-dir <path>` reports mIoU
- `python eval_cam.py --dataset gta5 --split val --cam-dir <path> --crf` saves CRF masks
- mIoU values match original CLIP-ES results
