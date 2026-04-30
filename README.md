# Damp_es: DAMP x CLIP-ES — SYNTHIA to Cityscapes UDA Pipeline

Unified codebase merging **DAMP** (Domain-Aware Mutual Prompting) and **CLIP-ES** (CLIP-based Explicit Segmentation) for unsupervised domain adaptation in semantic segmentation.

Pipeline: train DAMP prompts on SYNTHIA (source) + Cityscapes (target), then generate refined CAM pseudo-masks using CLIP-ES.

```
Download data (HuggingFace)
        |
        v
Prepare data (parquet -> PNG)
        |
        v
Build multilabel.json
        |
        v
DAMP training  ------------->  prompt_learner.pth
                                       |
                                       v
CLIP-ES CAM generation  ----->  refined .npy CAMs
                                       |
                                       v
Eval + DenseCRF  ------------>  pseudo-masks --> segmentation training (external)
```

## Project Layout

```
Damp_es/
├── clip/               # Modified CLIP (variable-resolution + attention extraction)
├── cam/                # CAM generation, Grad-CAM, CRF, evaluation
├── datasets/           # Dassl DatasetBase wrappers (SYNTHIA, GTA5, VOC12, COCO14)
├── trainers/           # DAMP trainer (multi-label + single-label)
├── configs/
│   ├── datasets/       # Dataset YAML configs
│   └── trainers/       # Trainer YAML configs (damp_synthia.yaml, ...)
├── scripts/            # Shell pipelines (pipeline_synthia.sh, ...)
├── tools/              # Data download & preparation utilities
├── tests/              # Unit tests + integration tests
├── train.py            # DAMP training entry point
├── generate_cams.py    # CAM generation entry point
├── eval_cam.py         # CAM evaluation + CRF entry point
└── requirements.txt
```

## Quick Start (Google Colab)

### 1. Clone and install

```bash
!git clone https://github.com/baominh5xx2/Damp_es_CS338.git /content/Damp_es
%cd /content/Damp_es
!pip install -r requirements.txt
!pip install pyarrow joblib
```

### 2. Download and prepare SYNTHIA

```bash
# Download SYNTHIA parquet from HuggingFace
!python tools/download_synthia_hf.py \
    --output-dir /content/synthia_parquet

# Convert parquet to images/labels/splits (parallel, ~9400 images)
!python tools/prepare_synthia_hf.py \
    --parquet-dir /content/synthia_parquet/parquet \
    --output-root /content/data/raw/synthia \
    --num-workers 8
```

### 3. Download and prepare Cityscapes

```bash
# Download + convert Cityscapes from HuggingFace (parallel)
!python tools/prepare_cityscapes_hf.py \
    --dataset-id Chris1/cityscapes \
    --output-root /content/data/raw/cityscapes \
    --splits train,validation \
    --num-workers 8
```

### 4. Build multilabel index (REQUIRED before training)

```bash
!python tools/build_synthia_multilabel.py \
    --split-file /content/data/raw/synthia/splits/train.txt \
    --label-dir /content/data/raw/synthia/labels \
    --output-dir /content/data/processed/synthia_multilabel \
    --num-workers 8
```

> **Without this step, training will only detect 1 class instead of 16.**

### 5. Train DAMP

```bash
!python train.py --config-file configs/trainers/damp_synthia.yaml \
    DATASET.ROOT /content/data \
    OUTPUT_DIR /content/output/damp_synthia
```

Output: `output/damp_synthia/prompt_learner.pth`

### 6. Generate CAMs

```bash
!python generate_cams.py \
    --dataset synthia \
    --img_root /content/data/raw/synthia/images \
    --label_root /content/data/raw/synthia/labels \
    --split_file /content/data/raw/synthia/splits/train.txt \
    --cam_out_dir /content/output/cams_synthia \
    --damp_prompt_ckpt /content/output/damp_synthia/prompt_learner.pth \
    --max_long_side 1024 \
    --skip_existing
```

### 7. Evaluate CAMs

```bash
!python eval_cam.py \
    --dataset synthia \
    --cam_out_dir /content/output/cams_synthia \
    --gt_root /content/data/raw/synthia/labels \
    --split_file /content/data/raw/synthia/splits/train.txt \
    --cam_type attn_highres
```

### 8. CRF post-processing (pseudo-masks)

```bash
!python eval_cam.py \
    --dataset synthia \
    --cam_out_dir /content/output/cams_synthia \
    --gt_root /content/data/raw/synthia/labels \
    --split_file /content/data/raw/synthia/splits/train.txt \
    --cam_type attn_highres \
    --crf \
    --image_root /content/data/raw/synthia/images \
    --mask_output_dir /content/output/synthia/pseudo_masks
```

## One-liner (shell script)

If data is already prepared at `DATA_ROOT`:

```bash
bash scripts/pipeline_synthia.sh /content/data /content/output
```

This runs steps 4-8 automatically.

## Expected Data Layout

After steps 2-3, the directory should look like:

```
/content/data/
├── raw/
│   ├── synthia/
│   │   ├── images/          # train_000000.png ... train_009399.png
│   │   ├── labels/          # same filenames (Cityscapes 19-class train IDs)
│   │   └── splits/
│   │       └── train.txt
│   └── cityscapes/
│       ├── images/          # train_000000.png ... / validation_000000.png ...
│       ├── labels/          # same filenames (raw Cityscapes label IDs)
│       └── splits/
│           ├── train.txt
│           └── val.txt
└── processed/
    └── synthia_multilabel/
        └── multilabel.json  # {filename: [cit19_ids]} - built by step 4
```

## SYNTHIA 16-class Label Mapping

SYNTHIA uses 16 of Cityscapes' 19 classes (missing: terrain, truck, train):

| Local ID | Class Name     | Cityscapes-19 Train ID |
|----------|----------------|------------------------|
| 0        | road           | 0                      |
| 1        | sidewalk       | 1                      |
| 2        | building       | 2                      |
| 3        | wall           | 3                      |
| 4        | fence          | 4                      |
| 5        | pole           | 5                      |
| 6        | traffic light  | 6                      |
| 7        | traffic sign   | 7                      |
| 8        | vegetation     | 8                      |
| 9        | sky            | 10                     |
| 10       | person         | 11                     |
| 11       | rider          | 12                     |
| 12       | car            | 13                     |
| 13       | bus            | 15                     |
| 14       | motorcycle     | 17                     |
| 15       | bicycle        | 18                     |

## Tests

```bash
# Unit tests (no GPU needed, no torch needed)
python -m pytest tests/test_synthia_labels.py -v

# Integration tests (needs torch + dassl, CPU OK)
python -m pytest tests/test_synthia_mock_pipeline.py -v -s
```

## Other Datasets

The codebase also supports VOC12, COCO14, and GTA5:

```bash
# VOC12
bash scripts/pipeline_voc12.sh /path/to/data

# COCO14
bash scripts/pipeline_coco14.sh /path/to/data

# GTA5
bash scripts/pipeline_gta5.sh /path/to/data
```
