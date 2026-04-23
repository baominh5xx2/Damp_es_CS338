# Phase 7: Scripts & Configs

## Goal
Create all shell scripts and config files for the unified project, covering training, CAM generation, evaluation, and the end-to-end pipeline.

## Target Structure
```
scripts/
├── pipeline_voc12.sh        # Full pipeline: DAMP → CAM → eval
├── pipeline_gta5.sh         # Full pipeline for GTA5→Cityscapes
├── pipeline_coco14.sh       # Full pipeline for COCO14
├── train_damp_voc12.sh      # DAMP training only
├── train_damp_gta5.sh       # DAMP training only
├── train_damp_coco14.sh     # DAMP training only
├── generate_cams_voc12.sh   # CAM generation only
├── generate_cams_gta5.sh    # CAM generation only
├── generate_cams_coco14.sh  # CAM generation only
└── eval_cams.sh             # Evaluation only

configs/
├── trainers/
│   └── damp.yaml            # DAMP trainer config
└── datasets/
    ├── voc12.yaml
    ├── gta5.yaml
    ├── coco14.yaml          # New
    ├── office_home.yaml
    ├── visda17.yaml
    └── mini_domainnet.yaml
```

## Tasks

### 7.1 Port DAMP dataset configs
- Copy from `DAMP/configs/datasets/`:
  - `voc12.yaml` — train_aug → val, ViT-B/16
  - `gta5.yaml` — gta5_train → cityscapes_val, ViT-B/16
  - `office_home.yaml` — all 12 pairs, RN50
  - `visda17.yaml` — synthetic → real, ViT-B/16
  - `miniDomainNet.yaml` → rename to `mini_domainnet.yaml`
- Create new:
  - `coco14.yaml` — train → val, ViT-B/16

### 7.2 Port DAMP trainer config
- Copy `DAMP/configs/trainers/DAMP/damp.yaml` → `configs/trainers/damp.yaml`
- Key settings:
  ```yaml
  DATALOADER:
    TRAIN_X: {BATCH_SIZE: 32}
    TRAIN_U: {BATCH_SIZE: 32}
    TEST: {BATCH_SIZE: 128}
  OPTIM:
    NAME: adam
    LR: 3e-3
    MAX_EPOCH: 30
    LR_SCHEDULER: cosine
  OPTIM_C:
    NAME: adam
    LR: 3e-3
  TRAINER:
    DAMP:
      PREC: amp
      N_CTX: 32
      N_CLS: 2
      TAU: 0.5
      U: 1.0
  ```

### 7.3 Create DAMP training scripts
- Port from `DAMP/scripts/`:
  - `voc12.sh` → `train_damp_voc12.sh` (TAU=0.5, U=1.0)
  - `gta5.sh` → `train_damp_gta5.sh` (TAU=0.85, U=0.5, auto-download)
  - `VisDA17.sh` → `train_damp_visda17.sh` (TAU=0.5, U=2.0)
  - `office_home.sh` → `train_damp_office_home.sh` (TAU=0.6, U=1.0, all 12 pairs)
  - `miniDomainNet.sh` → `train_damp_mini_domainnet.sh` (TAU=0.5, U=1.0, all 12 pairs)
- Update paths to new structure

### 7.4 Create CAM generation scripts
- New scripts based on original CLIP-ES logic:
  - `generate_cams_voc12.sh`: ViT-B/16, box_threshold=0.4
  - `generate_cams_gta5.sh`: ViT-B/16, box_threshold=0.4, max_long_side=1024
  - `generate_cams_coco14.sh`: ViT-B/16, box_threshold=0.7
- Each supports `--damp_prompt_ckpt` flag

### 7.5 Create pipeline scripts
- End-to-end scripts that chain: DAMP train → CAM generate → eval
- Example `pipeline_voc12.sh`:
  ```bash
  #!/bin/bash
  SEED=1
  OUTPUT_ROOT="output"

  # Train DAMP
  python train.py --config-file configs/datasets/voc12.yaml \
    --trainer DAMP --seed $SEED \
    --output-dir $OUTPUT_ROOT/damp/voc12/seed$SEED

  # Generate CAMs with DAMP prompts
  python generate_cams.py --dataset voc12 --split trainval \
    --damp_prompt_ckpt $OUTPUT_ROOT/damp/voc12/seed$SEED/prompt_learner.pth \
    --output-dir $OUTPUT_ROOT/cams/voc12_damp/seed$SEED

  # Evaluate with CRF
  python eval_cam.py --dataset voc12 --split val \
    --cam-dir $OUTPUT_ROOT/cams/voc12_damp/seed$SEED \
    --crf --mask-output-dir $OUTPUT_ROOT/masks/voc12_damp/seed$SEED
  ```

### 7.6 Create evaluation-only script
- `scripts/eval_cams.sh`: Generic eval script with dataset/cam-dir args

## Verification
- All scripts run without path errors
- Config files are valid YAML
- Pipeline scripts execute end-to-end
