# Phase 5: End-to-End Pipeline Integration (DAMP → CLIP-ES)

## Goal
Create the unified end-to-end pipeline: train DAMP prompts → use them for CLIP-ES CAM generation → evaluate pseudo-masks. This is the key value of merging the two repos.

## Pipeline Flow
```
DAMP Training
    ↓ (saves prompt_learner.pth)
CAM Generation with DAMP Prompts
    ↓ (saves refined CAMs as .npy)
CAM Evaluation + CRF Post-processing
    ↓ (saves pseudo-masks)
Segmentation Model Training (external, e.g. DeepLabV2)
```

## Tasks

### 5.1 Unified pipeline script
- Create `scripts/pipeline.sh` that runs the full pipeline:
  ```bash
  # Step 1: Train DAMP
  python train.py --config-file configs/datasets/voc12.yaml --trainer DAMP \
    --seed 1 --output-dir output/damp/voc12

  # Step 2: Generate CAMs with DAMP prompts
  python generate_cams.py --dataset voc12 --split trainval \
    --damp_prompt_ckpt output/damp/voc12/prompt_learner.pth \
    --output-dir output/cams/voc12_damp

  # Step 3: Evaluate CAMs
  python eval_cam.py --dataset voc12 --split val \
    --cam-dir output/cams/voc12_damp

  # Step 4: CRF + evaluate
  python eval_cam.py --dataset voc12 --split val \
    --cam-dir output/cams/voc12_damp --crf --mask-output-dir output/masks/voc12_damp
  ```

### 5.2 DAMP → CLIP-ES checkpoint bridge
- In `trainers/damp.py`, after saving best model, also save:
  ```python
  # Extract prompt learner weights for CLIP-ES
  prompt_state = self.model.prompt_learner.state_dict()
  torch.save(prompt_state, os.path.join(self.output_dir, 'prompt_learner.pth'))
  ```
- In `cam/generate.py`, load DAMP prompt:
  ```python
  def load_damp_prompts(ckpt_path, n_ctx, clip_model):
      state = torch.load(ckpt_path)
      # Reconstruct prompt learner to get text embeddings
      prompt_learner = PromptLearner(n_ctx, clip_model)
      prompt_learner.load_state_dict(state)
      # Get learned context vectors
      ctx = prompt_learner.ctx  # [n_ctx, dim]
      return ctx
  ```
- When generating text embeddings with DAMP prompts:
  ```python
  # Naive: "a clean origami {class_name}."
  # DAMP:  [SOS] + [learned CTX] + [CLASS] + [EOS]
  if damp_ctx is not None:
      tokens = construct_prompt_with_ctx(class_names, damp_ctx, clip_model)
      text_features = model.encode_text(tokens)
  ```

### 5.3 Dataset-specific pipeline scripts
Create `scripts/pipeline_<dataset>.sh` for each dataset:

**VOC12** (`scripts/pipeline_voc12.sh`):
- DAMP: train_aug → val, ViT-B/16, TAU=0.5, U=1.0
- CAM: trainval split, box_threshold=0.4
- Eval: val split, CRF enabled

**GTA5→Cityscapes** (`scripts/pipeline_gta5.sh`):
- DAMP: gta5_train → cityscapes_val, ViT-B/16, TAU=0.85, U=0.5
- CAM: GTA5 images, max_long_side=1024
- Eval: Cityscapes val, 19 classes

**COCO14** (`scripts/pipeline_coco14.sh`):
- DAMP: train → val (multi-label classification)
- CAM: train split, box_threshold=0.7
- Eval: val split

### 5.4 Output directory convention
```
output/
├── damp/
│   ├── voc12/seed1/           # DAMP training outputs
│   │   ├── model-best.pth.tar
│   │   ├── prompt_learner.pth  # Extracted for CLIP-ES
│   │   └── log.txt
│   ├── gta5/seed1/
│   └── coco14/seed1/
├── cams/
│   ├── voc12_damp/            # CAMs with DAMP prompts
│   ├── voc12_naive/           # CAMs without DAMP (baseline)
│   ├── gta5_damp/
│   └── coco14_damp/
└── masks/
    ├── voc12_damp/            # CRF-processed pseudo-masks
    ├── gta5_damp/
    └── coco14_damp/
```

## Verification
- Run `scripts/pipeline_voc12.sh` end-to-end
- DAMP training completes and saves `prompt_learner.pth`
- CAM generation loads DAMP prompts and produces .npy files
- Evaluation reports mIoU metrics
- Compare mIoU: with DAMP prompts vs without (baseline)
