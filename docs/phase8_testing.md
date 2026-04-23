# Phase 8: Testing & Verification

## Goal
Verify the entire unified project works end-to-end: training, CAM generation, evaluation, and the pipeline.

## Test Matrix

### 8.1 Unit Tests

**CLIP module**:
- [ ] `import clip` works
- [ ] `clip.load('ViT-B/16')` loads model
- [ ] `clip.load('RN50')` loads model
- [ ] `clip.tokenize(["a photo of a dog."])` tokenizes correctly
- [ ] `model.encode_image()` returns features + attention weights
- [ ] `model.encode_text()` returns text embeddings
- [ ] `model.forward_last_layer()` returns CAM-ready activations

**Dataset module**:
- [ ] `from datasets import VOC12, GTA5, COCO14` imports
- [ ] Dassl registry finds all datasets
- [ ] VOC12 loads train_aug + val splits correctly
- [ ] GTA5 loads with multilabel JSON annotations
- [ ] Split files found at new paths

**Grad-CAM module**:
- [ ] `from cam.grad_cam import GradCAM` imports
- [ ] `GradCAM.forward()` produces valid CAM for ViT
- [ ] Attention reshape works for ViT-B/16

### 8.2 Integration Tests

**DAMP Training**:
- [ ] `python train.py --config-file configs/datasets/voc12.yaml --trainer DAMP --seed 1` runs 1 epoch
- [ ] `prompt_learner.pth` saved in output directory
- [ ] `model-best.pth.tar` saved in output directory
- [ ] Loss decreases over epoch

**CAM Generation (without DAMP)**:
- [ ] `python generate_cams.py --dataset voc12 --split trainval` generates .npy files
- [ ] CAM shape matches image dimensions
- [ ] All present classes have activation maps

**CAM Generation (with DAMP prompts)**:
- [ ] `python generate_cams.py --dataset voc12 --damp_prompt_ckpt <path>` runs
- [ ] DAMP prompts loaded and applied to text embeddings
- [ ] CAMs differ from baseline (no DAMP) — should be better quality

**Evaluation**:
- [ ] `python eval_cam.py --dataset voc12 --split val --cam-dir <path>` reports mIoU
- [ ] `python eval_cam.py --dataset gta5 --split val --cam-dir <path>` handles 19 classes
- [ ] CRF post-processing runs and saves masks

### 8.3 End-to-End Pipeline Test

Run `scripts/pipeline_voc12.sh` with minimal settings:
- [ ] DAMP trains for 1 epoch (override MAX_EPOCH=1 for testing)
- [ ] prompt_learner.pth generated
- [ ] CAMs generated using DAMP prompts
- [ ] Evaluation reports mIoU
- [ ] CRF masks saved for segmentation training

### 8.4 Regression Tests

Compare outputs against original repos:
- [ ] DAMP training loss matches original `DAMP/train.py` (same seed)
- [ ] CAM generation produces same .npy files as `CLIP-ES/generate_cams_voc12.py`
- [ ] Evaluation mIoU matches original `CLIP-ES/eval_cam.py`

### 8.5 Edge Cases

- [ ] Large images (GTA5 1914×1052) handled with `--max_long_side`
- [ ] Multi-label datasets (VOC12, COCO14) work for both DAMP and CAM
- [ ] Single-label datasets (Office-Home, VisDA17) work for DAMP
- [ ] Missing DAMP checkpoint falls back to naive text embeddings gracefully
- [ ] CUDA and CPU modes both work

## Success Criteria
- All unit tests pass
- End-to-end pipeline runs without errors
- mIoU with DAMP prompts ≥ mIoU without (baseline)
- No regression compared to original repos
