# Phase 1: Project Scaffolding & Shared CLIP Code

## Goal
Set up the unified project structure and port the shared CLIP model code that both DAMP and CLIP-ES depend on.

## Target Structure
```
Damp_es/
├── clip/                    # Shared CLIP model (from CLIP-ES, enhanced)
│   ├── __init__.py
│   ├── clip.py              # load(), tokenize(), _transform()
│   ├── model.py             # VisionTransformer, CLIP, upsample_pos_emb()
│   └── simple_tokenizer.py  # BPE tokenizer
├── datasets/                # Shared dataset loaders
├── trainers/                # DAMP trainer
├── cam/                     # CLIP-ES CAM generation & grad-cam
├── configs/                 # All config files
├── scripts/                 # Shell scripts
├── tools/                   # Data preparation utilities
├── docs/                    # Documentation
├── train.py                 # Main training entry (DAMP)
├── generate_cams.py         # Main CAM generation entry (CLIP-ES)
├── eval_cam.py              # CAM evaluation
├── requirements.txt
└── README.md
```

## Tasks

### 1.1 Create directory structure
- Create all directories listed above with `__init__.py` files

### 1.2 Port CLIP model code
- Copy `CLIP-ES/clip/` → `Damp_es/clip/`
- This is the authoritative CLIP code (includes ViT attention extraction, `forward_last_layer()`, `upsample_pos_emb()`)
- Verify DAMP's CLIP usage is compatible (DAMP uses CLIP via `import clip` — same API surface)
- Key functions both repos need:
  - `clip.load()` — load model + preprocessing
  - `clip.tokenize()` — text tokenization
  - `model.encode_image()` — with attention weights (CLIP-ES needs this)
  - `model.encode_text()` — standard text encoding
  - `model.forward_last_layer()` — CAM-specific (CLIP-ES only)

### 1.3 Create requirements.txt
- Merge dependencies from both repos:
  - From DAMP: dassl, yacs, timm, datasets, huggingface_hub
  - From CLIP-ES: pydensecrf, lxml, ttach, ftfy, regex
  - Common: torch, torchvision, numpy, opencv, pillow, tqdm, matplotlib, scikit_learn

### 1.4 Verify CLIP compatibility
- DAMP's `TextEncoder` wraps `clip_model.transformer` directly
- CLIP-ES's `model.encode_image()` returns `(features, attn_weights_list)`
- Ensure the shared `clip/model.py` supports both use cases
- No changes expected — DAMP accesses submodules directly, CLIP-ES uses the top-level API

## Verification
- `python -c "import clip; model, preprocess = clip.load('ViT-B/16', device='cpu')"` runs without error
- All CLIP submodules accessible: `model.visual`, `model.transformer`, `model.token_embedding`
