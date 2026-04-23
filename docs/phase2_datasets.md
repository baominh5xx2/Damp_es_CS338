# Phase 2: Unified Dataset Module

## Goal
Merge dataset loaders from both repos into a single `datasets/` module. Both repos handle VOC12, GTA5, Cityscapes but differently — DAMP uses Dassl's `DatasetBase`, CLIP-ES reads split files directly.

## Source Files
- `DAMP/custom_datasets/gta5.py` — GTA5/Cityscapes as Dassl DatasetBase
- `DAMP/custom_datasets/voc12.py` — VOC12 as Dassl DatasetBase
- `CLIP-ES/voc12/*.txt` — VOC12 split files
- `CLIP-ES/coco14/*.txt` — COCO14 split files
- `DAMP/tools/build_gta5_multilabel.py` — multilabel annotation builder
- `DAMP/tools/download_gta5_hf.py` — GTA5 downloader
- `DAMP/tools/prepare_cityscapes_hf.py` — Cityscapes preparer

## Target Structure
```
datasets/
├── __init__.py
├── voc12.py          # VOC12 (Dassl DatasetBase)
├── gta5.py           # GTA5/Cityscapes (Dassl DatasetBase)
├── coco14.py         # COCO14 (new Dassl DatasetBase, from CLIP-ES splits)
├── visda17.py        # VisDA17 (use Dassl built-in)
├── office_home.py    # Office-Home (use Dassl built-in)
├── mini_domainnet.py # Mini-DomainNet (use Dassl built-in)
└── splits/           # All split files
    ├── voc12/
    │   ├── train.txt
    │   ├── trainval.txt
    │   ├── train_aug.txt
    │   └── val.txt
    └── coco14/
        ├── train.txt
        └── val.txt
```

## Tasks

### 2.1 Port split files
- Copy `CLIP-ES/voc12/*.txt` → `datasets/splits/voc12/`
- Copy `CLIP-ES/coco14/*.txt` → `datasets/splits/coco14/`

### 2.2 Port VOC12 dataset
- Copy `DAMP/custom_datasets/voc12.py` → `datasets/voc12.py`
- Update split file paths to use `datasets/splits/voc12/`
- Keep Dassl `DatasetBase` inheritance
- Keep both `CLASS_NAMES` and `PROMPT_CLASS_NAMES` (needed by CLIP-ES)

### 2.3 Port GTA5/Cityscapes dataset
- Copy `DAMP/custom_datasets/gta5.py` → `datasets/gta5.py`
- Verify it works with Dassl registry
- Keep multilabel JSON loading support

### 2.4 Create COCO14 dataset (new)
- Create `datasets/coco14.py` as Dassl `DatasetBase`
- Use split format from CLIP-ES: `image_id class_id1 class_id2 ...`
- 80 classes with prompt names from `CLIP-ES/clip_text.py`
- Domains: `["train", "val"]`

### 2.5 Port data tools
- Copy `DAMP/tools/` → `Damp_es/tools/`
- Update paths in tools to reference new structure

### 2.6 Register datasets
- In `datasets/__init__.py`, import all dataset classes
- Dassl `@DATASET_REGISTRY.register()` handles auto-registration

## Key Design Decisions
- **Dassl DatasetBase for all datasets**: Unified interface, compatible with DAMP trainer
- **CLIP-ES reads splits directly**: When generating CAMs, we'll add a utility function that reads split files without needing Dassl — this is for inference-only use
- **Prompt class names stored in dataset**: Each dataset defines `PROMPT_CLASS_NAMES` for CLIP text encoding

## Verification
- `python -c "from datasets import VOC12, GTA5, COCO14"` imports successfully
- Dassl registry finds all datasets
- Split files are accessible from new paths
