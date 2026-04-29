# Damp_es: DAMP × CLIP-ES Unified Pipeline

Unified codebase merging:
- **DAMP** — *Domain-Aware Mutual Prompting* (semi-/unsupervised domain adaptation, CoOp-style learnable prompts).
- **CLIP-ES** — *CLIP-based Explicit Segmentation* CAM generation pipeline for weakly-supervised semantic segmentation.

The repo lets you (1) train DAMP prompts on a labeled+unlabeled dataset, then (2) plug those learned prompts into the CLIP-ES Grad-CAM + attention-affinity refinement pipeline to obtain higher-quality pseudo-masks.

```
DAMP training  ─────────────►  prompt_learner.pth
                                       │
                                       ▼
CLIP-ES CAM generation  ─────►  refined .npy CAMs
                                       │
                                       ▼
Eval + DenseCRF  ─────────────►  pseudo-masks  ─► segmentation training (external)
```

## Layout
```
Damp_es/
├── clip/             # Modified CLIP (CLIP-ES variant: variable-resolution + attention extraction)
├── datasets/         # Dassl DatasetBase wrappers (VOC12, COCO14, GTA5, ...) + split files
├── trainers/         # DAMP trainer (multi-label & single-label)
├── cam/              # CLIP-ES CAM generation, Grad-CAM, CRF, evaluation  (added in phase 4-6)
├── configs/          # YAML configs (trainer + datasets)
├── scripts/          # Shell entry points
├── tools/            # Data prep utilities (GTA5 / Cityscapes)
├── docs/             # Phase-by-phase implementation notes
├── train.py          # DAMP training entry point
└── requirements.txt
```

## Status
- [x] Phase 1 — Scaffolding & shared CLIP code
- [x] Phase 2 — Unified dataset module
- [x] Phase 3 — DAMP trainer (multi-label + single-label)
- [ ] Phase 4 — CLIP-ES CAM generation
- [ ] Phase 5 — End-to-end pipeline
- [ ] Phase 6 — Evaluation & CRF
- [ ] Phase 7 — Scripts & configs (full)
- [ ] Phase 8 — Testing

## Quickstart
```bash
pip install -r requirements.txt

# Train DAMP on VOC12 (multi-label)
python train.py --root $DATA --trainer DAMP \
    --dataset-config-file configs/datasets/voc12.yaml \
    --config-file configs/trainers/damp.yaml \
    --output-dir output/damp/voc12/seed1 --seed 1

# Train DAMP on Office-Home (single-label)
python train.py --root $DATA --trainer DAMP \
    --dataset-config-file configs/datasets/office_home.yaml \
    --config-file configs/trainers/damp.yaml \
    --source-domains art --target-domains clipart \
    --output-dir output/damp/oh_a2c/seed1 --seed 1
```

After training, `output/.../prompt_learner.pth` is exported automatically — this is the bridge file consumed by the CLIP-ES CAM pipeline (phase 4).
