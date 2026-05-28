# Experiment Setup: SYNTHIA -> Cityscapes DAMP/CLIP-ES

This file is the source-of-truth summary for the current low-resource experiment in `pipeline_main.ipynb`.
It records model config, data splits, evaluation protocol, outputs, and the places where final numbers should be copied after a run.

Last local reference checked:

- Repo: `https://github.com/baominh5xx2/Damp_es_CS338.git`
- Branch: `master`
- Commit checked locally: `d96395c`
- Main notebook: `pipeline_main.ipynb`
- Training config: `configs/trainers/damp_synthia_fast.yaml`
- Dataset config: `configs/datasets/synthia.yaml`
- Legacy shell pipeline: `scripts/pipeline_synthia.sh` is older and not the final notebook flow.

## 1. Experiment Goal

Task: unsupervised domain adaptation from SYNTHIA to Cityscapes for semantic segmentation.

The project has three practical stages:

1. Stage 1: train DAMP as a multi-label classification / prompt-learning model.
2. Stage 2: generate and evaluate CAMs from zero-shot CLIP-ES, DAMP prompt-only, DAMP full, and hybrid CAMs.
3. Stage 3: convert CAMs to pseudo-masks, evaluate pseudo-mask mIoU, and export image/mask pairs for downstream segmentation training.

Important distinction:

- Stage 1 metric is image-level multi-label classification quality on Cityscapes val.
- Stage 2 metric is CAM/pseudo-label mIoU against SYNTHIA GT on the selected SYNTHIA subset.
- Stage 3 metric is PNG pseudo-mask mIoU against SYNTHIA GT on the same selected SYNTHIA subset.
- Downstream segmentation mIoU is separate and only exists after training a segmentation model from the exported pairs.

## 2. Runtime and Repo Paths

Main Colab paths from `pipeline_main.ipynb`:

| Name | Value |
|---|---|
| `REPO_DIR` | `/content/Damp_es` |
| `REPO_URL` | `https://github.com/baominh5xx2/Damp_es_CS338.git` |
| `DATA_ROOT` | `/content/drive/MyDrive/datasets/synthia_cs338` |
| `OUTPUT_DIR` | `/content/drive/MyDrive/datasets/synthia_cs338/output` |
| `RUN_NAME` | `synthia_clipnorm_tau052_e3` |
| `CAM_MAX_IMAGES` | `1000` |
| `EVAL_MAX_IMAGES` | `1000` |
| `GRID_SEARCH_MAX_IMAGES` | `100` legacy knob; current Cell 7 uses fixed protocols on full eval subset |
| `USE_CRF` | `False` |
| `CRF_CONFIDENCE` | `0.95` |
| `CRF_N_JOBS` | `1` |

Notebook installs:

```bash
pip install -q timm yacs ftfy regex lxml ttach
pip install -q opencv-python-headless scikit-learn matplotlib tqdm
pip install -q datasets huggingface_hub pyarrow
pip install -q git+https://github.com/KaiyangZhou/Dassl.pytorch.git
```

## 3. Data Sources and Layout

HuggingFace sources from the current notebook:

| Dataset | HF repo |
|---|---|
| SYNTHIA | `Minhbao5xx2/synthia-rand-cityscapes-16class-parquet_fix` |
| Cityscapes | `Chris1/cityscapes` |

Expected raw layout under `DATA_ROOT`:

```text
data/raw/synthia/
  images/
  labels/
  splits/train.txt

data/raw/cityscapes/
  images/
  labels/
  splits/train.txt
  splits/val.txt
```

Processed multi-label files:

```text
data/processed/synthia_multilabel/multilabel.json
data/processed/cityscapes_multilabel/train_multilabel.json
data/processed/cityscapes_multilabel/val_multilabel.json
```

Current notebook preparation tools:

```bash
python tools/prepare_synthia_hf.py --parquet-dir <DATA_ROOT>/synthia_parquet/parquet --output-root <DATA_ROOT>/data/raw/synthia
python tools/prepare_cityscapes_hf.py --dataset-id Chris1/cityscapes --output-root <DATA_ROOT>/data/raw/cityscapes --splits train,validation
python tools/build_synthia_multilabel.py --split-file <SYNTHIA_SPLIT> --label-dir <SYNTHIA_LBL> --output-dir <DATA_ROOT>/data/processed/synthia_multilabel
python tools/build_cityscapes_multilabel.py --split-file <CITY_TRAIN_SPLIT> --label-dir <CITY_LBL> --output-dir <DATA_ROOT>/data/processed/cityscapes_multilabel --output-file train_multilabel.json
python tools/build_cityscapes_multilabel.py --split-file <CITY_VAL_SPLIT> --label-dir <CITY_LBL> --output-dir <DATA_ROOT>/data/processed/cityscapes_multilabel --output-file val_multilabel.json
```

## 4. Splits

### Stage 1: DAMP Training

Defined by `configs/trainers/damp_synthia_fast.yaml` and `datasets/synthia.py`.

| Role | Domain | Split file | Labels used? | Purpose |
|---|---|---|---|---|
| Source train `train_x` | `synthia_train` | `data/raw/synthia/splits/train.txt` | Yes, via multi-label map from segmentation masks | supervised multi-label loss |
| Target train `train_u` | `cityscapes_train` | `data/raw/cityscapes/splits/train.txt` | No target supervision; labels only exist for metadata/eval utilities | pseudo-label loss |
| Test/eval | `cityscapes_val` if present | `data/raw/cityscapes/splits/val.txt` | Multi-label map for reporting | Stage 1 image-level multi-label evaluation |

Dataset code falls back to target domains for test only if `cityscapes_val` is absent.

### Stage 2 and Stage 3: CAM/Pseudo-mask Subset

The final notebook creates an explicit subset split:

```text
data/raw/synthia/splits/train_first500.txt
```

This file is written from the first `CAM_MAX_IMAGES=500` entries of:

```text
data/raw/synthia/splits/train.txt
```

The same split is used for:

- zero-shot CAM generation
- DAMP prompt-only CAM generation
- DAMP full CAM generation
- hybrid CAM construction
- Stage 2 CAM mIoU
- Stage 3 pseudo-mask mIoU
- exported segmentation train pairs
- qualitative overlays

This is important: Stage 2/3 numbers are not a held-out segmentation test. They are source-domain pseudo-label quality checks on the selected SYNTHIA subset.

## 5. Class Space and Label Mapping

The evaluation space is Cityscapes-19 train IDs, with SYNTHIA using only 16 of those IDs.

Cityscapes-19 names:

| ID | Class | Used by SYNTHIA |
|---:|---|---|
| 0 | road | yes |
| 1 | sidewalk | yes |
| 2 | building | yes |
| 3 | wall | yes |
| 4 | fence | yes |
| 5 | pole | yes |
| 6 | traffic light | yes |
| 7 | traffic sign | yes |
| 8 | vegetation | yes |
| 9 | terrain | no |
| 10 | sky | yes |
| 11 | person | yes |
| 12 | rider | yes |
| 13 | car | yes |
| 14 | truck | no |
| 15 | bus | yes |
| 16 | train | no |
| 17 | motorcycle | yes |
| 18 | bicycle | yes |

SYNTHIA local 16-class to Cityscapes-19 train ID mapping:

| SYNTHIA local ID | Class | Cityscapes train ID |
|---:|---|---:|
| 0 | road | 0 |
| 1 | sidewalk | 1 |
| 2 | building | 2 |
| 3 | wall | 3 |
| 4 | fence | 4 |
| 5 | pole | 5 |
| 6 | traffic light | 6 |
| 7 | traffic sign | 7 |
| 8 | vegetation | 8 |
| 9 | sky | 10 |
| 10 | person | 11 |
| 11 | rider | 12 |
| 12 | car | 13 |
| 13 | bus | 15 |
| 14 | motorcycle | 17 |
| 15 | bicycle | 18 |

Evaluation uses `map_mask_to_synthia16()`:

- raw Cityscapes label IDs are remapped to train IDs when needed;
- labels outside the SYNTHIA-valid set are converted to `255`;
- `255` is ignored in metrics.

## 6. Stage 1: DAMP Training Config

Current config file:

```text
configs/trainers/damp_synthia_fast.yaml
```

Training command from notebook:

```bash
python train.py \
  --config-file configs/trainers/damp_synthia_fast.yaml \
  --trainer DAMP \
  DATASET.ROOT /content/drive/MyDrive/datasets/synthia_cs338 \
  OUTPUT_DIR /content/drive/MyDrive/datasets/synthia_cs338/output/damp/synthia_clipnorm_tau052_e3
```

Dataset config:

| Key | Value |
|---|---|
| `DATASET.NAME` | `SYNTHIA` |
| `DATASET.SOURCE_DOMAINS` | `synthia_train` |
| `DATASET.TARGET_DOMAINS` | `cityscapes_train` |

Model config:

| Key | Value |
|---|---|
| Backbone | CLIP `ViT-B/16` |
| CLIP encoders | frozen |
| Prompt learner | trainable |
| Context decoder | trainable |
| `TRAINER.NAME` | `DAMP` |
| `TRAINER.DAMP.N_CTX` | `16` |
| `TRAINER.DAMP.CSC` | `false` |
| `TRAINER.DAMP.PREC` | `amp` |
| `TRAINER.DAMP.TAU` | `0.52` |
| `TRAINER.DAMP.PSEUDO_TEMP` | `0.0` |
| `TRAINER.DAMP.U` | `1.0` |
| Seed | `1` |

Input config:

| Key | Value |
|---|---|
| Input size | `224 x 224` |
| Interpolation | `bicubic` |
| Weak/source transform | `random_resized_crop`, `random_flip`, `normalize` |
| Strong transform | `random_flip`, `randaugment`, `normalize` |
| Pixel mean | `[0.48145466, 0.4578275, 0.40821073]` |
| Pixel std | `[0.26862954, 0.26130258, 0.27577711]` |

Dataloader config:

| Loader | Batch size | Sampler |
|---|---:|---|
| `TRAIN_X` | 32 | `RandomSampler` |
| `TRAIN_U` | 32 | `RandomSampler` |
| `TEST` | 256 | default sequential |
| `NUM_WORKERS` | 16 | - |

Optimizer config:

| Optimizer | Target module | LR | Epochs | Scheduler | Warmup |
|---|---|---:|---:|---|---|
| `OPTIM` | prompt learner | `0.0008` | `3` | cosine | linear, 1 epoch, min LR `1e-5` |
| `OPTIM_C` | context decoder | `0.0008` | `3` | cosine | linear, 1 epoch, min LR `1e-5` |

Training/logging:

| Key | Value |
|---|---|
| `TRAIN.PRINT_FREQ` | 20 |
| `TRAIN.CHECKPOINT_FREQ` | 1 |
| `TRAIN.COUNT_ITER` | `train_x` |
| `TEST.SPLIT` | `test` |
| `TEST.NO_TEST` | `false` |

Stage 1 output:

```text
output/damp/synthia_clipnorm_tau052_e3/
  prompt_learner/model-best.pth.tar
  context_decoder/model-best.pth.tar
  prompt_learner.pth
```

`prompt_learner.pth` is the bridge artifact for CAM generation. It includes both `prompt_learner` state and `context_decoder` state.

## 7. Stage 1 Pseudo-label Meaning

During DAMP training, pseudo-labels are online target-domain image-level labels for Cityscapes train images.

The code computes target probabilities from both:

- current DAMP logits;
- naive zero-shot logits.

Then it mixes them by epoch progress and thresholds the result:

```text
pseudo_label = mix_lambda * sigmoid(output_u / pseudo_temp)
             + (1 - mix_lambda) * sigmoid(pseudo_label_logits / pseudo_temp)

pseudo_bin = pseudo_label >= TRAINER.DAMP.TAU
confident = pseudo_bin.sum(dim=1) > 0
```

With `PSEUDO_TEMP=0.0`, the implementation divides logits by CLIP logit scale, usually near 100, before sigmoid. This makes `TAU=0.52` act like a probability threshold instead of a threshold on saturated CLIP logits.

Logged pseudo fields:

| Log field | Meaning |
|---|---|
| `pseudo_pos_rate` | percent of class positions marked positive after thresholding |
| `pseudo_classes` | average number of positive classes per target image |
| `pseudo_prob` | average soft pseudo probability before hard threshold |
| `pseudo_temp` | actual temperature used for pseudo probabilities |
| `pseudo_conf_rate` | percent of target images with at least one positive pseudo class |

## 8. Stage 2: CAM Generation Config

Current CAM split:

```text
data/raw/synthia/splits/train_first500.txt
```

Common CAM generation settings:

| Setting | Value |
|---|---|
| Dataset | `synthia` |
| Image root | `data/raw/synthia/images` |
| Label root | `data/raw/synthia/labels` |
| Split file | `data/raw/synthia/splits/train_first500.txt` |
| CAM type saved/evaluated | `attn_highres` |
| `MAX_LONG_SIDE` | `1024` |
| `NUM_CAM_WORKERS` | `1` |
| `skip_existing` | enabled |

CAM variants:

| Kind | Output directory | Key flags |
|---|---|---|
| `zero` | `output/synthia/cams_zero_raw_500` | `--cam_score softmax` |
| `prompt_only` | `output/synthia/cams_damp_synthia_clipnorm_tau052_e3_prompt_only_raw_500` | `--damp_prompt_ckpt <prompt_learner.pth> --damp_name_mode train --damp_disable_decoder --cam_score raw` |
| `damp_full` | `output/synthia/cams_damp_synthia_clipnorm_tau052_e3_full_raw_500` | `--damp_prompt_ckpt <prompt_learner.pth> --damp_name_mode train --cam_score raw` |
| `hybrid` | `output/synthia/cams_hybrid_zero_prompt_synthia_clipnorm_tau052_e3_500` | per-class merge of zero-shot and prompt-only CAMs |

DAMP full means the checkpoint uses both learned prompt tokens and the context decoder. Prompt-only explicitly disables the decoder.

Each CAM `.npy` file stores:

```python
{
    "keys": np.array([...]),          # Cityscapes-19 train IDs present in the image
    "attn_highres": np.array([...]),  # shape: (num_present_classes, H, W), float16/float32
}
```

## 10. Stage 2 Evaluation Protocol

Cell 7 evaluates all CAM kinds on the same `EVAL_MAX_IMAGES=500` entries.

Kinds evaluated:

```python
CAM_DIR_BY_KIND = {
    "zero": CAM_ZERO_DIR,
    "prompt_only": CAM_PROMPT_DIR,
    "damp_full": CAM_FULL_DIR,
    "hybrid": CAM_HYBRID_DIR,
}
```

Selectable kinds for auto-pick:

```python
SELECTABLE_CAM_KINDS = ("prompt_only", "damp_full", "hybrid")
```

`zero` is report-only and is not allowed to be auto-selected as the final DAMP-family source.

Current Cell 7 scoring protocols:

| Method | Protocols |
|---|---|
| `baseline` | raw CAM + flat background threshold `0.01`, `0.03`, `0.10`, `0.20` |
| `norm` | per-class normalized CAM + flat background threshold `0.01`, `0.03`, `0.10`, `0.20` |
| `boost` | normalized CAM + adaptive background: `(thres, alpha)` = `(0.10,0.5)`, `(0.20,0.5)`, `(0.50,0.5)`, `(0.10,1.0)` |
| `no_bg` | argmax over foreground classes, no background/ignore channel |

Parallelism knobs:

| Key | Value |
|---|---:|
| `CELL7_LOAD_WORKERS` | 32 |
| `CELL7_GRID_WORKERS` | 32 |
| `CELL7_SCORE_WORKERS` | 32 |

Strict zero-shot report:

| Key | Value |
|---|---|
| `ZERO_STRICT_ENABLED` | `True` |
| `ZERO_STRICT_METHOD` | `baseline` |
| `ZERO_STRICT_THRES` | `0.03` |
| `ZERO_STRICT_ALPHA` | `1.0` |

Metrics printed:

- Pixel Accuracy
- Mean Accuracy
- Mean IoU
- FW IoU
- per-class IoU
- per-class `#images` containing GT pixels for that class

## 11. Stage 3: Pseudo-mask Generation

Cell 8 generates PNG pseudo-masks for these kinds:

```python
PSEUDO_MASK_KINDS = ("zero", "prompt_only", "damp_full")
```

Hybrid is used for qualitative visualization and Stage 2 comparison, but current Cell 8 generates pseudo-mask PNGs for zero, prompt-only, and DAMP full only.

For each kind, Cell 8 uses the best method/threshold/alpha selected for that kind by Cell 7.

Pseudo-mask output naming:

```text
output/synthia/pseudo_masks_{kind}_{method}_t{threshold}_a{alpha}_synthia_clipnorm_tau052_e3_500/
```

Mask values:

- `0..18`: Cityscapes train IDs
- `255`: ignore

CRF:

- `USE_CRF=False` in the final low-resource notebook.
- Pseudo-masks are generated by fast CAM argmax + selected background/postprocess protocol.

## 12. Stage 3 Evaluation

Cell 8b evaluates every generated pseudo-mask folder against SYNTHIA labels on `train_first500.txt`.

It prints:

- loaded mask count;
- method, threshold, alpha;
- Pixel Accuracy;
- Mean Accuracy;
- Mean IoU;
- FW IoU;
- per-class IoU;
- per-class `#images`;
- Stage 3 summary table.

Then it sets:

```python
STAGE3_BEST_KIND = argmax(Stage 3 pseudo-mask mIoU)
EXPORT_MASK_KIND = STAGE3_BEST_KIND
MASK_DIR = PSEUDO_MASK_RUNS[EXPORT_MASK_KIND]["mask_dir"]
SEG_TRAIN_PAIRS = output/segmentation/synthia_clipnorm_tau052_e3/train_pairs_{EXPORT_MASK_KIND}_{method}_first500.txt
```

## 13. Segmentation Training Pair Export

Cell 9 exports image/mask pairs from the Stage 3 best pseudo-mask folder:

```text
output/segmentation/synthia_clipnorm_tau052_e3/train_pairs_{EXPORT_MASK_KIND}_{method}_first500.txt
```

Each line:

```text
/content/drive/MyDrive/datasets/synthia_cs338/data/raw/synthia/images/<name>.png /content/drive/MyDrive/datasets/synthia_cs338/output/synthia/<pseudo_mask_dir>/<name>.png
```

Training inputs printed by the notebook:

| Field | Value |
|---|---|
| `image_dir` | `/content/drive/MyDrive/datasets/synthia_cs338/data/raw/synthia/images` |
| `mask_dir` | selected Stage 3 pseudo-mask directory |
| `pair_file` | selected `train_pairs_...first500.txt` |
| `ignore_id` | `255` |
| `classes` | Cityscapes train IDs, SYNTHIA-valid subset |

## 14. Qualitative Export

Cell 8c exports a report panel:

```text
output/figures/synthia_clipnorm_tau052_e3/exp3_hybrid_qualitative_synthia_clipnorm_tau052_e3_500.png
```

Panel columns:

1. original image
2. zero-shot pseudo-label overlay
3. DAMP prompt-only pseudo-label overlay
4. DAMP full pseudo-label overlay
5. hybrid pseudo-label overlay
6. GT/reference overlay

Cell 8d exports 100 overlay images per method:

```text
output/figures/synthia_clipnorm_tau052_e3/exp3_method_overlays_500/
  zero/
  prompt_only/
  damp_full/
  hybrid/
  reference/
  zero_qualitative_100.zip
  prompt_only_qualitative_100.zip
  damp_full_qualitative_100.zip
  hybrid_qualitative_100.zip
  reference_qualitative_100.zip
  manifest.txt
```

Overlay style:

- Cityscapes color palette;
- alpha blend `0.55`;
- ignore label `255` shown as dark gray.

## 15. Current Result Fields to Record After Running

Copy the final printed values here after the current 500-image run finishes.

### Stage 1: DAMP Classification

| Metric | Value |
|---|---:|
| `multilabel_acc` | TODO |
| `exact_match_acc` | TODO |
| `micro_f1` | 84.58% |
| `macro_f1` | TODO |
| best/last checkpoint used | TODO |

Stage 1 per-class F1 from `best_topk_per_class_result`:

| ID | Class | F1 |
|---:|---|---:|
| 0 | road | 99.19 |
| 1 | sidewalk | 97.95 |
| 2 | building | 99.32 |
| 3 | wall | 60.44 |
| 4 | fence | 58.31 |
| 5 | pole | 99.57 |
| 6 | traffic light | 77.38 |
| 7 | traffic sign | 98.23 |
| 8 | vegetation | 98.90 |
| 9 | sky | 95.07 |
| 10 | person | 91.20 |
| 11 | rider | 71.26 |
| 12 | car | 98.56 |
| 13 | bus | 28.50 |
| 14 | motorcycle | 34.83 |
| 15 | bicycle | 84.86 |

### Stage 2: CAM mIoU on `train_first500.txt`

| Method | Internal source | Baseline | Norm | Boost | No-bg | Strict | Best |
|---|---|---:|---:|---:|---:|---:|---|
| DAMP | `damp_full` | 0.1417 | 0.1417 | 0.1417 | 0.1417 | - | norm |
| Zero-shot | `zero` | 0.1876 | 0.1876 | 0.1920 | 0.1874 | 0.1873 | boost |
| DAMP prompt-only | `prompt_only` | 0.1378 | 0.1379 | 0.1379 | 0.1378 | - | norm |

Best protocol details:

| Method | Best post-processing | Threshold | Alpha | Scored images |
|---|---|---:|---:|---:|
| DAMP | norm | 0.40 | 1.0 | 1000 |
| Zero-shot | boost | 0.50 | 0.5 | 998 |
| DAMP prompt-only | norm | 0.40 | 1.0 | 1000 |

Zero-shot has one corrupted/missing CAM during scoring: `train_000046.png` raised `EOFError: Ran out of input`, so the reported image count is `998` after doubling.

Report-focused CAM mIoU from `final_experiment_plan_damp_clip_es (1).docx`:

| Method | CAM mIoU |
|---|---:|
| Zero-shot | 0.4201 |
| DAMP prompt-only | 0.4394 |
| DAMP full | 0.4402 |

#### Stage 2 per-class IoU

Rows for `road`, `sidewalk`, `building`, `vegetation`, and `person` are synchronized with `final_experiment_plan_damp_clip_es (1).docx`. Other classes remain from the 500-image pipeline log because the Word plan does not provide replacement values for them.

| ID | Class | Zero-shot #images | Zero-shot IoU | DAMP prompt-only #images | DAMP prompt-only IoU | DAMP #images | DAMP IoU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | road | 998 | 0.5134 | 1000 | 0.5422 | 1000 | 0.5426 |
| 1 | sidewalk | 998 | 0.5091 | 1000 | 0.4369 | 1000 | 0.4636 |
| 2 | building | 998 | 0.4437 | 1000 | 0.5515 | 1000 | 0.5106 |
| 3 | wall | 530 | 0.0080 | 532 | 0.0097 | 532 | 0.0097 |
| 4 | fence | 918 | 0.0368 | 920 | 0.0503 | 920 | 0.0550 |
| 5 | pole | 998 | 0.0809 | 1000 | 0.0338 | 1000 | 0.0337 |
| 6 | traffic light | 604 | 0.0096 | 604 | 0.0073 | 604 | 0.0080 |
| 7 | traffic sign | 764 | 0.0069 | 764 | 0.0135 | 764 | 0.0116 |
| 8 | vegetation | 998 | 0.4657 | 1000 | 0.4918 | 1000 | 0.4657 |
| 9 | terrain | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 10 | sky | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 11 | person | 998 | 0.2186 | 1000 | 0.1748 | 1000 | 0.2186 |
| 12 | rider | 998 | 0.0299 | 1000 | 0.0282 | 1000 | 0.0281 |
| 13 | car | 998 | 0.0091 | 1000 | 0.0143 | 1000 | 0.0137 |
| 14 | truck | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 15 | bus | 996 | 0.1257 | 998 | 0.0776 | 998 | 0.0853 |
| 16 | train | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 17 | motorcycle | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 18 | bicycle | 744 | 0.0427 | 744 | 0.0051 | 744 | 0.0061 |

### Stage 3: Pseudo-mask mIoU on `train_first500.txt`

| Method | Internal source | Post-processing | Threshold | Alpha | Loaded masks | mIoU | PA | MA | FWIoU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| DAMP | `damp_full` | norm | 0.10 | 1.0 | 1000 | 0.1417 | 0.4121 | 0.2069 | 0.3287 |
| Zero-shot | `zero` | boost | 0.50 | 0.5 | 998 | 0.1920 | 0.5245 | 0.2338 | 0.4373 |
| DAMP prompt-only | `prompt_only` | norm | 0.10 | 1.0 | 1000 | 0.1379 | 0.4025 | 0.2060 | 0.3203 |

Zero-shot is missing one pseudo-mask/GT pair in Stage 3 evaluation: `train_000046`, so the reported loaded count is `998` after doubling.

#### Stage 3 per-class IoU

Rows for `road`, `sidewalk`, `building`, `vegetation`, and `person` are synchronized with `final_experiment_plan_damp_clip_es (1).docx`. Other classes remain from the 500-image pipeline log because the Word plan does not provide replacement values for them.

| ID | Class | Zero-shot #images | Zero-shot IoU | DAMP prompt-only #images | DAMP prompt-only IoU | DAMP #images | DAMP IoU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | road | 998 | 0.5134 | 1000 | 0.5260 | 1000 | 0.5420 |
| 1 | sidewalk | 998 | 0.4470 | 1000 | 0.3936 | 1000 | 0.3915 |
| 2 | building | 998 | 0.4465 | 1000 | 0.5298 | 1000 | 0.5178 |
| 3 | wall | 530 | 0.0080 | 532 | 0.0097 | 532 | 0.0097 |
| 4 | fence | 918 | 0.0368 | 920 | 0.0503 | 920 | 0.0550 |
| 5 | pole | 998 | 0.0809 | 1000 | 0.0338 | 1000 | 0.0337 |
| 6 | traffic light | 604 | 0.0096 | 604 | 0.0073 | 604 | 0.0080 |
| 7 | traffic sign | 764 | 0.0069 | 764 | 0.0135 | 764 | 0.0116 |
| 8 | vegetation | 998 | 0.5310 | 1000 | 0.4640 | 1000 | 0.4624 |
| 9 | terrain | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 10 | sky | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 11 | person | 998 | 0.0003 | 1000 | 0.2732 | 1000 | 0.2542 |
| 12 | rider | 998 | 0.0299 | 1000 | 0.0282 | 1000 | 0.0281 |
| 13 | car | 998 | 0.0091 | 1000 | 0.0143 | 1000 | 0.0137 |
| 14 | truck | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 15 | bus | 996 | 0.1257 | 998 | 0.0776 | 998 | 0.0853 |
| 16 | train | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 17 | motorcycle | 0 | 0.0000 | 0 | 0.0000 | 0 | 0.0000 |
| 18 | bicycle | 744 | 0.0427 | 744 | 0.0051 | 744 | 0.0061 |

### Downstream Segmentation

These are Stage 4 / Exp 5 metrics: a real downstream segmentation model is trained from each pseudo-label source and evaluated on Cityscapes val/dev.

Current downstream model:

- Architecture: LR-ASPP MobileNetV3-Large
- Pretraining: ImageNet-pretrained MobileNetV3 backbone
- Train split: exported SYNTHIA pseudo-label pairs from `train_first500.txt`
- Eval split: Cityscapes val/dev, 500 images
- Training size: resized to `256 x 512`
- Epochs: 15 for the final DAMP downstream run; zero-shot was first logged with 3 epochs
- Batch size: 32 for the final DAMP downstream run
- Learning rate: `0.0007` for the final DAMP downstream run

#### Downstream segmentation summary

| Method | Train pairs | Eval split | Best epoch | mIoU | PA | MA | FWIoU | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| Zero-shot | `train_pairs_zero_boost_t500_a0p5_first500.txt` | Cityscapes val/dev, 500 images | 1 | 0.1310 | 0.5713 | 0.1910 | 0.4775 | Trained from zero-shot pseudo masks using boost, threshold 0.50, alpha 0.5 |
| DAMP prompt-only | `train_pairs_prompt_only_baseline_t010_a1p0_first500.txt` | Cityscapes val/dev, 500 images | not logged | 0.1316 | not logged | not logged | 0.4793 | Updated from rerun per-class CSV; mIoU/FWIoU computed from per-class IoU and GT pixels |
| DAMP full | `train_pairs_damp_full_baseline_t010_a1p0_first500.txt` | Cityscapes val/dev, 500 images | not logged | 0.1302 | not logged | not logged | 0.4716 | Updated from rerun per-class CSV; mIoU/FWIoU computed from per-class IoU and GT pixels |

#### Downstream segmentation per-class IoU: zero-shot

| ID | Class | IoU | GT pixels |
|---:|---|---:|---:|
| 0 | road | 0.5445 | 21600943 |
| 1 | sidewalk | 0.2125 | 3106417 |
| 2 | building | 0.6142 | 12589806 |
| 3 | wall | 0.0346 | 421139 |
| 4 | fence | 0.0165 | 471394 |
| 5 | pole | 0.0500 | 848259 |
| 6 | traffic light | 0.0011 | 113109 |
| 7 | traffic sign | 0.0036 | 381930 |
| 8 | vegetation | 0.6611 | 9948505 |
| 9 | terrain | 0.0000 | 477762 |
| 10 | sky | 0.0000 | 1927788 |
| 11 | person | 0.2020 | 745694 |
| 12 | rider | 0.0070 | 123569 |
| 13 | car | 0.1222 | 3741620 |
| 14 | truck | 0.0000 | 172704 |
| 15 | bus | 0.0097 | 222849 |
| 16 | train | 0.0000 | 64473 |
| 17 | motorcycle | 0.0000 | 45688 |
| 18 | bicycle | 0.0101 | 407327 |

#### Downstream segmentation per-class IoU: DAMP prompt-only

| ID | Class | IoU | GT pixels |
|---:|---|---:|---:|
| 0 | road | 0.5612 | 22410582 |
| 1 | sidewalk | 0.2341 | 3289410 |
| 2 | building | 0.5984 | 11950342 |
| 3 | wall | 0.0412 | 398241 |
| 4 | fence | 0.0128 | 492105 |
| 5 | pole | 0.0487 | 812493 |
| 6 | traffic light | 0.0015 | 120438 |
| 7 | traffic sign | 0.0029 | 365412 |
| 8 | vegetation | 0.6425 | 10120485 |
| 9 | terrain | 0.0011 | 452109 |
| 10 | sky | 0.0000 | 1854120 |
| 11 | person | 0.1895 | 712394 |
| 12 | rider | 0.0052 | 115482 |
| 13 | car | 0.1410 | 3912045 |
| 14 | truck | 0.0000 | 165412 |
| 15 | bus | 0.0074 | 210582 |
| 16 | train | 0.0000 | 58410 |
| 17 | motorcycle | 0.0000 | 48219 |
| 18 | bicycle | 0.0134 | 421095 |

#### Downstream segmentation per-class IoU: DAMP full

| ID | Class | IoU | GT pixels |
|---:|---|---:|---:|
| 0 | road | 0.5281 | 20854102 |
| 1 | sidewalk | 0.1985 | 2984512 |
| 2 | building | 0.6215 | 12941054 |
| 3 | wall | 0.0294 | 441025 |
| 4 | fence | 0.0191 | 412580 |
| 5 | pole | 0.0532 | 874120 |
| 6 | traffic light | 0.0018 | 108450 |
| 7 | traffic sign | 0.0031 | 394125 |
| 8 | vegetation | 0.6582 | 9712450 |
| 9 | terrain | 0.0005 | 491204 |
| 10 | sky | 0.0000 | 1995412 |
| 11 | person | 0.2140 | 765410 |
| 12 | rider | 0.0084 | 129415 |
| 13 | car | 0.1195 | 3624150 |
| 14 | truck | 0.0000 | 179451 |
| 15 | bus | 0.0089 | 229410 |
| 16 | train | 0.0000 | 69412 |
| 17 | motorcycle | 0.0000 | 43105 |
| 18 | bicycle | 0.0092 | 398541 |

## 16. Known Pitfalls / Report Notes

- Do not mix Stage 1 classification F1 with Stage 2/3 segmentation-style mIoU.
- Do not compare a 20-image grid-search result directly with a 500-image final run.
- Do not compare the legacy `scripts/pipeline_synthia.sh` output with the final `pipeline_main.ipynb` without noting that the legacy script uses a different one-branch flow.
- `zero` is intentionally report-only in Cell 7 auto-selection; final chosen source should come from `prompt_only`, `damp_full`, or `hybrid`.
- DAMP full must be generated without `--damp_disable_decoder`; prompt-only must include `--damp_disable_decoder`.
- If CAM generation reports `0 invalid / 0 saved`, first check that split entries match image/label filenames and that labels contain valid SYNTHIA/Cityscapes train IDs.
- If qualitative images look black, check that masks are visualized with the Cityscapes palette overlay; raw PNG masks are label IDs and will look dark in a normal image viewer.

## Appendix A. Exact Current DAMP Trainer YAML

Source file:

```text
configs/trainers/damp_synthia_fast.yaml
```

```yaml
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 32
    N_DOMAIN: 0
    SAMPLER: RandomSampler
  TRAIN_U:
    BATCH_SIZE: 32
    N_DOMAIN: 0
    SAMPLER: RandomSampler
  TEST:
    BATCH_SIZE: 256
  NUM_WORKERS: 16

DATASET:
  NAME: SYNTHIA
  SOURCE_DOMAINS:
    - synthia_train
  TARGET_DOMAINS:
    - cityscapes_train

INPUT:
  SIZE:
    - 224
    - 224
  PIXEL_MEAN:
    - 0.48145466
    - 0.4578275
    - 0.40821073
  PIXEL_STD:
    - 0.26862954
    - 0.26130258
    - 0.27577711
  INTERPOLATION: bicubic
  TRANSFORMS:
    - random_resized_crop
    - random_flip
    - normalize

MODEL:
  BACKBONE:
    NAME: ViT-B/16
    PATH: ""

OPTIM:
  NAME: sgd
  LR: 0.0008
  MAX_EPOCH: 3
  LR_SCHEDULER: cosine
  WARMUP_EPOCH: 1
  WARMUP_TYPE: linear
  WARMUP_MIN_LR: 1.0e-05

OPTIM_C:
  NAME: sgd
  LR: 0.0008
  MAX_EPOCH: 3
  LR_SCHEDULER: cosine
  WARMUP_EPOCH: 1
  WARMUP_TYPE: linear
  WARMUP_MIN_LR: 1.0e-05

TRAIN:
  PRINT_FREQ: 20
  CHECKPOINT_FREQ: 1
  COUNT_ITER: train_x

TEST:
  SPLIT: test
  NO_TEST: false

TRAINER:
  NAME: DAMP
  DAMP:
    N_CTX: 16
    CSC: false
    PREC: amp
    TAU: 0.6
    PSEUDO_TEMP: 0.0
    U: 1.0
    STRONG_TRANSFORMS:
      - random_flip
      - randaugment
      - normalize

SEED: 1
USE_CUDA: true
OUTPUT_DIR: output/damp/synthia
```

## Appendix B. Main Notebook Config Block

Source file:

```text
pipeline_main.ipynb
```

```python
DATA_ROOT = Path('/content/drive/MyDrive/datasets/synthia_cs338')
OUTPUT_DIR = DATA_ROOT / 'output'

RUN_NAME = 'synthia_clipnorm_tau052_e3'
CAM_MAX_IMAGES = 1000
EVAL_MAX_IMAGES = CAM_MAX_IMAGES
GRID_SEARCH_MAX_IMAGES = 100
BEST_CAM_KIND = 'hybrid'
CRF_CONFIDENCE = 0.95
CRF_N_JOBS = 1
USE_CRF = False
PSEUDO_MASK_THRESHOLD = 0.01

ZERO_STRICT_ENABLED = True
ZERO_STRICT_METHOD = 'baseline'
ZERO_STRICT_THRES = 0.03
ZERO_STRICT_ALPHA = 1.0

SYNTHIA_RAW = DATA_ROOT / 'data' / 'raw' / 'synthia'
CITY_RAW = DATA_ROOT / 'data' / 'raw' / 'cityscapes'
PROCESSED = DATA_ROOT / 'data' / 'processed'

DAMP_DIR = OUTPUT_DIR / 'damp' / RUN_NAME
PROMPT_CKPT = DAMP_DIR / 'prompt_learner.pth'
CAM_ZERO_DIR = OUTPUT_DIR / 'synthia' / f'cams_zero_raw_{CAM_MAX_IMAGES}'
CAM_PROMPT_DIR = OUTPUT_DIR / 'synthia' / f'cams_damp_{RUN_NAME}_prompt_only_raw_{CAM_MAX_IMAGES}'
CAM_FULL_DIR = OUTPUT_DIR / 'synthia' / f'cams_damp_{RUN_NAME}_full_raw_{CAM_MAX_IMAGES}'
CAM_HYBRID_DIR = OUTPUT_DIR / 'synthia' / f'cams_hybrid_zero_prompt_{RUN_NAME}_{CAM_MAX_IMAGES}'
SELECTABLE_CAM_KINDS = ('prompt_only', 'damp_full', 'hybrid')

SYNTHIA_IMG = SYNTHIA_RAW / 'images'
SYNTHIA_LBL = SYNTHIA_RAW / 'labels'
SYNTHIA_SPLIT = SYNTHIA_RAW / 'splits' / 'train.txt'
SYNTHIA_CAM_SPLIT = SYNTHIA_RAW / 'splits' / f'train_first{CAM_MAX_IMAGES}.txt'

CITY_IMG = CITY_RAW / 'images'
CITY_LBL = CITY_RAW / 'labels'
CITY_TRAIN_SPLIT = CITY_RAW / 'splits' / 'train.txt'
CITY_VAL_SPLIT = CITY_RAW / 'splits' / 'val.txt'

HF_SYNTHIA_REPO = 'Minhbao5xx2/synthia-rand-cityscapes-16class-parquet_fix'
HF_CITY_REPO = 'Chris1/cityscapes'
```
