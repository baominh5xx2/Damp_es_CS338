# Phase 3: DAMP Trainer Integration (DETAILED)

## Goal
Port DAMP trainer vào unified project. Chỉ hỗ trợ **multi-label** (VOC12, COCO14, GTA5/Cityscapes). Bỏ hoàn toàn single-label path (Office-Home, VisDA17, Mini-DomainNet).

---

## Source Files
- `DAMP/trainers/damp.py` — Full model + training logic (~1100 lines)
- `DAMP/train.py` — Training entry point (235 lines)
- `DAMP/configs/trainers/DAMP/damp.yaml` — Trainer config
- `DAMP/configs/datasets/*.yaml` — Dataset configs

---

## Target Structure
```
trainers/
├── __init__.py
└── damp.py

configs/
├── trainers/
│   └── damp.yaml
└── datasets/
    ├── voc12.yaml
    ├── gta5.yaml
    └── coco14.yaml

train.py
```

---

## ARCHITECTURE OVERVIEW

### Model Architecture (CustomCLIP)

```
Input Image ──→ ImageEncoder (ViT/ResNet, FROZEN)
                    │
                    ├── global_feat: [B, C]          (CLS token / attn-pool)
                    └── visual_embeddings: [B, C, H, W]  (spatial features)
                              │
                              └── visual_contexts: [B, 1+H*W, C]  (concat global + spatial)

PromptLearner ──→ raw_prompt: [K, 77, 512]    (K = num_classes)
     │                [SOS] + [CTX (learnable)] + [CLASS_NAME] + [EOS]
     │
     ├── gamma_v: scalar (learnable, init=0.01)  ← controls visual prompt strength
     └── gamma_t: scalar (learnable, init=0.01)  ← controls text prompt strength

TextEncoder (FROZEN) ──→ text_embeddings: [B, K, C]  (at EOS token)
                     └── text_contexts: [B, L, C]     (at context positions only)

ContextDecoder (TRAINABLE):
  ├── Text Prompting:
  │   updated_text = text_embeddings + gamma_t * context_decoder(text_embeddings, visual_contexts)
  │
  └── Visual Prompting:
      updated_visual = global_feat + gamma_v * context_decoder(global_feat, text_contexts)

Final Logits:
  logits = logit_scale * (normalize(updated_visual) @ normalize(updated_text))
```

### Forward modes of CustomCLIP

```python
def forward(self, img, ind=False, pse=False, fea=False):
```

| Mode | `ind` | `pse` | `fea` | Returns |
|------|-------|-------|-------|---------|
| Classification only | False | False | False | `(logits,)` |
| + Individuation | True | False | False | `(logits, logits_ind)` |
| + Pseudo-label | False | True | False | `(logits, pseudo_logits)` |
| + Ind + Pse | True | True | False | `(logits, logits_ind, pseudo_logits)` |
| + Features | any | any | True | + `global_feat`, `updated_vision_embedding` |

---

## LOSS FUNCTIONS — CHI TIẾT TỪNG CÁI

### Tổng công thức loss cuối cùng (dòng 921)

```
loss = (loss_x + loss_x2)
     + U * (loss_u + loss_u2)
     + loss_ind
     + 0.1 * loss_im
```

Fallback khi loss không finite:
```
loss = (loss_x + loss_x2) + U * (loss_u + loss_u2) + loss_ind
```

Config weights:
- `U = cfg.TRAINER.DAMP.U` (thay đổi theo dataset)
- `loss_ind` weight = 1.0 (hardcoded)
- `loss_im` weight = 0.1 (hardcoded)

---

### Loss 1: `loss_x` — Supervised BCE (Labeled, weak augmentation)

**Mục đích**: Học phân loại trên labeled data với weak augmentation.

```python
output_x, output_x_ind = self.model(image_x, ind=True, pse=False)
# output_x: [B_x, K] — logits
# output_x_ind: [B_x, B_x] — individuation logits

loss_x = F.binary_cross_entropy_with_logits(output_x, label)
```

- `label`: multi-hot vector `[B_x, K]`, giá trị 0/1
- **BCE with logits** = sigmoid + cross-entropy, mỗi class independent

---

### Loss 2: `loss_x2` — Supervised BCE (Labeled, strong augmentation)

**Mục đích**: Consistency — same label, different augmentation.

```python
output_x2 = self.model(image_x2)[0]
loss_x2 = F.binary_cross_entropy_with_logits(output_x2, label)
```

---

### Loss 3: `loss_ind` — Individuation Loss (Instance Discrimination)

**Mục đích**: Mỗi sample phải phân biệt được với các sample khác trong batch.

```python
# Từ CustomCLIP.forward(ind=True):
logits_individuation = torch.einsum('ac,bkc->abk', visual, text).mean(dim=-1)  # [B, B]
logits_individuation = logits_individuation * logit_scale

# Labeled:
x_ind_label = torch.arange(B_x, dtype=torch.long).to(self.device)
loss_x_ind = (F.cross_entropy(output_x_ind, x_ind_label)
            + F.cross_entropy(output_x_ind.permute(1, 0), x_ind_label)) / 2.0

# Unlabeled:
u_ind_label = torch.arange(B_u, dtype=torch.long).to(self.device)
loss_u_ind = (F.cross_entropy(output_u_ind, u_ind_label)
            + F.cross_entropy(output_u_ind.permute(1, 0), u_ind_label)) / 2.0

loss_ind = loss_x_ind + loss_u_ind
```

**Giải thích**:
- Shape `[B, B]` — cosine similarity giữa visual_i và text_j (mean over K classes)
- Bidirectional CE: visual→text + text→visual
- Label = `arange(B)` → sample i match text i (identity)
- `F.cross_entropy` ở đây dùng cho instance matching (B instances), không phải semantic class

---

### Loss 4: `loss_u` — Pseudo-Label BCE (Unlabeled, weak augmentation)

**Mục đích**: Học trên unlabeled data bằng pseudo-labels.

```python
output_u, output_u_ind, pseudo_label_logits = self.model(image_u, ind=True, pse=True)

# === TẠO PSEUDO-LABEL ===
mix_lambda = self.epoch / self.max_epoch  # 0 → 1
pseudo_label = (
    torch.sigmoid(output_u.reshape(-1, self.n_cls)) * mix_lambda
    + torch.sigmoid(pseudo_label_logits.reshape(-1, self.n_cls)) * (1 - mix_lambda)
).detach()

pseudo_label_bin = (pseudo_label >= self.pseudo_threshold).float()  # 0.55
confident_mask = pseudo_label_bin.sum(dim=1) > 0  # ≥1 class positive

# === TÍNH LOSS ===
bce_u = F.binary_cross_entropy_with_logits(output_u, pseudo_label_bin, reduction="none")
if confident_mask.any():
    confidence_weights = confident_mask.float().unsqueeze(1)
    normalizer = confidence_weights.sum() * self.n_cls
    loss_u = (bce_u * confidence_weights).sum() / normalizer
else:
    loss_u = output_u.new_tensor(0.0)
```

**Pseudo-labeling chi tiết**:

1. **Mixing schedule**: `mix_lambda = epoch / max_epoch`
   - Epoch đầu: `λ ≈ 0` → pseudo-label từ **naive CLIP** (stable)
   - Epoch cuối: `λ ≈ 1` → pseudo-label từ **model hiện tại** (adapted)

2. **Naive CLIP logits** (`pseudo_label_logits`):
   ```python
   # Trong CustomCLIP.forward(pse=True):
   image_features = global_feat                          # raw, chưa update
   text_features = self.naive_text_embedding              # frozen naive prompt
   pseudo_logits = logit_scale * (normalize(image_features) @ normalize(text_features).T)
   ```
   - Dùng raw features + naive text ("a {domain} photo of a {class}.")

3. **Threshold**: `self.pseudo_threshold = 0.55` (hardcoded)
4. **Confident mask**: sample phải có ≥1 class ≥ threshold
5. **Weighted reduction**: chỉ tính loss trên confident samples

---

### Loss 5: `loss_u2` — Pseudo-Label BCE (Unlabeled, strong augmentation)

**Mục đích**: Consistency trên unlabeled data.

```python
output_u2 = self.model(image_u2)[0]
bce_u2 = F.binary_cross_entropy_with_logits(output_u2, pseudo_label_bin, reduction="none")
if confident_mask.any():
    confidence_weights = confident_mask.float().unsqueeze(1)
    normalizer = confidence_weights.sum() * self.n_cls
    loss_u2 = (bce_u2 * confidence_weights).sum() / normalizer
else:
    loss_u2 = output_u2.new_tensor(0.0)
```

- Cùng `pseudo_label_bin` và `confident_mask` với `loss_u`
- Pseudo-labels từ weak aug → learn consistency

---

### Loss 6: `loss_im` — Instance Mutual Loss (Distribution Alignment)

**Mục đích**: Tránh mode collapse — khuyến khích prediction đa dạng across samples.

```python
loss_im = IM_loss_multilabel(output_u)
```

**Implementation** (`IM_loss_multilabel`):

```python
def IM_loss_multilabel(logits_target):
    logits_target = logits_target.float()       # fp32 cho stability
    probs = torch.sigmoid(logits_target)         # [B_u, K]
    eps = 1e-4
    probs = probs.clamp(min=eps, max=1.0 - eps)

    # Term 1: Per-sample binary entropy (mean over batch)
    sample_entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    item1 = sample_entropy.sum(dim=1).mean()     # sum classes, mean batch

    # Term 2: Batch-wise class entropy
    avg_probs = probs.mean(dim=0).clamp(min=eps, max=1.0 - eps)  # [K]
    class_entropy = -(avg_probs * torch.log(avg_probs) + (1.0 - avg_probs) * torch.log(1.0 - avg_probs))
    item2 = class_entropy.sum()                   # sum classes

    loss_im = item1 - item2
    if not torch.isfinite(loss_im):
        loss_im = torch.zeros((), device=logits_target.device, dtype=torch.float32)
    return loss_im
```

**Intuition**:
- `item1` (sample entropy): minimize → mỗi sample confident (prediction gần 0 hoặc 1)
- `item2` (class entropy): maximize → batch-average cho mỗi class đa dạng (không collapse về toàn 0 hoặc toàn 1)
- `loss_im = item1 - item2`: minimize → confident per sample, diverse across batch

---

## PSEUDO-LABELING — TỔNG HỢP

```
                        ┌─────────────────────────────┐
                        │   Unlabeled Image (image_u)  │
                        │   + weak augmentation        │
                        └──────────┬──────────────────┘
                                   │
                    model(img, ind=True, pse=True)
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
           output_u (updated)    pseudo_label_logits (NAIVE CLIP)
                    │                             │
                    ▼                             ▼
         sigmoid(output_u) * λ  +  sigmoid(pseudo_logits) * (1-λ)
                    │                             │
                    └──────────┬──────────────────┘
                               │
                      pseudo_label (soft, per-class)
                               │
                      binarize (≥ 0.55 per class)
                               │
                      pseudo_label_bin [B, K] (0/1)
                               │
                      confident_mask: sum(dim=1) > 0
                               │
                    ┌──────────┴──────────┐
                    │                     │
            loss_u (weak aug)     loss_u2 (strong aug)
```

Key parameters:
- `mix_lambda = epoch / max_epoch` — linear 0→1
- `pseudo_threshold = 0.55` — hardcoded
- `confident_mask` — ≥1 class positive

---

## THỨ TỰ FORWARD-BACKWARD

```python
# 1. Forward labeled (weak aug) → logits + individuation
output_x, output_x_ind = self.model(image_x, ind=True, pse=False)

# 2. Forward unlabeled (weak aug) → logits + ind + pseudo-label logits
output_u, output_u_ind, pseudo_label_logits = self.model(image_u, ind=True, pse=True)

# 3. Forward labeled (strong aug) → logits only
output_x2 = self.model(image_x2)[0]

# 4. Forward unlabeled (strong aug) → logits only
output_u2 = self.model(image_u2)[0]

# 5. Pseudo-labels
mix_lambda = self.epoch / self.max_epoch
pseudo_label = (sigmoid(output_u) * λ + sigmoid(pseudo_logits) * (1-λ)).detach()
pseudo_label_bin = (pseudo_label >= 0.55).float()
confident_mask = pseudo_label_bin.sum(dim=1) > 0

# 6. Losses
loss_x  = BCE_with_logits(output_x, label)
loss_x2 = BCE_with_logits(output_x2, label)
loss_u  = weighted_BCE(output_u, pseudo_label_bin, confident_mask)
loss_u2 = weighted_BCE(output_u2, pseudo_label_bin, confident_mask)
loss_ind = CE_bidir(output_x_ind) + CE_bidir(output_u_ind)
loss_im = IM_loss_multilabel(output_u)

# 7. Total
loss = (loss_x + loss_x2) + U * (loss_u + loss_u2) + loss_ind + 0.1 * loss_im

# 8. Backward (AMP)
optim_p.zero_grad()
optim_c.zero_grad()
scaler.scale(loss).backward()
scaler.step(optim_p)   # PromptLearner
scaler.step(optim_c)   # ContextDecoder
scaler.update()
```

**2 optimizers**:
- `optim_p`: Adam, LR=3e-3 → PromptLearner (ctx, gamma_t, gamma_v)
- `optim_c`: Adam, LR=3e-3 → ContextDecoder (transformer decoder)
- CLIP backbone: **FROZEN**

---

## MULTI-LABEL HANDLING

### `_build_multihot_labels()`

```python
def _build_multihot_labels(self, impaths, fallback_labels):
    labels = torch.zeros(len(impaths), self.num_classes)

    for i, impath in enumerate(impaths):
        key = osp.abspath(impath)
        if key in self.multi_label_lookup:
            labels[i] = self.multi_label_lookup[key].to(self.device)
        else:
            cls_id = int(fallback_labels[i])
            if 0 <= cls_id < self.num_classes:
                labels[i, cls_id] = 1.0

    return labels
```

- VOC12, COCO14: 1 ảnh nhiều class → multi-hot
- GTA5/Cityscapes: segmentation mask → multi-hot
- `multi_label_lookup`: dict `{image_path: multi-hot vector}` từ JSON

---

## EVALUATION

**Metrics** (multi-label):
- `multilabel_acc`: (correct labels / total labels) × 100
- `exact_match_acc`: % samples ALL labels correct
- `micro_f1`: F1 from global TP/FP/FN
- `macro_f1`: average F1 per class

**Best model selection**: `micro_f1 × 100`

---

## CONFIG PARAMETERS

| Parameter | Default | Where used | Description |
|-----------|---------|------------|-------------|
| `N_CTX` | 16 | PromptLearner | Số learnable context tokens |
| `N_CLS` | 2 | PromptLearner | Số class tokens |
| `CSC` | False | PromptLearner | Class-specific context |
| `PREC` | "fp16" | build_model | Precision: fp16/fp32/amp |
| `TAU` | 0.5 | run_epoch | Gán `self.threshold` (không dùng trong forward_backward) |
| `U` | 1.0 | forward_backward | Weight unsupervised loss |
| `IND` | 1.0 | extend_cfg | Không dùng (hardcoded 1.0) |
| `IM` | 1.0 | extend_cfg | Không dùng (hardcoded 0.1) |
| `STRONG_TRANSFORMS` | [] | build_data_loader | Strong augmentation choices |

**Per-dataset overrides**:

| Dataset | TAU | U | Backbone |
|---------|-----|---|----------|
| VOC12 | 0.5 | 1.0 | ViT-B/16 |
| GTA5 | 0.85 | 0.5 | ViT-B/16 |
| COCO14 | 0.5 | 1.0 | ViT-B/16 |

---

## CLEANUP: BỎ NHỮNG GÌ

Khi port sang `trainers/damp.py`, **xóa hoàn toàn**:
- `IM_loss()` (dòng 46-54) — single-label softmax version, không dùng
- Dataset configs: `office_home.yaml`, `visda17.yaml`, `mini_domainnet.yaml`
- Dataset loaders: không cần port OfficeHome, VisDA17, miniDomainNet
- Scripts: `office_home.sh`, `VisDA17.sh`, `miniDomainNet.sh`
- Imports trong `train.py`: bỏ `from dassl.data.datasets import VisDA17, OfficeHome, miniDomainNet`

---

## TASKS

### 3.1 Port DAMP trainer
- Copy `DAMP/trainers/damp.py` → `trainers/damp.py`
- **Xóa** `IM_loss()` function (chỉ giữ `IM_loss_multilabel()`)
- Update imports: `from clip import clip` → shared `clip/` module
- Verify `TextEncoder` accesses: `clip_model.transformer`, `clip_model.positional_embedding`, `clip_model.ln_final`, `clip_model.text_projection`

### 3.2 Port training entry point
- Copy `DAMP/train.py` → `train.py`
- Update imports:
  - `from datasets.voc12 import VOC12`
  - `from datasets.gta5 import GTA5`
  - `from trainers import damp`
- **Xóa** imports: `VisDA17`, `OfficeHome`, `miniDomainNet`
- **Xóa** `DAPL` config block trong `extend_cfg()` (legacy, không dùng)

### 3.3 Port configs (chỉ multi-label datasets)
- Copy `configs/datasets/voc12.yaml`, `gta5.yaml`
- Tạo mới `configs/datasets/coco14.yaml`
- Copy `configs/trainers/DAMP/damp.yaml` → `configs/trainers/damp.yaml`

### 3.4 Add prompt_learner.pth saving
- After saving best model in `after_epoch()`:
  ```python
  prompt_state = self.model.prompt_learner.state_dict()
  torch.save(prompt_state, osp.join(self.output_dir, 'prompt_learner.pth'))
  ```
- Contains: `ctx`, `gamma_t`, `gamma_v`, `token_prefix`, `token_suffix`

---

## VERIFICATION CHECKLIST
- [ ] `python train.py --config-file configs/datasets/voc12.yaml --trainer DAMP` starts
- [ ] `loss_x`, `loss_u`, `loss_ind`, `loss_im` all appear in log
- [ ] `gamma_v` and `gamma_t` logged and change during training
- [ ] `prompt_learner.pth` saved after best epoch
- [ ] `pseudo_conf_rate` logged
- [ ] Loss values reasonable (not NaN, not exploding)
- [ ] `IM_loss()` (single-label) không còn tồn tại trong code
