# Phase 3: DAMP Trainer Integration (DETAILED)

## Goal
Port DAMP trainer vào unified project, hỗ trợ **cả single-label VÀ multi-label** đúng cách. Phase này cực quan trọng — sai 1 chỗ loss là kết quả sai hoàn toàn.

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
    ├── office_home.yaml
    ├── visda17.yaml
    └── mini_domainnet.yaml

train.py
```

---

## VẤN ĐỀ QUAN TRỌNG: SINGLE-LABEL vs MULTI-LABEL

### Code gốc DAMP có gì sai?

Code hiện tại trong `forward_backward()` **hardcode multi-label** cho tất cả datasets:
- `loss_x` / `loss_x2`: luôn dùng `F.binary_cross_entropy_with_logits`
- `loss_u` / `loss_u2`: luôn dùng `F.binary_cross_entropy_with_logits` + sigmoid pseudo-labels
- `loss_im`: luôn dùng `IM_loss_multilabel` (sigmoid-based)
- `IM_loss()` (softmax-based, single-label) **được định nghĩa nhưng KHÔNG BAO GIỜ được gọi**

Vấn đề: Cho single-label datasets (Office-Home, VisDA17, Mini-DomainNet):
- BCE với sigmoid KHÔNG enforce mutual exclusivity giữa classes → prediction có thể sum > 1
- Pseudo-labeling bằng sigmoid + threshold per class không phù hợp → nên dùng argmax
- `IM_loss_multilabel` dùng binary entropy per class, không enforce class competition

### Giải pháp: Thêm config `MULTILABEL` flag

```python
# Trong extend_cfg():
cfg.TRAINER.DAMP.MULTILABEL = True  # True cho VOC12/COCO14/GTA5, False cho Office-Home/VisDA17/MiniDomainNet
```

Dataset mapping:
| Dataset | MULTILABEL | Loss function | Pseudo-label strategy |
|---------|-----------|---------------|----------------------|
| VOC12 | True | BCE + sigmoid | sigmoid + per-class threshold |
| COCO14 | True | BCE + sigmoid | sigmoid + per-class threshold |
| GTA5/Cityscapes | True | BCE + sigmoid | sigmoid + per-class threshold |
| Office-Home | False | CE + softmax | softmax + max-prob threshold |
| VisDA17 | False | CE + softmax | softmax + max-prob threshold |
| Mini-DomainNet | False | CE + softmax | softmax + max-prob threshold |

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
  ├── Text Prompting:  text_embeddings + context_decoder(text_embeddings, visual_contexts)
  │   updated_text = text_embeddings + gamma_t * context_decoder(text_embeddings, visual_contexts)
  │
  └── Visual Prompting: global_feat + context_decoder(global_feat, text_contexts)
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

## LOSS FUNCTIONS — CHI TIẾT TỪNG CÁI (MULTI-LABEL vs SINGLE-LABEL)

### Tổng công thức loss cuối cùng

```
loss = (loss_x + loss_x2)
     + U * (loss_u + loss_u2)
     + loss_ind
     + 0.1 * loss_im
```

Fallback khi loss_im không finite:
```
loss = (loss_x + loss_x2) + U * (loss_u + loss_u2) + loss_ind
```

Config weights:
- `U = cfg.TRAINER.DAMP.U` (thay đổi theo dataset)
- `loss_ind` weight = 1.0 (hardcoded)
- `loss_im` weight = 0.1 (hardcoded)

---

### Loss 1: `loss_x` — Supervised Loss (Labeled, weak augmentation)

**Mục đích**: Học phân loại trên labeled data với weak augmentation.

#### Multi-label mode (VOC12, COCO14, GTA5/Cityscapes):

```python
output_x, output_x_ind = self.model(image_x, ind=True, pse=False)
# output_x: [B_x, K]  — logits từ updated embeddings
# output_x_ind: [B_x, B_x] — individuation logits (dùng cho loss 3)

loss_x = F.binary_cross_entropy_with_logits(output_x, label)
```

- `label`: multi-hot vector shape `[B_x, K]`, giá trị 0/1
- **BCE with logits** = tự apply sigmoid rồi tính cross-entropy, numerically stable
- Mỗi class tính independently → cho phép nhiều class = 1 cùng lúc

#### Single-label mode (Office-Home, VisDA17, Mini-DomainNet):

```python
# label: class indices shape [B_x], giá trị 0..K-1
loss_x = F.cross_entropy(output_x, label)
```

- `label`: integer tensor `[B_x]` — single class index per sample
- **CE** = softmax + negative log-likelihood → enforce mutual exclusivity
- Chỉ 1 class đúng per sample → competition giữa classes

---

### Loss 2: `loss_x2` — Supervised Loss (Labeled, strong augmentation)

**Mục đích**: Consistency regularization — same label, different augmentation, same prediction.

```python
output_x2 = self.model(image_x2)[0]
```

#### Multi-label:
```python
loss_x2 = F.binary_cross_entropy_with_logits(output_x2, label)
```

#### Single-label:
```python
loss_x2 = F.cross_entropy(output_x2, label)
```

---

### Loss 3: `loss_ind` — Individuation Loss (Instance Discrimination)

**Mục đích**: Mỗi sample phải phân biệt được với các sample khác trong batch → học discriminative features.

**GIỐNG NHAU cho cả multi-label và single-label** — vì đây là instance-level loss, không liên quan đến class labels.

```python
# Từ CustomCLIP.forward(ind=True):
logits_individuation = torch.einsum('ac,bkc->abk', visual, text).mean(dim=-1)  # [B, B]
logits_individuation = logits_individuation * logit_scale

# Loss cho labeled data:
x_ind_label = torch.arange(B_x, dtype=torch.long).to(self.device)  # [0, 1, ..., B_x-1]
loss_x_ind = (F.cross_entropy(output_x_ind, x_ind_label)
            + F.cross_entropy(output_x_ind.permute(1, 0), x_ind_label)) / 2.0

# Loss cho unlabeled data:
u_ind_label = torch.arange(B_u, dtype=torch.long).to(self.device)
loss_u_ind = (F.cross_entropy(output_u_ind, u_ind_label)
            + F.cross_entropy(output_u_ind.permute(1, 0), u_ind_label)) / 2.0

# Total individuation loss:
loss_ind = loss_x_ind + loss_u_ind
```

**Giải thích**:
- `logits_individuation`: shape `[B, B]` — cosine similarity giữa MỖI visual embedding và MỖI text embedding (averaged over K classes)
- **Bidirectional**: CE 2 chiều rồi trung bình
  - `F.cross_entropy(output_x_ind, labels)`: visual→text
  - `F.cross_entropy(output_x_ind.T, labels)`: text→visual
- Label = `arange(B)` → sample i phải match với text i (identity mapping)
- **Lưu ý**: `F.cross_entropy` ở đây dùng cho instance matching (B classes = B samples), KHÔNG phải semantic classification → vẫn dùng CE bất kể dataset là multi/single-label

---

### Loss 4: `loss_u` — Pseudo-Label Loss (Unlabeled, weak augmentation)

**Mục đích**: Học trên unlabeled data bằng pseudo-labels tự tạo.

#### Multi-label mode:

```python
output_u, output_u_ind, pseudo_label_logits = self.model(image_u, ind=True, pse=True)

# === TẠO PSEUDO-LABEL ===
mix_lambda = self.epoch / self.max_epoch  # tăng dần từ 0 → 1
pseudo_label = (
    torch.sigmoid(output_u.reshape(-1, self.n_cls)) * mix_lambda
    + torch.sigmoid(pseudo_label_logits.reshape(-1, self.n_cls)) * (1 - mix_lambda)
).detach()

# Binarize per-class
pseudo_label_bin = (pseudo_label >= self.pseudo_threshold).float()  # 0.55
confident_mask = pseudo_label_bin.sum(dim=1) > 0  # sample có ≥1 class confident

# === TÍNH LOSS ===
bce_u = F.binary_cross_entropy_with_logits(output_u, pseudo_label_bin, reduction="none")
if confident_mask.any():
    confidence_weights = confident_mask.float().unsqueeze(1)  # [B_u, 1]
    normalizer = confidence_weights.sum() * self.n_cls
    loss_u = (bce_u * confidence_weights).sum() / normalizer
else:
    loss_u = output_u.new_tensor(0.0)
```

**Pseudo-labeling multi-label chi tiết**:
1. **Mixing schedule**: `mix_lambda = epoch / max_epoch`
   - Epoch đầu: `λ ≈ 0` → pseudo-label chủ yếu từ **naive CLIP** (stable)
   - Epoch cuối: `λ ≈ 1` → pseudo-label chủ yếu từ **model hiện tại** (adapted)
2. **Per-class binarize**: Mỗi class独立 threshold → có thể có 0, 1, hoặc nhiều class positive
3. **Confident masking**: Chỉ tính loss cho samples có ít nhất 1 class confident

#### Single-label mode:

```python
output_u, output_u_ind, pseudo_label_logits = self.model(image_u, ind=True, pse=True)

# === TẠO PSEUDO-LABEL ===
mix_lambda = self.epoch / self.max_epoch
pseudo_probs = (
    F.softmax(output_u, dim=1) * mix_lambda
    + F.softmax(pseudo_label_logits, dim=1) * (1 - mix_lambda)
).detach()

# Lấy class có probability cao nhất
max_probs, pseudo_labels = pseudo_probs.max(dim=1)  # [B_u]
confident_mask = max_probs >= self.pseudo_threshold  # single threshold

# === TÍNH LOSS ===
if confident_mask.any():
    loss_u = F.cross_entropy(output_u[confident_mask], pseudo_labels[confident_mask])
else:
    loss_u = output_u.new_tensor(0.0)
```

**Pseudo-labeling single-label chi tiết**:
1. **Softmax mixing**: Dùng softmax thay sigmoid → class probabilities sum = 1
2. **Argmax**: Chọn class có probability cao nhất → chỉ 1 class per sample
3. **Confident masking**: Threshold trên **max probability** → sample confident nếu class có khả năng nhất đủ cao

---

### Loss 5: `loss_u2` — Pseudo-Label Loss (Unlabeled, strong augmentation)

**Mục đích**: Consistency regularization trên unlabeled data.

#### Multi-label:
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

#### Single-label:
```python
output_u2 = self.model(image_u2)[0]
if confident_mask.any():
    loss_u2 = F.cross_entropy(output_u2[confident_mask], pseudo_labels[confident_mask])
else:
    loss_u2 = output_u2.new_tensor(0.0)
```

- Pseudo-labels được tạo từ **weak augmentation** → học consistency giữa 2 views

---

### Loss 6: `loss_im` — Instance Mutual Loss (Distribution Alignment)

**Mục đích**: Align phân phối prediction trên unlabeled data → tránh mode collapse.

#### Multi-label mode — `IM_loss_multilabel` (sigmoid-based):

```python
def IM_loss_multilabel(logits_target):
    logits_target = logits_target.float()
    probs = torch.sigmoid(logits_target)    # [B_u, K]
    eps = 1e-4
    probs = probs.clamp(min=eps, max=1.0 - eps)

    # Term 1: Per-sample binary entropy (mean over batch)
    sample_entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    item1 = sample_entropy.sum(dim=1).mean()  # sum classes, mean batch

    # Term 2: Batch-wise class entropy regularization
    avg_probs = probs.mean(dim=0).clamp(min=eps, max=1.0 - eps)  # [K]
    class_entropy = -(avg_probs * torch.log(avg_probs) + (1.0 - avg_probs) * torch.log(1.0 - avg_probs))
    item2 = class_entropy.sum()  # sum classes

    # Minimize item1 - item2
    loss_im = item1 - item2
    if not torch.isfinite(loss_im):
        loss_im = torch.zeros((), device=logits_target.device, dtype=torch.float32)
    return loss_im
```

**Intuition**:
- `item1` (sample entropy): Mong mỗi sample prediction **uncertain** → exploration
- `item2` (class entropy): Mong **batch-average** prediction **concentrated** → không bị uniform
- `loss_im = item1 - item2`: Balance exploration vs concentration
- Binary entropy `H(p) = -[p·log(p) + (1-p)·log(1-p)]` phù hợp cho multi-label (mỗi class independent)

#### Single-label mode — `IM_loss` (softmax-based):

```python
def IM_loss(outputs_target, mask_lt):
    outputs_target = outputs_target[mask_lt]  # filter confident samples
    batch_size = mask_lt.sum()
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)  # [B_conf, K]
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size) + 1e-5  # [K]
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)  # batch-average entropy
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t + 1e-5)) / float(batch_size)  # mean sample entropy
    return item2 - item1
```

**Intuition**:
- **Softmax** → class probabilities sum = 1 → mutual exclusivity giữa classes
- `item1` (batch-average entropy): Mong phân phối trung bình **concentrated** (low entropy) → class balance
- `item2` (mean sample entropy): Mong mỗi sample prediction **diverse** (high entropy) → exploration
- `loss_im = item2 - item1`: Ngược sign với multi-label version!
  - Multi-label: `item1 - item2` (minimize sample entropy - class entropy)
  - Single-label: `item2 - item1` (minimize mean sample entropy - batch-average entropy)
- `mask_lt`: Chỉ tính trên confident samples (trong code gốc, truyền từ confident_mask)

**Lưu ý về sign**:
- Cả 2 version đều minimize loss → kết quả cuối cùng giống nhau (đều muốn: sample predictions đa dạng, batch-average concentrated)
- Chỉ khác công thức do binary entropy vs Shannon entropy có cấu trúc khác nhau

---

## PSEUDO-LABELING STRATEGY — TỔNG HỢP CẢ 2 MODE

### Multi-label mode:

```
                        ┌─────────────────────────────┐
                        │   Unlabeled Image (image_u)  │
                        │   + weak augmentation        │
                        └──────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            model(img, ind=True, pse=True)       │
                    │                             │
           output_u (updated)    pseudo_label_logits (NAIVE CLIP)
                    │                             │
                    │    mix_lambda = epoch/max_epoch
                    ▼                             ▼
         sigmoid(output_u) * λ  +  sigmoid(pseudo_logits) * (1-λ)
                    │                             │
                    └──────── ──────┬─────────────┘
                                   │
                          pseudo_label (soft, per-class)
                                   │
                          binarize (≥ 0.55 per class)
                                   │
                          pseudo_label_bin [B, K] (0/1)
                                   │
                          confident_mask: sum(dim=1) > 0
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            loss_u (BCE, weak aug)       loss_u2 (BCE, strong aug)
```

### Single-label mode:

```
                        ┌─────────────────────────────┐
                        │   Unlabeled Image (image_u)  │
                        │   + weak augmentation        │
                        └──────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            model(img, ind=True, pse=True)       │
                    │                             │
           output_u (updated)    pseudo_label_logits (NAIVE CLIP)
                    │                             │
                    │    mix_lambda = epoch/max_epoch
                    ▼                             ▼
         softmax(output_u) * λ  +  softmax(pseudo_logits) * (1-λ)
                    │                             │
                    └──────── ──────┬─────────────┘
                                   │
                          pseudo_probs (soft, sum=1)
                                   │
                          argmax → pseudo_labels [B] (class index)
                          max_prob → confidence [B]
                                   │
                          confident_mask: max_prob ≥ 0.55
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            loss_u (CE, weak aug)        loss_u2 (CE, strong aug)
```

---

## THỨ TỰ FORWARD-BACKWARD — UNIFIED PSEUDOCODE

```python
def forward_backward(self, batch_x, batch_u):
    image_x, image_x2, label, image_u, image_u2, label_u = self.parse_batch_train(batch_x, batch_u)
    multilabel = self.cfg.TRAINER.DAMP.MULTILABEL

    # 1. Forward tất cả inputs
    output_x, output_x_ind = self.model(image_x, ind=True, pse=False)
    output_u, output_u_ind, pseudo_label_logits = self.model(image_u, ind=True, pse=True)
    output_x2 = self.model(image_x2)[0]
    output_u2 = self.model(image_u2)[0]

    # 2. Tạo pseudo-labels — CHẠNH NHÁNH theo multilabel
    mix_lambda = self.epoch / self.max_epoch

    if multilabel:
        # ─── MULTI-LABEL PATH ───
        pseudo_label = (
            torch.sigmoid(output_u) * mix_lambda
            + torch.sigmoid(pseudo_label_logits) * (1 - mix_lambda)
        ).detach()
        pseudo_label_bin = (pseudo_label >= self.pseudo_threshold).float()
        confident_mask = pseudo_label_bin.sum(dim=1) > 0

        # Supervised losses
        loss_x  = F.binary_cross_entropy_with_logits(output_x, label)
        loss_x2 = F.binary_cross_entropy_with_logits(output_x2, label)

        # Pseudo-label losses
        bce_u  = F.binary_cross_entropy_with_logits(output_u, pseudo_label_bin, reduction="none")
        bce_u2 = F.binary_cross_entropy_with_logits(output_u2, pseudo_label_bin, reduction="none")
        if confident_mask.any():
            cw = confident_mask.float().unsqueeze(1)
            norm = cw.sum() * self.n_cls
            loss_u  = (bce_u * cw).sum() / norm
            loss_u2 = (bce_u2 * cw).sum() / norm
        else:
            loss_u  = output_u.new_tensor(0.0)
            loss_u2 = output_u2.new_tensor(0.0)

        # IM loss
        loss_im = IM_loss_multilabel(output_u)

    else:
        # ─── SINGLE-LABEL PATH ───
        pseudo_probs = (
            F.softmax(output_u, dim=1) * mix_lambda
            + F.softmax(pseudo_label_logits, dim=1) * (1 - mix_lambda)
        ).detach()
        max_probs, pseudo_labels = pseudo_probs.max(dim=1)
        confident_mask = max_probs >= self.pseudo_threshold

        # Supervised losses — label là class indices [B]
        loss_x  = F.cross_entropy(output_x, label)
        loss_x2 = F.cross_entropy(output_x2, label)

        # Pseudo-label losses
        if confident_mask.any():
            loss_u  = F.cross_entropy(output_u[confident_mask], pseudo_labels[confident_mask])
            loss_u2 = F.cross_entropy(output_u2[confident_mask], pseudo_labels[confident_mask])
        else:
            loss_u  = output_u.new_tensor(0.0)
            loss_u2 = output_u2.new_tensor(0.0)

        # IM loss — dùng softmax version
        mask_lt = confident_mask
        loss_im = IM_loss(output_u, mask_lt)

    # 3. Individuation loss — GIỐNG NHAU cho cả 2 mode
    x_ind_label = torch.arange(B_x, dtype=torch.long).to(self.device)
    loss_x_ind = (F.cross_entropy(output_x_ind, x_ind_label)
                + F.cross_entropy(output_x_ind.permute(1, 0), x_ind_label)) / 2.0
    u_ind_label = torch.arange(B_u, dtype=torch.long).to(self.device)
    loss_u_ind = (F.cross_entropy(output_u_ind, u_ind_label)
                + F.cross_entropy(output_u_ind.permute(1, 0), u_ind_label)) / 2.0
    loss_ind = loss_x_ind + loss_u_ind

    # 4. Fallback cho loss_im
    if not torch.isfinite(loss_im):
        loss_im = output_u.new_tensor(0.0)

    # 5. Tổng loss
    loss = (loss_x + loss_x2) + U * (loss_u + loss_u2) + loss_ind + 0.1 * loss_im
    if not torch.isfinite(loss):
        loss = (loss_x + loss_x2) + U * (loss_u + loss_u2) + loss_ind

    # 6. Backward
    self.optim_p.zero_grad()
    self.optim_c.zero_grad()
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optim_p)
    self.scaler.step(self.optim_c)
    self.scaler.update()
```

---

## MULTI-LABEL HANDLING

### `_build_multihot_labels()` (dòng 664-685)

```python
def _build_multihot_labels(self, impaths, fallback_labels):
    labels = torch.zeros(len(impaths), self.num_classes)  # [B, K] all zeros

    for i, impath in enumerate(impaths):
        key = osp.abspath(impath)
        if key in self.multi_label_lookup:
            labels[i] = self.multi_label_lookup[key].to(self.device)  # multi-hot từ JSON
        else:
            # Fallback: single-label → one-hot
            cls_id = int(fallback_labels[i])
            if 0 <= cls_id < self.num_classes:
                labels[i, cls_id] = 1.0

    return labels
```

**Lưu ý cho single-label mode**: Khi `MULTILABEL=False`, hàm này vẫn trả multi-hot (one-hot thật ra), nhưng `forward_backward` sẽ dùng `label.argmax(dim=1)` để lấy class indices cho `F.cross_entropy`. Hoặc đơn giản hơn: giữ nguyên `parse_batch_train` trả `label` trực tiếp từ batch (Dassl `TrainerXU` đã trả integer label cho single-label datasets).

**Cách đơn giản nhất**:
```python
if multilabel:
    label = self._build_multihot_labels(impath_x, label_x)  # [B, K] multi-hot
else:
    label = label_x  # [B] integer class indices — dùng trực tiếp từ batch
```

---

## EVALUATION — CŨNG CẦN CHẠNH NHÁNH

### Multi-label (VOC12, COCO14, GTA5/Cityscapes):
- `multilabel_acc`: (correct labels / total labels) * 100
- `exact_match_acc`: % samples where ALL labels correct
- `micro_f1`: F1 from global TP/FP/FN
- `macro_f1`: average F1 across classes
- **Best model**: `micro_f1 * 100`

### Single-label (Office-Home, VisDA17, Mini-DomainNet):
- `accuracy`: top-1 classification accuracy
- `class_accuracy`: per-class accuracy
- **Best model**: `accuracy`

```python
if multilabel:
    # existing multi-label evaluation code
    ...
    return float(micro_f1) * 100.0
else:
    # standard single-label evaluation
    preds = output.argmax(dim=1)
    acc = (preds == label).float().mean() * 100.0
    return acc
```

---

## CONFIG PARAMETERS — ALL

| Parameter | Default | Where used | Description |
|-----------|---------|------------|-------------|
| `N_CTX` | 16 | PromptLearner | Số learnable context tokens |
| `N_CLS` | 2 | PromptLearner | Số class tokens (không dùng trong forward) |
| `CSC` | False | PromptLearner | Class-specific context (False = shared) |
| `PREC` | "fp16" | build_model, forward_backward | Precision: fp16/fp32/amp |
| `MULTILABEL` | **True** | forward_backward, test | **NEW!** Single vs multi-label mode |
| `TAU` | 0.5 | run_epoch | Set `self.threshold` (không dùng trong forward_backward!) |
| `U` | 1.0 | forward_backward | Weight cho unsupervised loss |
| `IND` | 1.0 | extend_cfg | Không dùng (hardcoded 1.0) |
| `IM` | 1.0 | extend_cfg | Không dùng (hardcoded 0.1) |
| `STRONG_TRANSFORMS` | [] | build_data_loader | Augmentation cho strong aug |

**Per-dataset config overrides**:

| Dataset | MULTILABEL | TAU | U | Backbone |
|---------|-----------|-----|---|----------|
| VOC12 | True | 0.5 | 1.0 | ViT-B/16 |
| GTA5 | True | 0.85 | 0.5 | ViT-B/16 |
| COCO14 | True | 0.5 | 1.0 | ViT-B/16 |
| Office-Home | **False** | 0.6 | 1.0 | RN50 |
| VisDA17 | **False** | 0.5 | 2.0 | ViT-B/16 |
| Mini-DomainNet | **False** | 0.5 | 1.0 | RN50 |

---

## TASKS

### 3.1 Port DAMP trainer
- Copy `DAMP/trainers/damp.py` → `trainers/damp.py`
- Update imports: `from clip import clip` → use shared `clip/` module
- Verify `TextEncoder` accesses all CLIP submodules

### 3.2 Thêm MULTILABEL flag
- Thêm `cfg.TRAINER.DAMP.MULTILABEL = True` trong `extend_cfg()`
- Update `forward_backward()` với multi-label / single-label branching
- Update `test()` với multi-label / single-label evaluation branching
- Giữ cả 2 hàm `IM_loss()` và `IM_loss_multilabel()`

### 3.3 Port training entry point
- Copy `DAMP/train.py` → `train.py`
- Update imports

### 3.4 Port configs
- Copy `DAMP/configs/` → `configs/`
- Thêm `MULTILABEL: False` vào office_home.yaml, visda17.yaml, mini_domainnet.yaml
- Thêm `MULTILABEL: True` vào voc12.yaml, gta5.yaml

### 3.5 Add prompt_learner.pth saving for CLIP-ES bridge
- After saving best model in `after_epoch()`:
  ```python
  prompt_state = self.model.prompt_learner.state_dict()
  torch.save(prompt_state, osp.join(self.output_dir, 'prompt_learner.pth'))
  ```
- Contains: `ctx`, `gamma_t`, `gamma_v`, `token_prefix`, `token_suffix`

---

## VERIFICATION CHECKLIST
- [ ] `python train.py --config-file configs/datasets/voc12.yaml --trainer DAMP` → multi-label mode
- [ ] `python train.py --config-file configs/datasets/office_home.yaml --trainer DAMP` → single-label mode
- [ ] Multi-label: `loss_x` uses BCE, `loss_im` uses `IM_loss_multilabel`, pseudo-labels per-class threshold
- [ ] Single-label: `loss_x` uses CE, `loss_im` uses `IM_loss`, pseudo-labels argmax + max-prob threshold
- [ ] `loss_ind` same for both modes
- [ ] Evaluation metrics correct for each mode
- [ ] `prompt_learner.pth` saved after best epoch
- [ ] Compare loss values with original DAMP repo (same seed) — should match for multi-label datasets
