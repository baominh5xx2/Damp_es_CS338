# Phase 4: CLIP-ES CAM Generation Integration (DETAILED)

## Goal
Port CLIP-ES CAM generation pipeline vào `cam/` module. Phase này cực quan trọng — đây là nửa thứ 2 của pipeline, chuyển DAMP prompts thành pseudo-masks cho segmentation.

---

## Source Files
- `CLIP-ES/generate_cams_voc12.py` — VOC12 CAM generation (328 lines)
- `CLIP-ES/generate_cams_coco14.py` — COCO14 CAM generation (240 lines)
- `CLIP-ES/generate_cams_gta5.py` — GTA5/Cityscapes CAM generation (606 lines, most complete)
- `CLIP-ES/generate_cams_cityscapes.py` — Cityscapes wrapper (81 lines, delegates to gta5)
- `CLIP-ES/clip_text.py` — Text prompts & class names
- `CLIP-ES/utils.py` — IoU, bbox, XML parsing
- `CLIP-ES/clip/model.py` — Modified CLIP with attention extraction
- `CLIP-ES/pytorch_grad_cam/` — All Grad-CAM implementations

---

## Target Structure
```
cam/
├── __init__.py
├── generate.py           # Unified CAM generation
├── clip_text.py          # Text prompts & class names
├── utils.py              # IoU, bbox, XML parsing
├── grad_cam/             # Grad-CAM implementations
│   ├── __init__.py
│   ├── base_cam.py
│   ├── grad_cam.py
│   ├── ...
│   └── utils/
│       ├── __init__.py
│       ├── image.py
│       ├── model_targets.py
│       ├── reshape_transforms.py
│       ├── svd_on_activations.py
│       └── find_layers.py

generate_cams.py          # CLI entry point
```

---

## ARCHITECTURE OVERVIEW — CLIP-ES Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  1. IMAGE PREPROCESSING             │
│  Resize to multiple of patch_size   │
│  (H, W) → (ceil(H/16)*16,          │
│             ceil(W/16)*16)          │
│  Normalize (CLIP mean/std)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. CLIP IMAGE ENCODING             │
│  model.encode_image(img, H, W)      │
│  → (image_features, attn_weight_list)│
│                                     │
│  Internally:                        │
│  - Patch embed: conv1(img)          │
│  - Add CLS token + pos emb          │
│  - Upsampled pos_emb for variable H,W│
│  - Transformer blocks (all 12)      │
│  - Each block returns attention     │
│  - ln_post + proj → image_features  │
│  - attn_weight_list: list of 12     │
│    attention maps [1, H*W+1, H*W+1] │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. TEXT ENCODING                    │
│  Per image: concat fg + bg text emb │
│                                     │
│  fg: K_present classes              │
│  bg: background categories          │
│  text_features_temp = cat([fg, bg]) │
│                                     │
│  Input to Grad-CAM:                 │
│  input_tensor = [image_features,    │
│    text_features_temp, H, W]        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. GRAD-CAM PER CLASS              │
│  For each present class:            │
│    cam(input_tensor, target=cls_idx)│
│    → grayscale_cam [h//16, w//16]   │
│    → resize to (ori_W, ori_H)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. ATTENTION AFFINITY REFINEMENT   │
│  (chỉ tính 1 lần cho mỗi ảnh)      │
│                                     │
│  a. Lấy last 8 attn layers          │
│  b. Remove CLS token: [:, 1:, 1:]  │
│  c. Mean across layers              │
│  d. Row+Col normalize (2 iterations)│
│  e. Symmetrize: (T + T^T) / 2      │
│  f. Diffusion: T = T @ T            │
│  g. Mask với bbox từ CAM            │
│  h. Propagate: refined = T @ cam    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. SAVE .npy                       │
│  {                                  │
│    "keys": [class_id_0, ...],       │
│    "attn_highres": refined_cams     │
│      shape [K, ori_H, ori_W]        │
│      dtype: float16                 │
│  }                                  │
└─────────────────────────────────────┘
```

---

## CHI TIẾT TỪNG BƯỚC

### Bước 1: Image Preprocessing

```python
def _transform_resize(h, w):
    return Compose([
        Resize((h, w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
```

**Resize logic** (resize H,W thành bội số của patch_size=16):
```python
h = int(np.ceil(scale * ori_height / patch_size) * patch_size)
w = int(np.ceil(scale * ori_width / patch_size) * patch_size)
```

**GTA5/Cityscapes thêm max_long_side**:
```python
def _scaled_hw(ori_height, ori_width, scale=1.0, max_long_side=-1, patch_size=16):
    h = int(round(ori_height * scale))
    w = int(round(ori_width * scale))
    if max_long_side > 0 and max(h, w) > max_long_side:
        ratio = max_long_side / max(h, w)
        h = int(round(h * ratio))
        w = int(round(w * ratio))
    h = int(np.ceil(h / patch_size) * patch_size)
    w = int(np.ceil(w / patch_size) * patch_size)
    return h, w
```
- VOC12/COCO14: `scales=[1.0]`, không có max_long_side
- GTA5: `max_long_side=1024` (ảnh gốc 1914×1052 quá lớn)
- Cityscapes: `max_long_side=1024`

**Multi-scale + flip** (code có nhưng không dùng):
```python
ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
ms_imgs = [ms_imgs[0]]  # CHỈ LẤY scale=1.0, KHÔNG dùng flip
```
- `img_ms_and_flip` trả về `[ori, flip]` cho mỗi scale
- Nhưng chỉ dùng `ms_imgs[0]` (original, no flip)

---

### Bước 2: CLIP Image Encoding (MODIFIED)

**Khác biệt quan trọng**: CLIP model trong CLIP-ES đã được modify để trả về attention weights.

```python
# CLIP-ES/clip/model.py - VisionTransformer.forward()
def forward(self, x, H, W):
    # Upsample positional embedding cho resolution mới
    self.positional_embedding_new = upsample_pos_emb(
        self.positional_embedding, (H // 16, W // 16)
    )
    x = self.conv1(x)                    # patch embedding
    x = x.reshape(B, C, -1).permute(0, 2, 1)
    x = torch.cat([cls_token, x], dim=1)
    x = x + self.positional_embedding_new
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)              # NLD → LND
    x, attn_weight = self.transformer(x)  # ← MODIFY: trả về attn_weight
    ...
```

**Transformer.forward()** — Modified để thu thập attention:
```python
class Transformer(nn.Module):
    def forward(self, x):
        attn_weights = []
        with torch.no_grad():  # ← không tính gradient cho attention
            layers = self.layers if x.shape[0] == 77 else self.layers - 1
            for i in range(layers):
                x, attn_weight = self.resblocks[i](x)
                attn_weights.append(attn_weight)
        return x, attn_weights
```
- `x.shape[0] == 77`: Nếu input là text (77 tokens) → chạy tất cả layers
- Ngược lại (image): chạy `layers - 1` layers với no_grad, layer cuối sẽ chạy **có gradient** trong Grad-CAM
- **Lý do**: Layer cuối cần gradient cho Grad-CAM, các layer trước chỉ cần attention weights → no_grad tiết kiệm VRAM

**`upsample_pos_emb()`** — Key innovation cho variable resolution:
```python
def upsample_pos_emb(emb, new_size):
    first = emb[:1, :]          # CLS token position
    emb = emb[1:, :]            # Spatial positions [N-1, D]
    N, D = emb.size()
    size = int(np.sqrt(N))      # e.g. 196 = 14×14 for ViT-B/16 at 224px
    emb = emb.permute(1, 0).view(1, D, size, size)
    emb = F.upsample(emb, size=new_size, mode='bilinear')  # upsample to new H/16 × W/16
    emb = emb.view(D, -1).permute(1, 0)
    emb = torch.cat([first, emb], 0)  # prepend CLS position
    return nn.Parameter(emb.half())
```
- Cho phép input bất kỳ resolution (không chỉ 224×224)
- VOC12: ~480×480 → 30×30 grid (thay vì 14×14 mặc định)
- GTA5: 1024×512 → 64×32 grid

**`encode_image()`** returns:
```python
image_features, attn_weight_list = model.encode_image(image, h, w)
# image_features: tensor dùng cho Grad-CAM forward
# attn_weight_list: list của 11 attention maps (layer cuối chạy riêng trong Grad-CAM)
#   mỗi attention map: [1, H*W+1, H*W+1] (CLS + spatial tokens)
```

---

### Bước 3: Text Encoding

#### 3a. Zero-shot text features (không có DAMP)

```python
def zeroshot_classifier(classnames, templates, model, device):
    # templates = ['a clean origami {}.']
    zeroshot_weights = []
    for classname in classnames:
        texts = [template.format(classname) for template in templates]
        texts = clip.tokenize(texts).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)  # mean over templates
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    return torch.stack(zeroshot_weights, dim=1).t()  # [K, D]
```
- Template: `'a clean origami {}.'` cho tất cả datasets
- Chỉ 1 template (không ensemble nhiều templates)
- Normalize: L2 normalize từng embedding, mean, normalize lại

**Text features per image**:
```python
fg_features_temp = fg_text_features[label_id_list]     # [K_present, D] — chỉ lấy classes có mặt
bg_features_temp = bg_text_features                     # [K_bg, D] — tất cả background categories
text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)  # [K_present + K_bg, D]
```

**Background categories** (hardcoded trong `clip_text.py`):
```python
# VOC12:
BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake',
    'water','river','sea','railway','railroad','keyboard','helmet','cloud','house',
    'mountain','ocean','road','rock','street','valley','bridge','sign']

# COCO14:
BACKGROUND_CATEGORY_COCO = ['ground','land','grass','tree','building','wall','sky','lake',
    'water','river','sea','railway','railroad','helmet','cloud','house','mountain',
    'ocean','road','rock','street','valley','bridge']
```

**Class names với synonyms**:
```python
# VOC12 — original vs enhanced:
new_class_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair seat', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen']

# GTA5/Cityscapes:
CITYSCAPES_PROMPT_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person with clothes,people,human', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle']

# COCO14 — 80 classes với synonyms (xem clip_text.py)
```

---

#### 3b. DAMP prompt text features

Khi có `--damp_prompt_ckpt`, thay thế zero-shot text bằng DAMP learned prompts:

```python
def damp_prompt_classifier(classnames, model, ckpt_path, device, n_ctx=-1):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Tìm context vectors trong checkpoint
    ctx = checkpoint.get("ctx", None)
    if ctx is None:
        for key in checkpoint:
            if key.endswith(".ctx"):
                ctx = checkpoint[key]
                break

    if n_ctx <= 0:
        n_ctx = ctx.shape[-2]  # auto-detect từ checkpoint shape

    # Tạo prompt: "X X X ... {class_name}."
    prompt_prefix = " ".join(["X"] * n_ctx)
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        prompt_embeddings = model.token_embedding(tokenized_prompts).type(model.dtype)

    # Thay thế X tokens bằng learned ctx
    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(len(classnames), -1, -1)  # broadcast
    elif ctx.dim() == 3 and ctx.shape[0] == 1:
        ctx = ctx.expand(len(classnames), -1, -1)

    ctx = ctx[:, :n_ctx, :].to(device=device, dtype=model.dtype)
    prompt_embeddings[:, 1:1+n_ctx, :] = ctx  # REPLACE positions 1..n_ctx

    # Encode through CLIP text transformer
    text_features = _encode_text_with_prompt_embeddings(model, prompt_embeddings, tokenized_prompts)
    return text_features
```

**`_encode_text_with_prompt_embeddings()`** — Bypass `token_embedding`, inject learned embeddings:
```python
def _encode_text_with_prompt_embeddings(model, prompt_embeddings, tokenized_prompts):
    x = prompt_embeddings + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)  # NLD → LND
    x = model.transformer(x)
    if isinstance(x, (tuple, list)):  # CLIP-ES transformer returns (x, attn)
        x = x[0]
    x = x.permute(1, 0, 2)  # LND → NLD
    x = model.ln_final(x).type(model.dtype)
    # Lấy embedding tại EOS token
    text_features = x[
        torch.arange(x.shape[0], device=tokenized_prompts.device),
        tokenized_prompts.argmax(dim=-1)
    ] @ model.text_projection
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features
```

**Mixing DAMP + zero-shot** (`damp_mix_alpha`):
```python
damp_mix_alpha = max(0.0, min(1.0, args.damp_mix_alpha))
if damp_mix_alpha < 1.0:
    anchor_text_features = zeroshot_classifier(
        CITYSCAPES_PROMPT_NAMES, ["a clean origami {}."], model, device
    )
    fg_text_features = F.normalize(
        damp_mix_alpha * damp_text_features + (1.0 - damp_mix_alpha) * anchor_text_features,
        dim=-1,
    )
else:
    fg_text_features = damp_text_features
```
- `damp_mix_alpha=1.0` (default): Dùng 100% DAMP features
- `damp_mix_alpha=0.5`: Blend 50/50 DAMP + zero-shot
- `damp_mix_alpha=0.0`: Dùng 100% zero-shot (như không có DAMP)
- **Rationale**: Zero-shot anchor tránh "collapsed class prompts" — DAMP prompts có thể bị mode collapse nếu train không tốt

**DAMP class names mode** (`damp_use_plain_names`):
```python
damp_use_plain_names = bool(getattr(args, "damp_use_plain_names", True))
# True  → dùng CITYSCAPES_CLASS_NAMES (plain: "person", "road", ...)
# False → dùng CITYSCAPES_PROMPT_NAMES (synonym: "person with clothes,people,human", ...)
```
- Khi DAMP train với plain names → load với plain names (phải match)
- Default: `True` (plain names recommended)

---

### Bước 4: Grad-CAM Generation

**Target layer**: `model.visual.transformer.resblocks[-1].ln_1` (LayerNorm trước attention block cuối)

```python
target_layers = [model.visual.transformer.resblocks[-1].ln_1]
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
```

**ViT reshape transform** (chuyển từ token format sang spatial format):
```python
def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)  # LND → NLD
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bỏ CLS token, reshape thành grid
    result = result.transpose(2, 3).transpose(1, 2)  # NLD → NCHW
    return result
```

**Custom target class**:
```python
class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]
```

**Grad-CAM forward pass**:
```python
input_tensor = [image_features, text_features_temp, h, w]

for idx, label in enumerate(label_list):
    targets = [ClipOutputTarget(idx)]  # target = index trong text_features_temp (fg first)

    grayscale_cam, logits_per_image, attn_weight_last = cam(
        input_tensor=input_tensor,
        targets=targets,
        target_size=None,
    )
    grayscale_cam = grayscale_cam[0, :]  # [h//16, w//16]
```

**Grad-CAM internal flow** (trong `pytorch_grad_cam/grad_cam.py`):
1. Forward pass qua model → lấy activations tại target layer
2. Tính gradient của target score w.r.t. activations
3. Global average pool gradients → weights
4. `cam = ReLU(sum(weights * activations))`
5. `reshape_transform` → spatial format
6. Normalize về [0, 1]

**Lưu ý**: `input_tensor` là list `[image_features, text_features_temp, h, w]` — Grad-CAM sẽ forward qua `model.forward_last_layer()` thay vì `model()`.

**`forward_last_layer()`** trong CLIP-ES CLIP model:
```python
def forward_last_layer(self, image_features, text_features):
    # Chạy chỉ layer cuối của visual transformer (có gradient)
    x, attn_weight = self.visual.transformer.resblocks[-1](image_features)
    x = x.permute(1, 0, 2)
    x = self.visual.ln_post(x)
    x = torch.mean(x[:, 1:, :], dim=1)  # mean spatial tokens (bỏ CLS)
    if self.visual.proj is not None:
        x = x @ self.visual.proj
    image_features = x
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_image = logits_per_image.softmax(dim=-1)
    return logits_per_image, attn_weight
```
- Nhận `image_features` (đã qua 11 layers) và `text_features` (đã encode)
- Chạy **chỉ layer cuối** (layer 12) → có gradient cho Grad-CAM
- Softmax trên logits → output probability per class
- `attn_weight_last`: attention map của layer cuối (dùng cho refinement)

---

### Bước 5: Attention Affinity Refinement — CHI TIẾT ĐỰC

Đây là core innovation của CLIP-ES. Chỉ tính 1 lần cho mỗi ảnh (khi `idx == 0`).

#### 5a. Thu thập attention weights

```python
if idx == 0:  # chỉ tính 1 lần
    attn_weight_list.append(attn_weight_last)  # thêm attention của layer cuối
    # attn_weight_list bây giờ có 12 attention maps (11 từ encode_image + 1 từ Grad-CAM)

    attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
    # Bỏ CLS token: [:, 1:, 1:] → shape mỗi cái [1, H*W, H*W]

    attn_weight = torch.stack(attn_weight, dim=0)[-8:]
    # Stack rồi lấy 8 layers cuối → [8, 1, H*W, H*W]

    attn_weight = torch.mean(attn_weight, dim=0)
    # Mean across 8 layers → [1, H*W, H*W]

    attn_weight = attn_weight[0].cpu().detach()
    # → [H*W, H*W] trên CPU
```

**Tại sao 8 layers cuối?**
- Transformer có 12 layers, layers đầu học low-level features, layers cuối học semantic
- 8 layers cuối capture semantic affinity tốt hơn → chỉ dùng 8 cuối
- Lấy mean → denoise, stabilizes affinity matrix

#### 5b. Bounding box extraction

```python
box, cnt = scoremap2bbox(
    scoremap=grayscale_cam,
    threshold=args.box_threshold,   # 0.4 (VOC12/GTA5) hoặc 0.7 (COCO14)
    multi_contour_eval=True,        # detect nhiều contours
)
```

**`scoremap2bbox()` chi tiết**:
```python
def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    # Threshold
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),  # relative threshold!
        maxval=255,
        type=cv2.THRESH_BINARY
    )

    # Find contours
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE
    )[0]  # or [1] tùy OpenCV version

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1  # fallback

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]  # largest only

    # Bounding box cho mỗi contour
    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        estimated_boxes.append([x, y, x + w, y + h])

    return np.asarray(estimated_boxes), len(contours)
```
- **Relative threshold**: `thresh = threshold * max(scoremap)` → KHÔNG phải absolute
- `multi_contour_eval=True`: Cho nhiều objects cùng class
- Trả về: `(boxes [N_box, 4], count)`

#### 5c. Affinity mask từ bounding boxes

```python
aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
# [h//16, w//16] — all zeros

for i_ in range(cnt):
    x0_, y0_, x1_, y1_ = box[i_]
    aff_mask[y0_:y1_, x0_:x1_] = 1  # set 1 trong bbox

aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
# [1, H*W] — flatten
```
- Mask = 1 trong object region, 0 ở ngoài
- Dùng để **restrict propagation** — chỉ propagate activation trong object region

#### 5d. Doubly stochastic normalization

```python
aff_mat = attn_weight  # [H*W, H*W]

# Row + Column normalize (2 iterations)
trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)  # column normalize
trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)  # row normalize

for _ in range(2):
    trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

# Symmetrize
trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
```

**Tại sao doubly stochastic?**
- Đảm bảo mỗi token "phân bổ" đúng 1 unit attention cho các tokens khác (row sum = 1)
- Đảm bảo mỗi token "nhận" đúng 1 unit attention từ các tokens khác (col sum = 1)
- 2 iterations: hội tụ nhanh hơn về doubly stochastic matrix
- Symmetrize: attention(i→j) = attention(j→i), làm affinity symmetric

#### 5e. Diffusion (matrix multiplication)

```python
for _ in range(1):  # chỉ 1 lần
    trans_mat = torch.matmul(trans_mat, trans_mat)
```
- `T² = T @ T`: 1 step diffusion → thông tin propagate 2 hops
- Nhiều bước hơn (`_ > 1`) → propagate xa hơn nhưng có thể blur
- Paper dùng 1 bước — balance giữa localization và completeness

#### 5f. Apply bbox mask

```python
trans_mat = trans_mat * aff_mask
# [H*W, H*W] * [1, H*W] → broadcast
# Chỉ giữ connections trong bbox, zero hết ngoài bbox
```

#### 5g. Propagate CAM

```python
cam_to_refine = torch.FloatTensor(grayscale_cam)  # [h//16, w//16]
cam_to_refine = cam_to_refine.view(-1, 1)          # [H*W, 1]

# Affinity propagation
cam_refined = torch.matmul(trans_mat, cam_to_refine)  # [H*W, 1] * [H*W, H*W] → [H*W, 1]
cam_refined = cam_refined.reshape(h // 16, w // 16)

# Upsample to original resolution
cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
```
- `scale_cam_image()`: normalize về [0, 1] rồi resize → (ori_W, ori_H)
- Final CAM shape: `[ori_H, ori_W]`

---

### Bước 6: Save .npy

```python
np.save(out_path, {
    "keys": keys.numpy(),                              # [K] — class IDs
    "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),  # [K, ori_H, ori_W]
})
```

**Output format**:
- `keys`: mảng int — global class indices (0-19 cho VOC12, 0-79 cho COCO14, 0-18 cho GTA5)
- `attn_highres`: refined CAMs, shape `[K, ori_H, ori_W]`, dtype `float16`
- Lưu dưới dạng `.npy` (NumPy dict)

---

## DIFFERENCES BETWEEN DATASETS

| Feature | VOC12 | COCO14 | GTA5/Cityscapes |
|---------|-------|--------|-----------------|
| **Label source** | XML annotation | Split file (`id cls1 cls2...`) | Segmentation mask (PNG) |
| **Class names** | `new_class_names` (20) | `new_class_names_coco` (80) | `CITYSCAPES_PROMPT_NAMES` (19) |
| **Background** | `BACKGROUND_CATEGORY` (25) | `BACKGROUND_CATEGORY_COCO` (22) | `BACKGROUND_CATEGORY` (25) |
| **box_threshold** | 0.4 | 0.7 | 0.4 |
| **max_long_side** | -1 (none) | -1 (none) | 1024 |
| **Image scaling** | 1.0 | 1.0 | 1.0 (configurable) |
| **DAMP support** | Yes | No (original) | Yes |
| **damp_mix_alpha** | No (original) | No | Yes |
| **damp_use_plain_names** | No (original) | No | Yes |
| **Multi-worker** | `multiprocessing.spawn` | `multiprocessing.spawn` | `multiprocessing.spawn` |
| **Skip existing** | No | No | Yes |
| **--no_refine** | No | No | Yes |
| **--max_images** | No | No | Yes |

---

## DAMP PROMPT INTEGRATION — TỔNG HỢP

```
                  ┌─────────────────────┐
                  │  prompt_learner.pth  │
                  │  (from DAMP Phase 3) │
                  └──────────┬──────────┘
                             │
                    load state_dict
                             │
                             ▼
                    extract "ctx" tensor
                    shape: [n_ctx, dim]
                    hoặc [1, n_ctx, dim]
                             │
                  ┌──────────┴──────────┐
                  │  Build prompts:      │
                  │  "X X ... {class}."  │
                  │  Replace X with ctx  │
                  │  → prompt_embeddings │
                  └──────────┬──────────┘
                             │
                  encode through CLIP text transformer
                             │
                             ▼
                    damp_text_features [K, D]
                             │
              ┌──────────────┴──────────────┐
              │ damp_mix_alpha < 1.0?        │
              │                              │
         Yes: │  blend with zero-shot anchor  │
              │  fg = normalize(              │
              │    α*damp + (1-α)*zeroshot)   │
              │                              │
          No: │  fg = damp_text_features      │
              └──────────────┬──────────────┘
                             │
                             ▼
                    fg_text_features [K, D]
                   (dùng thay zeroshot cho CAM)
```

---

## TASKS

### 4.1 Port Grad-CAM module
- Copy `CLIP-ES/pytorch_grad_cam/` → `cam/grad_cam/`
- Update ALL internal imports:
  - `from pytorch_grad_cam.xxx` → `from cam.grad_cam.xxx`
- Files cần update imports:
  - `__init__.py`, `base_cam.py`, `grad_cam.py`, `grad_cam_plusplus.py`, `xgrad_cam.py`, `score_cam.py`, `eigen_cam.py`, `eigen_grad_cam.py`, `layer_cam.py`, `fullgrad_cam.py`, `ablation_cam.py`, `ablation_cam_multilayer.py`, `guided_backprop.py`
  - `utils/image.py`, `utils/model_targets.py`, `utils/reshape_transforms.py`

### 4.2 Port utility files
- Copy `CLIP-ES/utils.py` → `cam/utils.py` (scoremap2bbox, parse_xml_to_dict, calculate_multiple_iou)
- Copy `CLIP-ES/clip_text.py` → `cam/clip_text.py` (tất cả class names, background categories)
- Add `CITYSCAPES_CLASS_NAMES`, `CITYSCAPES_PROMPT_NAMES`, `ID_TO_TRAINID` từ `generate_cams_gta5.py`

### 4.3 Unify CAM generation
- Merge 4 scripts → `cam/generate.py`
- Common functions: `zeroshot_classifier`, `damp_prompt_classifier`, `_encode_text_with_prompt_embeddings`, `build_worker_components`, `ClipOutputTarget`, `img_ms_and_flip`, `reshape_transform`
- Dataset-specific: label parsing (XML vs split file vs mask), class names, thresholds
- GTA5 version có nhiều features nhất (damp_mix_alpha, skip_existing, no_refine, max_images, max_long_side) → lấy làm base

### 4.4 Create CLI entry point
- `generate_cams.py` at project root
- Args: `--dataset`, `--split`, `--clip_model`, `--output_dir`, `--box_threshold`, `--image_scale`, `--max_long_side`, `--damp_prompt_ckpt`, `--damp_n_ctx`, `--damp_mix_alpha`, `--damp_use_plain_names`, `--no_refine`, `--skip_existing`, `--num_workers`, `--max_images`

### 4.5 Port CLIP model modifications
- CRITICAL: `clip/model.py` đã được modify (upsample_pos_emb, encode_image returns attn, forward_last_layer, Transformer collects attn_weights)
- Must use CLIP-ES's `clip/model.py`, NOT vanilla CLIP
- Verify `VisionTransformer.forward(x, H, W)` signature (extra H, W params)
- Verify `Transformer.forward()` returns `(x, attn_weights)`

---

## VERIFICATION CHECKLIST
- [ ] `python generate_cams.py --dataset voc12 --split trainval` generates .npy files
- [ ] `python generate_cams.py --dataset gta5 --split train_full --max_long_side 1024` handles large images
- [ ] `python generate_cams.py --dataset coco14 --split train` handles 80 classes
- [ ] `--damp_prompt_ckpt` loads and applies DAMP prompts correctly
- [ ] `--damp_mix_alpha 0.5` blends DAMP + zero-shot features
- [ ] `--no_refine` skips refinement (saves raw CAM)
- [ ] `--skip_existing` skips already-generated files
- [ ] Output .npy format: `{"keys": [...], "attn_highres": [...]}`
- [ ] CAM values in range [0, 1]
- [ ] Compare output with original CLIP-ES on same input → should match
