"""DAMP trainer (multi-label only: VOC12, COCO14, GTA5/Cityscapes).

Ported from the original DAMP repo and adapted to:
    1. The shared CLIP-ES CLIP backbone (``Damp_es/clip``) — which wraps its
       ``Transformer`` in ``torch.no_grad()`` and skips the last layer. We
       bypass both by iterating ``resblocks`` manually inside the text /
       visual encoders so gradients reach the learnable prompt.
    2. Exporting ``prompt_learner.pth`` after a new best checkpoint, which is
       the bridge artifact consumed by the CLIP-ES CAM generation pipeline.
"""

import datetime
import json
import os
import os.path as osp
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import (
    AverageMeter,
    MetricMeter,
    load_checkpoint,
    load_pretrained_weights,
    save_checkpoint,
)
from timm.models.layers import trunc_normal_

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    download_root = cfg.MODEL.BACKBONE.PATH or os.path.expanduser("~/.cache/clip")
    model_path = clip._download(url, download_root)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return clip.build_model(state_dict or model.state_dict())


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def IM_loss_multilabel(logits_target):
    """Multi-label Information-Maximization loss (sigmoid-based)."""
    logits_target = logits_target.float()
    probs = torch.sigmoid(logits_target).clamp(min=1e-4, max=1.0 - 1e-4)

    sample_entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
    item1 = sample_entropy.sum(dim=1).mean()

    avg_probs = probs.mean(dim=0).clamp(min=1e-4, max=1.0 - 1e-4)
    class_entropy = -(avg_probs * torch.log(avg_probs)
                      + (1.0 - avg_probs) * torch.log(1.0 - avg_probs))
    item2 = class_entropy.sum()

    loss = item1 - item2
    if not torch.isfinite(loss):
        loss = torch.zeros((), device=logits_target.device, dtype=torch.float32)
    return loss


def multilabel_accuracy_from_logits(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    return (preds == targets).float().mean() * 100.0


# ---------------------------------------------------------------------------
# Context decoder (text/visual prompting)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, _ = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):
    def __init__(self, cfg, transformer_width=256, transformer_heads=4,
                 transformer_layers=6, dropout=0.1, **_kwargs):
        super().__init__()
        visual_dim = 1024 if cfg.MODEL.BACKBONE.NAME == 'RN50' else 512

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads, dropout)
            for _ in range(transformer_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):
        visual = self.memory_proj(visual)
        x = self.text_proj(text)
        for layer in self.decoder:
            x = layer(x, visual)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Text / image encoders (gradient-flowing wrappers around CLIP-ES backbone)
# ---------------------------------------------------------------------------

def _run_resblocks(resblocks, x):
    """Iterate all residual attention blocks with full gradient flow.

    The CLIP-ES ``Transformer.forward`` wraps the loop in ``torch.no_grad``
    and skips the last block; that breaks DAMP training because gradients
    must reach the learnable context vectors. We therefore run the blocks
    manually here.
    """
    for blk in resblocks:
        x, _ = blk(x)
    return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = _run_resblocks(self.transformer.resblocks, x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        text_at_eos = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        text_all = torch.einsum('kld,dc->klc', x, self.text_projection)
        return text_at_eos, text_all


class ResNetImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual
        self.attnpool = clip_model.visual.attnpool
        self.num_heads = self.attnpool.num_heads
        self.embed_dim = self.attnpool.k_proj.in_features
        self.spacial_dim = self.encoder.input_resolution // 32
        self.relu = nn.ReLU(inplace=True)
        self.out_indices = (0, 1, 2, 3)

    @autocast()
    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.encoder.conv1, self.encoder.bn1),
                (self.encoder.conv2, self.encoder.bn2),
                (self.encoder.conv3, self.encoder.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.encoder.avgpool(x)
            return x

        x = x.type(self.encoder.conv1.weight.dtype)
        x = stem(x)
        outs = []
        x = self.encoder.layer1(x); outs.append(x)
        x = self.encoder.layer2(x); outs.append(x)
        x = self.encoder.layer3(x); outs.append(x)
        x = self.encoder.layer4(x); outs.append(x)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.attnpool.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(
            self.attnpool.positional_embedding[1:, ].reshape(
                1, self.spacial_dim, self.spacial_dim, self.embed_dim
            ).permute(0, 3, 1, 2),
            size=(H, W), mode='bilinear',
        )
        spatial_pos = spatial_pos.reshape(self.embed_dim, H * W).permute(1, 0)
        pos_embed = torch.cat([cls_pos, spatial_pos], dim=0)
        x = x + pos_embed[:, None, :]

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.attnpool.q_proj.weight,
            k_proj_weight=self.attnpool.k_proj.weight,
            v_proj_weight=self.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([
                self.attnpool.q_proj.bias,
                self.attnpool.k_proj.bias,
                self.attnpool.v_proj.bias,
            ]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
            out_proj_weight=self.attnpool.c_proj.weight,
            out_proj_bias=self.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training, need_weights=False,
        )
        x = x.permute(1, 2, 0)  # NC(1+HW)
        x_global = x[:, :, 0]
        x_local = x[:, :, 1:].reshape(B, -1, H, W)

        final_outs = [outs[i] for i in self.out_indices]
        final_outs.append([x_global, x_local])
        return tuple(final_outs)


class VITImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual

    @autocast()
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)  # [B, C, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, grid**2, C]
        cls = (self.encoder.class_embedding.to(x.dtype)
               + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device))
        x = torch.cat([cls, x], dim=1)
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        features.append(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = _run_resblocks(self.encoder.transformer.resblocks, x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        features.append(x)

        x = self.encoder.ln_post(x)
        features.append(x)
        if self.encoder.proj is not None:
            x = x @ self.encoder.proj

        global_embedding = x[:, 0]
        visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        features.append([global_embedding, visual_embedding])
        return tuple(features)


# ---------------------------------------------------------------------------
# Prompt learner + CustomCLIP
# ---------------------------------------------------------------------------

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DAMP.N_CTX

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"
        )

        target_domain = (cfg.DATASET.TARGET_DOMAINS[0]
                         if cfg.DATASET.TARGET_DOMAINS else "target")
        naive_prompt_prefix = f'a {target_domain} photo of a'.replace("_", " ")

        if cfg.TRAINER.DAMP.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        self.gamma_t = nn.Parameter(torch.ones(1) * 0.01)
        self.gamma_v = nn.Parameter(torch.ones(1) * 0.01)

        prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        naive_prompts = [naive_prompt_prefix + " " + name + "." for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(f'Prompts: "{prompts[0]}"')
        print(f'Naive Prompts: "{naive_prompts[0]}"')

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            naive_embedding = clip_model.token_embedding(naive_tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])          # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLASS + EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.DAMP.CSC
        self.tokenized_prompts = tokenized_prompts
        self.naive_tokenized_prompts = naive_tokenized_prompts
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding

    @autocast()
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model,
                 context_feature='attention',
                 use_visual_prompt_generator=True,
                 use_text_prompt_generator=True):
        super().__init__()
        backbone = cfg.MODEL.BACKBONE.NAME
        if backbone in ('RN50', 'RN101'):
            self.image_encoder = ResNetImageEncoder(clip_model)
        else:
            self.image_encoder = VITImageEncoder(clip_model)

        self.text_encoder = TextEncoder(clip_model)
        self.context_decoder = ContextDecoder(cfg)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.context_feature = context_feature
        self.use_visual_prompt_generator = use_visual_prompt_generator
        self.use_text_prompt_generator = use_text_prompt_generator

        naive_text_embedding = self.text_encoder(
            self.prompt_learner.naive_embedding,
            self.prompt_learner.naive_tokenized_prompts,
        )[0]
        self.register_buffer("naive_text_embedding", naive_text_embedding)

    @autocast()
    def forward(self, img, ind=False, pse=False, fea=False):
        x = self.image_encoder(img)
        global_feat, visual_embeddings = x[-1]
        B, C, H, W = visual_embeddings.shape

        visual_contexts = torch.cat([
            global_feat.reshape(B, C, 1),
            visual_embeddings.reshape(B, C, H * W),
        ], dim=2).permute(0, 2, 1)  # B, 1+HW, C

        raw_prompt = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_embeddings, text_contexts = self.text_encoder(raw_prompt, tokenized_prompts)
        text_embeddings = text_embeddings.expand(B, -1, -1)
        text_contexts = text_contexts.expand(B, -1, -1, -1)[:, 0, :self.prompt_learner.n_ctx, :]

        if self.use_visual_prompt_generator:
            vis_prompt_diff = self.context_decoder(
                global_feat.reshape(B, C, 1).permute(0, 2, 1), text_contexts
            ).permute(0, 2, 1).reshape(B, C)
            updated_vision_embedding = global_feat + self.prompt_learner.gamma_v * vis_prompt_diff
        else:
            updated_vision_embedding = global_feat

        if self.use_text_prompt_generator:
            text_diff = self.context_decoder(text_embeddings, visual_contexts)
            updated_text_embeddings = text_embeddings + self.prompt_learner.gamma_t * text_diff
        else:
            updated_text_embeddings = text_embeddings

        visual = F.normalize(updated_vision_embedding, dim=1, p=2)
        text = F.normalize(updated_text_embeddings, dim=2, p=2)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.einsum('bc,bkc->bk', visual, text)

        out = [logits]
        if ind:
            logits_ind = torch.einsum('ac,bkc->abk', visual, text).mean(dim=-1) * logit_scale
            out.append(logits_ind)
        if pse:
            img_f = global_feat / global_feat.norm(dim=-1, keepdim=True)
            txt_f = self.naive_text_embedding / self.naive_text_embedding.norm(dim=-1, keepdim=True)
            out.append(logit_scale * img_f @ txt_f.t())
        if fea:
            out.append(global_feat)
            out.append(updated_vision_embedding)
        return tuple(out)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@TRAINER_REGISTRY.register()
class DAMP(TrainerXU):

    # ............................................................. setup
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DAMP.PREC in ("fp16", "fp32", "amp")

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        tfm_train_strong = build_transform(
            cfg, is_train=True, choices=cfg.TRAINER.DAMP.STRONG_TRANSFORMS
        )
        self.dm = DataManager(cfg, custom_tfm_train=[tfm_train, tfm_train_strong])
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.lab2cname = self.dm.lab2cname

        self.eval_threshold = 0.5
        self.pseudo_threshold = 0.55
        self.multi_label_lookup = self._build_multilabel_lookup()

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Classnames ({len(classnames)}): {classnames}")

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.DAMP.PREC in ("fp32", "amp"):
            clip_model.float()

        print("Building CustomCLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.n_cls = self.model.prompt_learner.n_cls

        print("Turning off gradients in CLIP encoders")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "context_decoder" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS_PRO:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS_PRO)
        if cfg.MODEL.INIT_WEIGHTS_CTX:
            load_pretrained_weights(self.model.context_decoder, cfg.MODEL.INIT_WEIGHTS_CTX)

        self.model.to(self.device)

        self.optim_p = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched_p = build_lr_scheduler(self.optim_p, cfg.OPTIM)
        self.optim_c = build_optimizer(self.model.context_decoder, cfg.OPTIM_C)
        self.sched_c = build_lr_scheduler(self.optim_c, cfg.OPTIM_C)

        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim_p, self.sched_p)
        self.register_model("context_decoder", self.model.context_decoder,
                            self.optim_c, self.sched_c)

        self.scaler = GradScaler() if cfg.TRAINER.DAMP.PREC == "amp" else None

    # ...................................................... multilabel I/O
    @staticmethod
    def _first_existing_dir(candidates, default):
        for p in candidates:
            if osp.isdir(p):
                return p
        return default

    _SYNTHIA_CIT19_TO_LOCAL = {
        cid: i for i, cid in enumerate(sorted(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18}
        ))
    }

    def _normalize_label_ids(self, labels):
        """Remap raw IDs to contiguous train IDs for the current dataset."""
        id_to_trainid = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
            22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
            32: 17, 33: 18,
        }
        labels = [int(v) for v in labels]
        if any(v > 18 and v != 255 for v in labels):
            mapped = [id_to_trainid[v] for v in labels if v in id_to_trainid]
        else:
            mapped = labels

        if self.cfg.DATASET.NAME == "SYNTHIA":
            mapped = [
                self._SYNTHIA_CIT19_TO_LOCAL[v]
                for v in mapped
                if v in self._SYNTHIA_CIT19_TO_LOCAL
            ]
        else:
            mapped = [v for v in mapped if 0 <= v < self.num_classes]
        return sorted(set(mapped))

    def _load_multilabel_json(self, path):
        if not osp.isfile(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        out = {}
        for k, labels in data.items():
            if not isinstance(labels, list):
                continue
            if len(labels) == self.num_classes and all(int(v) in (0, 1) for v in labels):
                norm = [i for i, flag in enumerate(labels)
                        if int(flag) > 0 and i < self.num_classes]
            elif labels and all(isinstance(v, (int, float)) for v in labels):
                norm = self._normalize_label_ids(labels)
            else:
                norm = []
            out[osp.basename(k)] = norm
        return out

    def _build_multilabel_lookup(self):
        """Walk dataset-specific multilabel JSONs and build a path→multi-hot map."""
        root = osp.abspath(osp.expanduser(self.cfg.DATASET.ROOT))
        dataset_dir = osp.join(root, "data")

        raw_gta5_dir = self._first_existing_dir(
            [osp.join(dataset_dir, "raw", "gta5_kaggle_full"),
             osp.join(root, "gta5-kaggle-full")],
            default=osp.join(dataset_dir, "raw", "gta5_kaggle_full"),
        )
        raw_city_dir = self._first_existing_dir(
            [osp.join(dataset_dir, "raw", "cityscapes"),
             osp.join(root, "cityscapes")],
            default=osp.join(dataset_dir, "raw", "cityscapes"),
        )
        raw_synthia_dir = self._first_existing_dir(
            [osp.join(dataset_dir, "raw", "synthia"),
             osp.join(root, "synthia")],
            default=osp.join(dataset_dir, "raw", "synthia"),
        )
        processed_dir = osp.join(dataset_dir, "processed")

        split_specs = [
            # (image_dir, split_file, multilabel_json)
            (osp.join(raw_gta5_dir, "images"),
             osp.join(raw_gta5_dir, "splits", "train_full.txt"),
             osp.join(processed_dir, "gta5_multilabel", "multilabel.json")),
            (osp.join(raw_city_dir, "images"),
             osp.join(raw_city_dir, "splits", "train.txt"),
             osp.join(processed_dir, "cityscapes_multilabel", "train_multilabel.json")),
            (osp.join(raw_city_dir, "images"),
             osp.join(raw_city_dir, "splits", "val.txt"),
             osp.join(processed_dir, "cityscapes_multilabel", "val_multilabel.json")),
            (osp.join(raw_synthia_dir, "images"),
             osp.join(raw_synthia_dir, "splits", "train.txt"),
             osp.join(processed_dir, "synthia_multilabel", "multilabel.json")),
        ]

        lookup = {}
        for image_dir, split_file, ml_file in split_specs:
            if not osp.isfile(split_file):
                continue
            ml_map = self._load_multilabel_json(ml_file)
            with open(split_file, "r") as f:
                names = [osp.basename(line.strip()) for line in f if line.strip()]
            for name in names:
                impath = osp.abspath(osp.join(image_dir, name))
                vec = torch.zeros(self.num_classes, dtype=torch.float32)
                for cls_id in ml_map.get(name, []):
                    if 0 <= int(cls_id) < self.num_classes:
                        vec[int(cls_id)] = 1.0
                lookup[impath] = vec
        return lookup

    def _build_multihot_labels(self, impaths, fallback_labels):
        if impaths is None:
            impaths = [""] * int(fallback_labels.size(0))
        if isinstance(impaths, str):
            impaths = [impaths]

        labels = torch.zeros(len(impaths), self.num_classes,
                             dtype=torch.float32, device=self.device)
        fallback_cpu = (fallback_labels.detach().cpu().tolist()
                        if fallback_labels is not None else None)

        for i, impath in enumerate(impaths):
            key = osp.abspath(impath) if impath else ""
            if key in self.multi_label_lookup:
                labels[i] = self.multi_label_lookup[key].to(self.device)
            elif fallback_cpu is not None:
                cls_id = int(fallback_cpu[i])
                if 0 <= cls_id < self.num_classes:
                    labels[i, cls_id] = 1.0
        return labels

    # ............................................................. batching
    def parse_batch_train(self, batch_x, batch_u):
        image_x = batch_x["img"].to(self.device)
        image_x2 = batch_x["img2"].to(self.device)
        label_x = batch_x["label"].to(self.device)
        image_u = batch_u["img"].to(self.device)
        image_u2 = batch_u["img2"].to(self.device)
        label_u = batch_u["label"].to(self.device)

        impath_x = batch_x.get("impath", None)
        impath_u = batch_u.get("impath", None)
        label_x = self._build_multihot_labels(impath_x, label_x)
        label_u = self._build_multihot_labels(impath_u, label_u)

        return image_x, image_x2, label_x, image_u, image_u2, label_u

    # .................................................... train/test loops
    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.threshold = self.cfg.TRAINER.DAMP.TAU
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time, data_time = AverageMeter(), AverageMeter()

        len_x = len(self.train_loader_x)
        len_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_x if self.cfg.DATASET.NAME == "OfficeHome" else 500
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_x, len_u)
        else:
            raise ValueError(f"Unknown COUNT_ITER: {self.cfg.TRAIN.COUNT_ITER}")

        it_x = iter(self.train_loader_x)
        it_u = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(it_x)
            except StopIteration:
                it_x = iter(self.train_loader_x); batch_x = next(it_x)
            try:
                batch_u = next(it_u)
            except StopIteration:
                it_u = iter(self.train_loader_u); batch_u = next(it_u)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 \
                    or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = (self.num_batches - self.batch_idx - 1
                             + (self.max_epoch - self.epoch - 1) * self.num_batches)
                eta = str(datetime.timedelta(seconds=int(batch_time.avg * nb_remain)))
                print(
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]"
                    f"[{self.batch_idx + 1}/{self.num_batches}]\t"
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"eta {eta}\t{losses}\tlr {self.get_current_lr():.6e}"
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()

    # ......................................................... core losses
    def forward_backward(self, batch_x, batch_u):
        image_x, image_x2, label, image_u, image_u2, label_u = \
            self.parse_batch_train(batch_x, batch_u)

        prec = self.cfg.TRAINER.DAMP.PREC
        if prec != "amp":
            raise NotImplementedError(
                "DAMP trainer currently only implements the AMP code path"
            )

        with autocast():
            output_x, output_x_ind = self.model(image_x, ind=True, pse=False)
            output_u, output_u_ind, pseudo_label_logits = self.model(
                image_u, ind=True, pse=True
            )
            output_x2 = self.model(image_x2)[0]
            output_u2 = self.model(image_u2)[0]

            mix_lambda = self.epoch / max(self.max_epoch, 1)

            (loss_x, loss_x2, loss_u, loss_u2,
             loss_im, pseudo_info) = self._multilabel_losses(
                output_x, output_x2, output_u, output_u2,
                pseudo_label_logits, label, mix_lambda,
            )

            # Individuation loss — instance-level, identical for both modes.
            B_x = image_x.size(0)
            B_u = image_u.size(0)
            x_ind_label = torch.arange(B_x, dtype=torch.long, device=self.device)
            u_ind_label = torch.arange(B_u, dtype=torch.long, device=self.device)
            loss_x_ind = (F.cross_entropy(output_x_ind, x_ind_label)
                          + F.cross_entropy(output_x_ind.permute(1, 0), x_ind_label)) / 2.0
            loss_u_ind = (F.cross_entropy(output_u_ind, u_ind_label)
                          + F.cross_entropy(output_u_ind.permute(1, 0), u_ind_label)) / 2.0
            loss_ind = loss_x_ind + loss_u_ind

            if not torch.isfinite(loss_im):
                loss_im = output_u.new_tensor(0.0)

            U = self.cfg.TRAINER.DAMP.U
            loss = (loss_x + loss_x2) + U * (loss_u + loss_u2) + loss_ind + 0.1 * loss_im
            if not torch.isfinite(loss):
                loss = (loss_x + loss_x2) + U * (loss_u + loss_u2) + loss_ind

            self.optim_p.zero_grad()
            self.optim_c.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim_p)
            self.scaler.step(self.optim_c)
            self.scaler.update()

        summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "loss_ind": loss_ind.item(),
            "loss_im": loss_im.item() if torch.is_tensor(loss_im) else float(loss_im),
            "gamma_v": self.model.prompt_learner.gamma_v.item(),
            "gamma_t": self.model.prompt_learner.gamma_t.item(),
        }
        summary.update(pseudo_info)
        self.update_lr()
        return summary

    def _multilabel_losses(self, output_x, output_x2, output_u, output_u2,
                           pseudo_label_logits, label, mix_lambda):
        pseudo_label = (
            torch.sigmoid(output_u) * mix_lambda
            + torch.sigmoid(pseudo_label_logits) * (1 - mix_lambda)
        ).detach()
        pseudo_bin = (pseudo_label >= self.pseudo_threshold).float()
        confident = pseudo_bin.sum(dim=1) > 0

        loss_x = F.binary_cross_entropy_with_logits(output_x, label)
        loss_x2 = F.binary_cross_entropy_with_logits(output_x2, label)

        bce_u = F.binary_cross_entropy_with_logits(output_u, pseudo_bin, reduction="none")
        bce_u2 = F.binary_cross_entropy_with_logits(output_u2, pseudo_bin, reduction="none")
        if confident.any():
            cw = confident.float().unsqueeze(1)
            norm = cw.sum() * self.n_cls
            loss_u = (bce_u * cw).sum() / norm
            loss_u2 = (bce_u2 * cw).sum() / norm
        else:
            loss_u = output_u.new_tensor(0.0)
            loss_u2 = output_u2.new_tensor(0.0)

        loss_im = IM_loss_multilabel(output_u)

        info = {
            "acc_x": multilabel_accuracy_from_logits(
                output_x, label, threshold=self.eval_threshold).item(),
            "pseudo_pos_rate": pseudo_bin.float().mean().item() * 100.0,
            "pseudo_conf_rate": confident.float().mean().item() * 100.0,
        }
        return loss_x, loss_x2, loss_u, loss_u2, loss_im, info

    # ............................................................ test
    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        if split is None:
            split = self.cfg.TEST.SPLIT
        print(f"Do evaluation on {split} set")
        return self._test_multilabel(split)

    def _test_multilabel(self, split):
        total = exact = label_correct = label_total = 0
        K = self.num_classes
        tp = torch.zeros(K, dtype=torch.float64)
        fp = torch.zeros(K, dtype=torch.float64)
        fn = torch.zeros(K, dtype=torch.float64)

        for batch in self.test_loader:
            inp = batch["img"].to(self.device)
            impaths = batch.get("impath", [])
            lb = batch.get("label", None)
            if lb is not None:
                lb = lb.to(self.device)
            label = self._build_multihot_labels(impaths, lb).detach().cpu()

            out = self.model_inference(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred = (torch.sigmoid(out) >= self.eval_threshold).float().detach().cpu()

            total += pred.size(0)
            exact += (pred == label).all(dim=1).sum().item()
            label_correct += (pred == label).sum().item()
            label_total += label.numel()
            tp += ((pred == 1) & (label == 1)).sum(dim=0).double()
            fp += ((pred == 1) & (label == 0)).sum(dim=0).double()
            fn += ((pred == 0) & (label == 1)).sum(dim=0).double()

        eps = 1e-12
        mtp, mfp, mfn = tp.sum(), fp.sum(), fn.sum()
        mp = mtp / (mtp + mfp + eps)
        mr = mtp / (mtp + mfn + eps)
        micro_f1 = 2 * mp * mr / (mp + mr + eps)
        class_f1 = 2 * (tp / (tp + fp + eps)) * (tp / (tp + fn + eps)) \
                   / ((tp / (tp + fp + eps)) + (tp / (tp + fn + eps)) + eps)
        macro_f1 = class_f1.mean()

        label_acc = (label_correct / max(label_total, 1)) * 100.0
        exact_acc = (exact / max(total, 1)) * 100.0
        print("=> result")
        print(f"* total: {total:,}")
        print(f"* multilabel_acc: {label_acc:.2f}%")
        print(f"* exact_match_acc: {exact_acc:.2f}%")
        print(f"* micro_f1: {float(micro_f1) * 100.0:.2f}%")
        print(f"* macro_f1: {float(macro_f1) * 100.0:.2f}%")

        self.write_scalar(f"{split}/multilabel_acc", label_acc, self.epoch)
        self.write_scalar(f"{split}/exact_match_acc", exact_acc, self.epoch)
        self.write_scalar(f"{split}/micro_f1", float(micro_f1) * 100.0, self.epoch)
        self.write_scalar(f"{split}/macro_f1", float(macro_f1) * 100.0, self.epoch)
        return float(micro_f1) * 100.0

    # ............................................................ save/load
    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()
        for name in names:
            save_checkpoint(
                {
                    "state_dict": self._models[name].state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": (self._optims[name].state_dict()
                                  if self._optims[name] is not None else None),
                    "scheduler": (self._scheds[name].state_dict()
                                  if self._scheds[name] is not None else None),
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        freq = self.cfg.TRAIN.CHECKPOINT_FREQ
        meet_freq = (freq > 0 and (self.epoch + 1) % freq == 0)

        if do_test:
            curr = self.test()
            if curr > self.best_result:
                self.best_result = curr
                self.save_model(self.epoch, self.output_dir,
                                model_name="model-best.pth.tar")
                self._export_prompt_learner_bridge()
            self.set_model_mode("train")

        if meet_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def _export_prompt_learner_bridge(self):
        """Export prompt_learner state dict for the CLIP-ES CAM pipeline.

        This file is the canonical hand-off artifact between phases:
        ``generate_cams.py --damp_prompt_ckpt <output_dir>/prompt_learner.pth``.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        out = osp.join(self.output_dir, "prompt_learner.pth")
        state = self.model.prompt_learner.state_dict()
        torch.save(
            {"state_dict": state, "n_ctx": self.model.prompt_learner.n_ctx},
            out,
        )
        print(f"[bridge] Saved prompt_learner weights → {out}")

    def load_model(self, directory, epoch=None):
        if not directory:
            print("load_model() skipped: no pretrained model given")
            return

        model_file = ("model.pth.tar-" + str(epoch)
                      if epoch is not None else "model-best.pth.tar")
        for name in self.get_model_names():
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            ckpt = load_checkpoint(model_path)
            state_dict = ckpt["state_dict"]
            # Fixed token vectors are dataset-dependent; rebuild from current cfg.
            for k in ("token_prefix", "token_suffix"):
                state_dict.pop(k, None)
            print(f'Loading weights for {name} from "{model_path}" '
                  f'(epoch = {ckpt["epoch"]})')
            self._models[name].load_state_dict(state_dict, strict=False)
