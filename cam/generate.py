"""Unified CLIP-ES CAM generation for VOC12, COCO14, and GTA5/Cityscapes.

Merges the four original scripts into one dataset-agnostic pipeline.
"""

import os
import os.path as osp

import clip
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lxml import etree
from PIL import Image
from torch import multiprocessing
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from cam.grad_cam import GradCAM
from cam.grad_cam.utils.image import scale_cam_image
from cam.utils import parse_xml_to_dict, scoremap2bbox
from cam.clip_text import (
    BACKGROUND_CATEGORY,
    BACKGROUND_CATEGORY_COCO,
    CITYSCAPES_CLASS_NAMES,
    CITYSCAPES_PROMPT_NAMES,
    ID_TO_TRAINID,
    class_names,
    new_class_names,
    new_class_names_coco,
)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# ---------------------------------------------------------------------------
# Dataset registry — maps dataset name to its class-name lists & bg list
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "voc12": {
        "fg_classnames": new_class_names,
        "plain_classnames": class_names,
        "bg_categories": BACKGROUND_CATEGORY,
        "box_threshold": 0.4,
        "max_long_side": -1,
    },
    "coco14": {
        "fg_classnames": new_class_names_coco,
        "plain_classnames": None,
        "bg_categories": BACKGROUND_CATEGORY_COCO,
        "box_threshold": 0.7,
        "max_long_side": -1,
    },
    "gta5": {
        "fg_classnames": CITYSCAPES_PROMPT_NAMES,
        "plain_classnames": CITYSCAPES_CLASS_NAMES,
        "bg_categories": BACKGROUND_CATEGORY,
        "box_threshold": 0.4,
        "max_long_side": 1024,
    },
    "cityscapes": {
        "fg_classnames": CITYSCAPES_PROMPT_NAMES,
        "plain_classnames": CITYSCAPES_CLASS_NAMES,
        "bg_categories": BACKGROUND_CATEGORY,
        "box_threshold": 0.4,
        "max_long_side": 1024,
    },
    "synthia": {
        "fg_classnames": CITYSCAPES_PROMPT_NAMES,
        "plain_classnames": CITYSCAPES_CLASS_NAMES,
        "bg_categories": BACKGROUND_CATEGORY,
        "box_threshold": 0.4,
        "max_long_side": 1024,
    },
}


# ---------------------------------------------------------------------------
# ViT reshape transform for Grad-CAM
# ---------------------------------------------------------------------------

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform_resize(h, w):
    return Compose([
        Resize((h, w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def _scaled_hw(ori_height, ori_width, scale=1.0, max_long_side=-1, patch_size=16):
    h = max(1, int(round(float(ori_height) * float(scale))))
    w = max(1, int(round(float(ori_width) * float(scale))))

    if max_long_side is not None and int(max_long_side) > 0:
        cur_long = max(h, w)
        if cur_long > int(max_long_side):
            ratio = float(max_long_side) / float(cur_long)
            h = max(1, int(round(h * ratio)))
            w = max(1, int(round(w * ratio)))

    h = int(np.ceil(h / patch_size) * patch_size)
    w = int(np.ceil(w / patch_size) * patch_size)
    return h, w


def img_ms_and_flip(img_path, ori_height, ori_width, scales=None,
                    patch_size=16, max_long_side=-1):
    if scales is None:
        scales = [1.0]

    all_imgs = []
    for scale in scales:
        resized_h, resized_w = _scaled_hw(
            ori_height, ori_width,
            scale=scale, max_long_side=max_long_side, patch_size=patch_size,
        )
        preprocess = _transform_resize(resized_h, resized_w)
        image = preprocess(Image.open(img_path))
        image_ori = image
        image_flip = torch.flip(image, [-1])
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs


# ---------------------------------------------------------------------------
# Text encoding helpers
# ---------------------------------------------------------------------------

def zeroshot_classifier(classnames, templates, model, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()


def _encode_text_with_prompt_embeddings(model, prompt_embeddings, tokenized_prompts):
    x = prompt_embeddings + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    if isinstance(x, (tuple, list)):
        x = x[0]
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x).type(model.dtype)
    text_features = x[
        torch.arange(x.shape[0], device=tokenized_prompts.device),
        tokenized_prompts.argmax(dim=-1)
    ] @ model.text_projection
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def damp_prompt_classifier(classnames, model, ckpt_path, device, n_ctx=-1):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    ctx = checkpoint.get("ctx", None)
    if ctx is None:
        for key in checkpoint:
            if key.endswith(".ctx"):
                ctx = checkpoint[key]
                break

    if ctx is None:
        raise KeyError("Cannot find ctx in DAMP prompt checkpoint")

    if n_ctx <= 0:
        n_ctx = ctx.shape[-2]

    prompt_prefix = " ".join(["X"] * n_ctx)
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        prompt_embeddings = model.token_embedding(tokenized_prompts).type(model.dtype)

    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(len(classnames), -1, -1)
    elif ctx.dim() == 3 and ctx.shape[0] == 1:
        ctx = ctx.expand(len(classnames), -1, -1)
    elif ctx.dim() == 3 and ctx.shape[0] != len(classnames):
        raise ValueError(
            "Mismatch between checkpoint ctx first dim and class count: "
            f"{ctx.shape[0]} vs {len(classnames)}"
        )

    ctx = ctx[:, :n_ctx, :].to(device=device, dtype=model.dtype)
    prompt_embeddings[:, 1:1+n_ctx, :] = ctx

    text_features = _encode_text_with_prompt_embeddings(
        model, prompt_embeddings, tokenized_prompts
    )
    return text_features


# ---------------------------------------------------------------------------
# Grad-CAM target
# ---------------------------------------------------------------------------

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# ---------------------------------------------------------------------------
# Label parsing — each dataset reads labels differently
# ---------------------------------------------------------------------------

def _parse_labels_voc12(im_name, args, fg_classnames):
    img_path = osp.join(args.img_root, im_name)
    xmlfile = img_path.replace('/JPEGImages', '/Annotations')
    xmlfile = xmlfile.replace('.jpg', '.xml')
    with open(xmlfile) as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)["annotation"]

    ori_width = int(data['size']['width'])
    ori_height = int(data['size']['height'])

    label_list = []
    label_id_list = []
    for obj in data["object"]:
        obj_name = new_class_names[class_names.index(obj["name"])]
        if obj_name not in label_list:
            label_list.append(obj_name)
            label_id_list.append(new_class_names.index(obj_name))

    return img_path, ori_height, ori_width, label_list, label_id_list


def _parse_labels_coco14(im_name, label_ids, args, fg_classnames):
    img_path = osp.join(args.img_root, im_name)
    ori_image = Image.open(img_path)
    ori_height, ori_width = np.asarray(ori_image).shape[:2]

    label_id_list = [int(lid) for lid in label_ids]
    label_list = [new_class_names_coco[lid] for lid in label_id_list]
    return img_path, ori_height, ori_width, label_list, label_id_list


def _normalize_label_ids_gta5(raw_ids):
    raw_ids = [int(v) for v in raw_ids]
    if any(v > 18 and v not in (255, -1) for v in raw_ids):
        labels = [ID_TO_TRAINID[v] for v in raw_ids if v in ID_TO_TRAINID]
    else:
        labels = [v for v in raw_ids if 0 <= v < len(CITYSCAPES_CLASS_NAMES)]
    return sorted(set(labels))


def _extract_image_labels_from_mask(mask_path):
    if not osp.isfile(mask_path):
        return []
    mask = np.array(Image.open(mask_path), dtype=np.int64)
    labels = np.unique(mask)
    return _normalize_label_ids_gta5(labels.tolist())


_SYNTHIA_VALID_CIT19_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18}


def _extract_synthia_labels_from_mask(mask_path):
    """Return Cityscapes-19 train IDs present in the mask (only the 16 valid ones)."""
    if not osp.isfile(mask_path):
        return []
    mask = np.array(Image.open(mask_path), dtype=np.int64)
    raw_ids = np.unique(mask)
    return sorted(int(v) for v in raw_ids if int(v) in _SYNTHIA_VALID_CIT19_IDS)


def _resolve_path(root, entry):
    candidates = [entry, osp.basename(entry)]
    for name in candidates:
        path = osp.join(root, name)
        if osp.isfile(path):
            return path
        stem, ext = osp.splitext(name)
        if ext == "":
            for guess_ext in [".png", ".jpg", ".jpeg"]:
                path_guess = osp.join(root, stem + guess_ext)
                if osp.isfile(path_guess):
                    return path_guess
    return osp.join(root, osp.basename(entry))


# ---------------------------------------------------------------------------
# Worker components
# ---------------------------------------------------------------------------

def build_worker_components(args, device):
    ds_cfg = DATASET_CONFIGS[args.dataset]
    model, _ = clip.load(args.model, device=device)

    bg_text_features = zeroshot_classifier(
        ds_cfg["bg_categories"], ["a clean origami {}."], model, device
    )

    if args.damp_prompt_ckpt:
        print("Using DAMP prompt checkpoint:", args.damp_prompt_ckpt)
        damp_use_plain = bool(getattr(args, "damp_use_plain_names", True))
        if ds_cfg["plain_classnames"] is not None and damp_use_plain:
            damp_classnames = ds_cfg["plain_classnames"]
        else:
            damp_classnames = ds_cfg["fg_classnames"]
        print("DAMP classnames:", "plain" if damp_use_plain else "synonym")

        damp_text_features = damp_prompt_classifier(
            damp_classnames, model, args.damp_prompt_ckpt, device, args.damp_n_ctx,
        )

        damp_mix_alpha = max(0.0, min(1.0, float(getattr(args, "damp_mix_alpha", 1.0))))
        if damp_mix_alpha < 1.0:
            anchor_text_features = zeroshot_classifier(
                ds_cfg["fg_classnames"], ["a clean origami {}."], model, device,
            )
            fg_text_features = F.normalize(
                damp_mix_alpha * damp_text_features
                + (1.0 - damp_mix_alpha) * anchor_text_features,
                dim=-1,
            )
            print(f"DAMP/zero-shot mix alpha={damp_mix_alpha:.2f}")
        else:
            fg_text_features = damp_text_features
    else:
        fg_text_features = zeroshot_classifier(
            ds_cfg["fg_classnames"], ["a clean origami {}."], model, device,
        )

    target_layers = [model.visual.transformer.resblocks[-1].ln_1]
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)
    return model, bg_text_features, fg_text_features, cam


# ---------------------------------------------------------------------------
# Attention-based affinity refinement
# ---------------------------------------------------------------------------

def refine_cam_with_attention(grayscale_cam, attn_weight, box_threshold, h, w,
                              ori_width, ori_height):
    box, cnt = scoremap2bbox(
        scoremap=grayscale_cam, threshold=box_threshold, multi_contour_eval=True,
    )
    aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
    for i_ in range(cnt):
        x0_, y0_, x1_, y1_ = box[i_]
        aff_mask[y0_:y1_, x0_:x1_] = 1

    aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat * aff_mask

    cam_to_refine = torch.FloatTensor(grayscale_cam)
    cam_to_refine = cam_to_refine.view(-1, 1)

    cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
    cam_refined = cam_refined.cpu().numpy().astype(np.float32)
    cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
    return cam_refined_highres


# ---------------------------------------------------------------------------
# Core per-image CAM generation
# ---------------------------------------------------------------------------

def generate_cam_for_image(model, cam_method, fg_text_features, bg_text_features,
                           img_path, ori_height, ori_width, label_list,
                           label_id_list, args, device, ds_cfg):
    image_scale = float(getattr(args, "image_scale", 1.0))
    max_long_side = int(getattr(args, "max_long_side", ds_cfg["max_long_side"]))
    no_refine = bool(getattr(args, "no_refine", False))
    box_threshold = float(getattr(args, "box_threshold", ds_cfg["box_threshold"]))

    ms_imgs = img_ms_and_flip(
        img_path, ori_height, ori_width,
        scales=[image_scale], max_long_side=max_long_side,
    )
    ms_imgs = [ms_imgs[0]]

    refined_cam_all_scales = []

    for image in ms_imgs:
        image = image.unsqueeze(0)
        h, w = image.shape[-2], image.shape[-1]
        image = image.to(device)
        image_features, attn_weight_list = model.encode_image(image, h, w)

        refined_cam_to_save = []
        keys = []

        bg_features_temp = bg_text_features.to(device)
        fg_features_temp = fg_text_features[label_id_list].to(device)
        text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
        input_tensor = [image_features, text_features_temp.to(device), h, w]

        attn_weight = None

        for idx, _ in enumerate(label_list):
            keys.append(label_id_list[idx])
            targets = [ClipOutputTarget(idx)]

            grayscale_cam, logits_per_image, attn_weight_last = cam_method(
                input_tensor=input_tensor, targets=targets, target_size=None,
            )

            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))

            if no_refine:
                refined_cam_to_save.append(torch.tensor(grayscale_cam_highres))
                continue

            if idx == 0:
                attn_weight_list.append(attn_weight_last)
                attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
                attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                attn_weight = torch.mean(attn_weight, dim=0)
                attn_weight = attn_weight[0].cpu().detach()
            attn_weight = attn_weight.float()

            cam_refined_highres = refine_cam_with_attention(
                grayscale_cam, attn_weight, box_threshold,
                h, w, ori_width, ori_height,
            )
            refined_cam_to_save.append(torch.tensor(cam_refined_highres))

        keys = torch.tensor(keys)
        refined_cam_all_scales.append(torch.stack(refined_cam_to_save, dim=0))

    refined_cam_all_scales = refined_cam_all_scales[0]
    return keys, refined_cam_all_scales


# ---------------------------------------------------------------------------
# Per-worker entry point
# ---------------------------------------------------------------------------

def split_dataset(dataset, n_splits):
    if n_splits <= 1 or len(dataset) == 0:
        return [dataset]
    n_splits = min(int(n_splits), len(dataset))
    part = len(dataset) // n_splits
    dataset_list = []
    start = 0
    for _ in range(n_splits - 1):
        end = start + part
        dataset_list.append(dataset[start:end])
        start = end
    dataset_list.append(dataset[start:])
    return dataset_list


def perform(process_id, dataset_list, args, all_label_list=None):
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        device_id = "cuda:{}".format(process_id % n_gpus)
    else:
        device_id = "cpu"

    databin = dataset_list[process_id]
    label_bin = all_label_list[process_id] if all_label_list is not None else None

    max_images = int(getattr(args, "max_images", -1))
    if max_images > 0:
        databin = databin[:max_images]
        if label_bin is not None:
            label_bin = label_bin[:max_images]

    skip_existing = bool(getattr(args, "skip_existing", False))

    ds_cfg = DATASET_CONFIGS[args.dataset]
    model, bg_text_features, fg_text_features, cam = build_worker_components(args, device_id)
    bg_text_features = bg_text_features.to(device_id)
    fg_text_features = fg_text_features.to(device_id)

    saved = 0
    skipped = 0
    errors = 0

    for im_idx, entry in enumerate(tqdm(databin)):
        try:
            stem = osp.splitext(osp.basename(entry))[0]
            out_name = stem + ".npy"
            out_path = osp.join(args.cam_out_dir, out_name)
            if skip_existing and osp.isfile(out_path):
                skipped += 1
                continue

            # Parse labels based on dataset type
            if args.dataset == "voc12":
                img_path, ori_h, ori_w, label_list, label_id_list = (
                    _parse_labels_voc12(entry, args, ds_cfg["fg_classnames"])
                )
            elif args.dataset == "coco14":
                label_ids = label_bin[im_idx] if label_bin else []
                img_path, ori_h, ori_w, label_list, label_id_list = (
                    _parse_labels_coco14(entry, label_ids, args, ds_cfg["fg_classnames"])
                )
            elif args.dataset in ("gta5", "cityscapes"):
                img_path = _resolve_path(args.img_root, entry)
                label_path = _resolve_path(args.label_root, entry)
                if not osp.isfile(img_path):
                    errors += 1
                    continue
                ori_image = Image.open(img_path)
                ori_h, ori_w = np.asarray(ori_image).shape[:2]
                label_id_list = _extract_image_labels_from_mask(label_path)
                if len(label_id_list) == 0:
                    errors += 1
                    continue
                label_list = [CITYSCAPES_PROMPT_NAMES[lid] for lid in label_id_list]
            elif args.dataset == "synthia":
                img_path = _resolve_path(args.img_root, entry)
                label_path = _resolve_path(args.label_root, entry)
                if not osp.isfile(img_path):
                    errors += 1
                    continue
                ori_image = Image.open(img_path)
                ori_h, ori_w = np.asarray(ori_image).shape[:2]
                label_id_list = _extract_synthia_labels_from_mask(label_path)
                if len(label_id_list) == 0:
                    errors += 1
                    continue
                label_list = [CITYSCAPES_PROMPT_NAMES[lid] for lid in label_id_list]
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}")

            if len(label_list) == 0:
                errors += 1
                continue

            keys, refined_cams = generate_cam_for_image(
                model, cam, fg_text_features, bg_text_features,
                img_path, ori_h, ori_w, label_list, label_id_list,
                args, device_id, ds_cfg,
            )

            np.save(out_path, {
                "keys": keys.numpy(),
                "attn_highres": refined_cams.cpu().numpy().astype(np.float16),
            })
            saved += 1

        except Exception as e:
            print(f"[worker {process_id}] Error processing {entry}: {e}")
            errors += 1

    print(f"[worker {process_id}] saved={saved}, skipped={skipped}, errors={errors}")
    return 0
