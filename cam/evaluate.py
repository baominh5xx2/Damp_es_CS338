"""Unified CAM evaluation for VOC12, COCO14, GTA5, and Cityscapes."""

import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from cam.clip_text import CITYSCAPES_CLASS_NAMES, ID_TO_TRAINID, SYNTHIA_16_TO_CITYSCAPES_19


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_split_file(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def entry_stem(entry):
    return osp.splitext(osp.basename(entry))[0]


def resolve_label_path(label_root, entry):
    base = osp.basename(entry)
    stem, ext = osp.splitext(base)
    candidates = [entry, base, stem + ".png", stem + ".jpg", stem + ".jpeg"]
    for name in candidates:
        path = osp.join(label_root, name)
        if osp.isfile(path):
            return path
    return osp.join(label_root, stem + ".png")


def map_mask_to_trainid(mask):
    mask = mask.astype(np.int64)
    if np.any((mask > 18) & (mask != 255)):
        out = np.full(mask.shape, 255, dtype=np.uint8)
        for raw_id, train_id in ID_TO_TRAINID.items():
            out[mask == raw_id] = train_id
        return out
    out = mask.copy()
    out[(out < 0) | (out > 18)] = 255
    return out.astype(np.uint8)


_SYNTHIA_VALID_CIT19 = set(SYNTHIA_16_TO_CITYSCAPES_19.values())


def map_mask_to_synthia16(mask):
    """Keep only the 16 Cityscapes train IDs used by SYNTHIA; ignore the rest."""
    mask = mask.astype(np.int64)
    if np.any((mask > 18) & (mask != 255)):
        mask = map_mask_to_trainid(mask)
    out = mask.copy()
    for v in np.unique(out):
        v = int(v)
        if v != 255 and v not in _SYNTHIA_VALID_CIT19:
            out[out == v] = 255
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    lt = label_true[mask].astype(int)
    lp = label_pred[mask].astype(int)
    lp[(lp < 0) | (lp >= n_class)] = n_class

    hist = np.bincount(
        (n_class + 1) * lt + lp,
        minlength=n_class * (n_class + 1),
    ).reshape(n_class, n_class + 1)
    return hist


def compute_scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class + 1))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    tp = np.diag(hist[:, :n_class])
    gt_count = hist.sum(axis=1)
    pred_count = hist[:, :n_class].sum(axis=0)

    acc = tp.sum() / max(gt_count.sum(), 1.0)
    acc_cls = tp / np.maximum(gt_count, 1.0)
    acc_cls = np.nanmean(acc_cls)
    iu = tp / np.maximum(gt_count + pred_count - tp, 1.0)
    valid = gt_count > 0
    mean_iu = np.nanmean(iu[valid])
    freq = gt_count / max(gt_count.sum(), 1.0)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
        "IoU Array": iu,
    }


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def load_pred_from_npy(cam_path, cam_type, cam_eval_thres, use_bg_channel=True,
                       n_class=21, dataset="voc12"):
    cam_dict = np.load(cam_path, allow_pickle=True).item()
    cams = cam_dict[cam_type]
    keys = cam_dict["keys"].astype(np.int64)

    # VOC12/COCO14 use 1-indexed class labels (bg=0); Cityscapes-family use
    # train IDs directly and bg is mapped to 255.
    voc_style = dataset in ("voc12", "coco14")

    if use_bg_channel:
        if cam_eval_thres < 1:
            bg_score = np.full((1, cams.shape[1], cams.shape[2]),
                               cam_eval_thres, dtype=cams.dtype)
        else:
            bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True),
                                cam_eval_thres)
        cams = np.concatenate((bg_score, cams), axis=0)
        pred_idx = np.argmax(cams, axis=0)

        if voc_style:
            keys_with_bg = np.pad(keys + 1, (1, 0), mode='constant')
            pred = keys_with_bg[pred_idx].astype(np.uint8)
        else:
            pred = np.full(pred_idx.shape, 255, dtype=np.uint8)
            fg_mask = pred_idx > 0
            pred[fg_mask] = keys[pred_idx[fg_mask] - 1].astype(np.uint8)
    else:
        pred_idx = np.argmax(cams, axis=0)
        pred = keys[pred_idx].astype(np.uint8)

    return pred


# ---------------------------------------------------------------------------
# CRF evaluation
# ---------------------------------------------------------------------------

def run_eval_with_crf(eval_list, cam_dir, gt_root, image_root, cam_type,
                      n_class, mask_output_dir=None, confidence_threshold=0.95,
                      n_jobs=-1, dataset="voc12"):
    from cam.crf import DenseCRF
    import joblib
    import multiprocessing

    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()

    postprocessor = DenseCRF(
        iter_max=10, pos_xy_std=1, pos_w=3,
        bi_xy_std=67, bi_rgb_std=3, bi_w=4,
    )

    voc_style = dataset in ("voc12", "coco14")
    mean_bgr = (104.008, 116.669, 122.675)

    def process(entry):
        stem = entry_stem(entry)
        cam_npy = osp.join(cam_dir, stem + ".npy")
        gt_file = resolve_label_path(gt_root, entry)

        if not osp.isfile(cam_npy) or not osp.isfile(gt_file):
            return None

        cam_dict = np.load(cam_npy, allow_pickle=True).item()
        cams = cam_dict[cam_type]
        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        cams = np.concatenate((bg_score, cams), axis=0)

        if voc_style:
            image_path = osp.join(image_root, stem + '.jpg')
        else:
            image_path = resolve_label_path(image_root, entry)
            for ext in ['.jpg', '.png', '.jpeg']:
                p = osp.join(image_root, stem + ext)
                if osp.isfile(p):
                    image_path = p
                    break

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image -= mean_bgr
        image = image.astype(np.uint8)

        prob = postprocessor(image, cams)

        label = np.argmax(prob, axis=0)
        if voc_style:
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            label = keys[label]
        else:
            raw_keys = cam_dict['keys'].astype(np.int64)
            out = np.full(label.shape, 255, dtype=np.uint8)
            fg_mask = label > 0
            out[fg_mask] = raw_keys[label[fg_mask] - 1].astype(np.uint8)
            label = out

        if mask_output_dir is not None:
            confidence = np.max(prob, axis=0)
            save_label = label.copy()
            save_label[confidence < confidence_threshold] = 255
            os.makedirs(mask_output_dir, exist_ok=True)
            cv2.imwrite(osp.join(mask_output_dir, stem + '.png'),
                        save_label.astype(np.uint8))

        gt_label = np.asarray(Image.open(gt_file), dtype=np.uint8)
        return label.astype(np.uint8), gt_label.astype(np.uint8)

    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(entry) for entry in eval_list]
    )

    preds = []
    gts = []
    for r in results:
        if r is not None:
            preds.append(r[0])
            gts.append(r[1])

    if len(preds) == 0:
        raise RuntimeError("No valid prediction/GT pairs found.")

    result = compute_scores(gts, preds, n_class=n_class)
    return result


# ---------------------------------------------------------------------------
# Main evaluation (no CRF)
# ---------------------------------------------------------------------------

def run_eval_cam(eval_list, cam_dir, gt_root, cam_type, cam_eval_thres,
                 n_class, dataset, use_bg_channel=True):
    preds = []
    labels = []
    missing = 0

    for entry in eval_list:
        stem = entry_stem(entry)
        cam_npy = osp.join(cam_dir, stem + ".npy")
        cam_png = osp.join(cam_dir, stem + ".png")
        gt_file = resolve_label_path(gt_root, entry)

        if not osp.isfile(gt_file):
            missing += 1
            continue

        if cam_type == "png":
            if not osp.isfile(cam_png):
                missing += 1
                continue
            cls_labels = np.asarray(Image.open(cam_png), dtype=np.uint8)
        else:
            if not osp.isfile(cam_npy):
                missing += 1
                continue
            cls_labels = load_pred_from_npy(
                cam_npy, cam_type, cam_eval_thres,
                use_bg_channel=use_bg_channel, n_class=n_class,
                dataset=dataset,
            )

        gt = np.asarray(Image.open(gt_file), dtype=np.uint8)

        if dataset in ("gta5", "cityscapes"):
            cls_labels = map_mask_to_trainid(cls_labels)
            gt = map_mask_to_trainid(gt)
        elif dataset == "synthia":
            cls_labels = map_mask_to_synthia16(cls_labels)
            gt = map_mask_to_synthia16(gt)

        preds.append(cls_labels)
        labels.append(gt)

    if len(preds) == 0:
        raise RuntimeError("No valid prediction/GT pairs found.")

    if missing > 0:
        print(f"[WARN] skipped {missing} items due to missing files")

    result = compute_scores(labels, preds, n_class=n_class)
    return result
