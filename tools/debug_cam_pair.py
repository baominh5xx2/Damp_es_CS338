"""Inspect generated CAM files against ground-truth masks.

This is intentionally small and print-oriented: use it when mIoU is exactly
zero and we need to know whether CAMs are empty, predictions are all ignored,
or labels are in an unexpected ID space.
"""

import argparse
import os
import os.path as osp
import sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cam.evaluate import (  # noqa: E402
    entry_stem,
    load_pred_from_npy,
    map_mask_to_synthia16,
    map_mask_to_trainid,
    read_split_file,
    resolve_label_path,
)


def _counts(arr, limit=24):
    values, counts = np.unique(arr, return_counts=True)
    pairs = list(zip(values.tolist(), counts.tolist()))
    return pairs[:limit]


def _mapped_gt(dataset, gt):
    if dataset in ("gta5", "cityscapes"):
        return map_mask_to_trainid(gt)
    if dataset == "synthia":
        return map_mask_to_synthia16(gt)
    return gt


def inspect_one(args, entry):
    stem = entry_stem(entry)
    cam_path = osp.join(args.cam_dir, stem + ".npy")
    gt_path = resolve_label_path(args.gt_root, entry)

    cam = np.load(cam_path, allow_pickle=True).item()
    cams = cam[args.cam_type]
    keys = cam["keys"].astype(np.int64)
    gt_raw = np.asarray(Image.open(gt_path), dtype=np.uint8)
    gt = _mapped_gt(args.dataset, gt_raw)
    pred_bg = load_pred_from_npy(
        cam_path,
        args.cam_type,
        args.threshold,
        use_bg_channel=True,
        n_class=args.n_class,
        dataset=args.dataset,
    )
    pred_no_bg = load_pred_from_npy(
        cam_path,
        args.cam_type,
        args.threshold,
        use_bg_channel=False,
        n_class=args.n_class,
        dataset=args.dataset,
    )
    pred_bg_mapped = _mapped_gt(args.dataset, pred_bg)
    pred_no_bg_mapped = _mapped_gt(args.dataset, pred_no_bg)

    valid = (gt >= 0) & (gt < args.n_class)
    acc_bg = float((pred_bg_mapped[valid] == gt[valid]).mean()) if valid.any() else 0.0
    acc_no_bg = (
        float((pred_no_bg_mapped[valid] == gt[valid]).mean()) if valid.any() else 0.0
    )

    print("=" * 80)
    print(f"entry: {entry}")
    print(f"cam: {cam_path}")
    print(f"gt : {gt_path}")
    print(f"keys: {keys.tolist()}")
    print(
        "cams: "
        f"shape={tuple(cams.shape)} dtype={cams.dtype} "
        f"finite={np.isfinite(cams).all()} "
        f"min={float(np.nanmin(cams)):.6f} "
        f"max={float(np.nanmax(cams)):.6f} "
        f"mean={float(np.nanmean(cams)):.6f}"
    )
    print(f"channel max: {[float(x) for x in np.nanmax(cams, axis=(1, 2)).tolist()]}")
    print(f"gt raw unique: {_counts(gt_raw)}")
    print(f"gt mapped unique: {_counts(gt)}")
    print(f"pred bg unique: {_counts(pred_bg_mapped)}  pixel_acc={acc_bg:.6f}")
    print(f"pred no-bg unique: {_counts(pred_no_bg_mapped)}  pixel_acc={acc_no_bg:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["voc12", "coco14", "gta5", "cityscapes", "synthia"])
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--cam-type", default="attn_highres")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--n-class", type=int, default=19)
    args = parser.parse_args()

    entries = read_split_file(args.split_file)
    kept = []
    for entry in entries:
        cam_path = osp.join(args.cam_dir, entry_stem(entry) + ".npy")
        gt_path = resolve_label_path(args.gt_root, entry)
        if osp.isfile(cam_path) and osp.isfile(gt_path):
            kept.append(entry)
        if len(kept) >= args.max_images:
            break

    if not kept:
        raise RuntimeError("No CAM/GT pairs found for the provided paths")

    print(f"Inspecting {len(kept)} CAM/GT pairs")
    for entry in kept:
        inspect_one(args, entry)


if __name__ == "__main__":
    main()
