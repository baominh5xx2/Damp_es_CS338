"""CLI entry point for CAM evaluation (with optional CRF post-processing).

Usage examples:
    python eval_cam.py --dataset voc12 --cam_out_dir output/voc12/cams \
        --gt_root /data/VOC2012/SegmentationClass \
        --split_file datasets/splits/voc12/trainval.txt

    python eval_cam.py --dataset gta5 --cam_out_dir output/gta5/cams \
        --gt_root /data/gta5/labels \
        --split_file /data/gta5/splits/val.txt

    python eval_cam.py --dataset voc12 --cam_out_dir output/voc12/cams \
        --gt_root /data/VOC2012/SegmentationClass \
        --split_file datasets/splits/voc12/trainval.txt \
        --crf --image_root /data/VOC2012/JPEGImages \
        --mask_output_dir output/voc12/pseudo_masks
"""

import argparse
import os

import numpy as np
from PIL import Image

from cam.evaluate import (
    compute_scores,
    entry_stem,
    load_pred_from_npy,
    map_mask_to_trainid,
    map_mask_to_synthia16,
    read_split_file,
    resolve_label_path,
    run_eval_cam,
    run_eval_with_crf,
)
from cam.clip_text import (
    class_names,
    class_names_coco,
    CITYSCAPES_CLASS_NAMES,
)


DATASET_N_CLASS = {
    "voc12": 21,
    "coco14": 81,
    "gta5": 19,
    "cityscapes": 19,
    "synthia": 19,
}

DATASET_CLASS_NAMES = {
    "voc12": ["background"] + class_names,
    "coco14": ["background"] + class_names_coco,
    "gta5": list(CITYSCAPES_CLASS_NAMES),
    "cityscapes": list(CITYSCAPES_CLASS_NAMES),
    "synthia": list(CITYSCAPES_CLASS_NAMES),
}


def load_split(args):
    if args.dataset == "voc12":
        entries = np.loadtxt(args.split_file, dtype=str).tolist()
        if isinstance(entries, str):
            entries = [entries]
        return [x + ".jpg" for x in entries]

    if args.dataset == "coco14":
        file_list = tuple(open(args.split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        return [x[0] + ".jpg" for x in file_list]

    return read_split_file(args.split_file)


def print_results(result, n_class, class_names_list):
    print(f"\n{'='*60}")
    print(f"  Pixel Accuracy  : {result['Pixel Accuracy']:.4f}")
    print(f"  Mean Accuracy   : {result['Mean Accuracy']:.4f}")
    print(f"  Mean IoU        : {result['Mean IoU']:.4f}")
    print(f"  FW IoU          : {result['Frequency Weighted IoU']:.4f}")
    print(f"{'='*60}")
    print(f"\n  Per-class IoU:")
    iu = result["IoU Array"]
    for i in range(min(n_class, len(iu))):
        name = class_names_list[i] if i < len(class_names_list) else f"class_{i}"
        print(f"    {name:30s} : {iu[i]:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAM predictions")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["voc12", "coco14", "gta5", "cityscapes", "synthia"])
    parser.add_argument("--cam_out_dir", type=str, required=True,
                        help="Directory containing .npy or .png CAM predictions")
    parser.add_argument("--gt_root", type=str, required=True,
                        help="Root directory of ground truth segmentation masks")
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--cam_type", type=str, default="attn_highres",
                        choices=["attn_highres", "highres", "rw", "png"],
                        help="Key in .npy to read (or 'png' for .png masks)")
    parser.add_argument("--max_images", type=int, default=-1)

    # Background threshold
    parser.add_argument("--cam_eval_thres", type=float, default=-1,
                        help="Single threshold (0-1). -1 = grid search.")
    parser.add_argument("--thres_start", type=float, default=0.05)
    parser.add_argument("--thres_end", type=float, default=0.80)
    parser.add_argument("--thres_step", type=float, default=0.05)
    parser.add_argument("--no_bg_channel", action="store_true",
                        help="Skip background channel (argmax over fg only)")

    # CRF options
    parser.add_argument("--crf", action="store_true",
                        help="Apply DenseCRF post-processing")
    parser.add_argument("--image_root", type=str, default="",
                        help="Image root (required for CRF)")
    parser.add_argument("--mask_output_dir", type=str, default="",
                        help="Save CRF pseudo-masks to this directory")
    parser.add_argument("--crf_confidence", type=float, default=0.95,
                        help="Confidence threshold for CRF pseudo-mask")
    parser.add_argument("--crf_n_jobs", type=int, default=-1)

    args = parser.parse_args()
    n_class = DATASET_N_CLASS[args.dataset]
    names = DATASET_CLASS_NAMES[args.dataset]

    eval_list = load_split(args)
    if args.max_images > 0:
        eval_list = eval_list[:args.max_images]

    # Auto-filter: only keep entries that have CAM files
    ext = ".png" if args.cam_type == "png" else ".npy"
    original_count = len(eval_list)
    eval_list = [e for e in eval_list
                 if os.path.isfile(os.path.join(args.cam_out_dir,
                                                entry_stem(e) + ext))]
    if len(eval_list) < original_count:
        print(f"Filtered: {original_count} entries → {len(eval_list)} with CAM files")

    if not eval_list:
        print("No CAM files found. Nothing to evaluate.")
        return

    print(f"Dataset: {args.dataset} ({n_class} classes, {len(eval_list)} images)")

    # ----- CRF evaluation -----
    if args.crf:
        if not args.image_root:
            raise ValueError("--image_root is required for CRF evaluation")

        mask_dir = args.mask_output_dir if args.mask_output_dir else None
        result = run_eval_with_crf(
            eval_list=eval_list,
            cam_dir=args.cam_out_dir,
            gt_root=args.gt_root,
            image_root=args.image_root,
            cam_type=args.cam_type,
            n_class=n_class,
            mask_output_dir=mask_dir,
            confidence_threshold=args.crf_confidence,
            n_jobs=args.crf_n_jobs,
            dataset=args.dataset,
        )
        print("\n[CRF Evaluation]")
        print_results(result, n_class, names)
        return

    # ----- Standard evaluation (with threshold grid search) -----
    use_bg = not args.no_bg_channel

    if args.cam_type == "png":
        print("Evaluating .png masks (no threshold search)")
        result = run_eval_cam(
            eval_list=eval_list,
            cam_dir=args.cam_out_dir,
            gt_root=args.gt_root,
            cam_type="png",
            cam_eval_thres=0.0,
            n_class=n_class,
            dataset=args.dataset,
            use_bg_channel=False,
        )
        print_results(result, n_class, names)
        return

    if args.cam_eval_thres >= 0:
        thresholds = [args.cam_eval_thres]
    else:
        thresholds = np.arange(
            args.thres_start, args.thres_end + 1e-6, args.thres_step
        ).tolist()

    best_miou = -1.0
    best_thres = thresholds[0]
    best_result = None

    if len(thresholds) == 1:
        # Single threshold — use normal path
        result = run_eval_cam(
            eval_list=eval_list,
            cam_dir=args.cam_out_dir,
            gt_root=args.gt_root,
            cam_type=args.cam_type,
            cam_eval_thres=thresholds[0],
            n_class=n_class,
            dataset=args.dataset,
            use_bg_channel=use_bg,
        )
        best_result = result
        best_thres = thresholds[0]
    else:
        # Grid search — load CAMs + GTs once, loop thresholds in memory
        print(f"Loading {len(eval_list)} CAMs + GTs (one-time)...")
        voc_style = args.dataset in ("voc12", "coco14")
        all_cams = []
        all_keys = []
        all_gts = []

        for entry in eval_list:
            stem = entry_stem(entry)
            cam_npy = os.path.join(args.cam_out_dir, stem + ".npy")
            gt_file = resolve_label_path(args.gt_root, entry)

            if not os.path.isfile(cam_npy) or not os.path.isfile(gt_file):
                continue

            cam_dict = np.load(cam_npy, allow_pickle=True).item()
            all_cams.append(cam_dict[args.cam_type])
            all_keys.append(cam_dict["keys"].astype(np.int64))
            gt = np.asarray(Image.open(gt_file), dtype=np.uint8)

            if args.dataset in ("gta5", "cityscapes"):
                gt = map_mask_to_trainid(gt)
            elif args.dataset == "synthia":
                gt = map_mask_to_synthia16(gt)
            all_gts.append(gt)

        print(f"Loaded {len(all_cams)} pairs. Searching {len(thresholds)} thresholds...")

        for thres in thresholds:
            thres = round(thres, 4)
            preds = []
            for i in range(len(all_cams)):
                cams = all_cams[i]
                keys = all_keys[i]

                if use_bg:
                    bg_score = np.full((1, cams.shape[1], cams.shape[2]),
                                       thres, dtype=cams.dtype)
                    cams_with_bg = np.concatenate((bg_score, cams), axis=0)
                    pred_idx = np.argmax(cams_with_bg, axis=0)

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

                preds.append(pred)

            result = compute_scores(all_gts, preds, n_class=n_class)
            miou = result["Mean IoU"]
            print(f"  thres={thres:.2f}  mIoU={miou:.4f}")

            if miou > best_miou:
                best_miou = miou
                best_thres = thres
                best_result = result

    print(f"\nBest threshold: {best_thres:.4f}")
    print_results(best_result, n_class, names)


if __name__ == "__main__":
    main()
