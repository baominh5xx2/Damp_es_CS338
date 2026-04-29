"""CLI entry point for CLIP-ES CAM generation.

Usage examples:
    python generate_cams.py --dataset voc12 --img_root /data/VOC2012/JPEGImages \\
        --split_file datasets/splits/voc12/trainval.txt --cam_out_dir output/voc12/cams

    python generate_cams.py --dataset gta5 --img_root /data/gta5/images \\
        --label_root /data/gta5/labels --split_file /data/gta5/splits/train_full.txt \\
        --cam_out_dir output/gta5/cams --max_long_side 1024

    python generate_cams.py --dataset voc12 --img_root /data/VOC2012/JPEGImages \\
        --split_file datasets/splits/voc12/trainval.txt --cam_out_dir output/voc12/cams_damp \\
        --damp_prompt_ckpt output/damp/prompt_learner.pth
"""

import argparse
import os

import numpy as np
import torch
from torch import multiprocessing

from cam.generate import perform, split_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP-ES CAMs (unified for VOC12, COCO14, GTA5, Cityscapes)"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["voc12", "coco14", "gta5", "cityscapes", "synthia"])
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--label_root", type=str, default="",
                        help="Label root (required for gta5/cityscapes)")
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--cam_out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--box_threshold", type=float, default=-1,
                        help="Override per-dataset default (-1 = use dataset default)")
    parser.add_argument("--image_scale", type=float, default=1.0)
    parser.add_argument("--max_long_side", type=int, default=-1,
                        help="Override per-dataset default (-1 = use dataset default)")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_refine", action="store_true",
                        help="Skip attention-based refinement (save raw Grad-CAM)")

    # DAMP integration
    parser.add_argument("--damp_prompt_ckpt", type=str, default="",
                        help="Path to DAMP prompt_learner.pth checkpoint")
    parser.add_argument("--damp_n_ctx", type=int, default=-1,
                        help="Number of context tokens (-1 = auto-detect from ckpt)")
    parser.add_argument("--damp_mix_alpha", type=float, default=1.0,
                        help="Blend weight: 1.0=DAMP only, 0.0=zero-shot only")
    parser.add_argument("--damp_use_plain_names", dest="damp_use_plain_names",
                        action="store_true",
                        help="Use plain class names with DAMP prompts (recommended)")
    parser.add_argument("--damp_use_synonym_names", dest="damp_use_plain_names",
                        action="store_false",
                        help="Use synonym-expanded class names with DAMP prompts")
    parser.set_defaults(damp_use_plain_names=True)

    args = parser.parse_args()

    # Override box_threshold / max_long_side only if explicitly set
    from cam.generate import DATASET_CONFIGS
    ds_cfg = DATASET_CONFIGS[args.dataset]
    if args.box_threshold < 0:
        args.box_threshold = ds_cfg["box_threshold"]
    if args.max_long_side < 0:
        args.max_long_side = ds_cfg["max_long_side"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Settings: image_scale={args.image_scale}, max_long_side={args.max_long_side}, "
          f"box_threshold={args.box_threshold}, no_refine={args.no_refine}")

    if not os.path.isfile(args.split_file):
        raise FileNotFoundError(f"Split file not found: {args.split_file}")

    os.makedirs(args.cam_out_dir, exist_ok=True)

    # Load split + labels depending on dataset
    all_label_list = None

    if args.dataset == "voc12":
        train_list = np.loadtxt(args.split_file, dtype=str).tolist()
        if isinstance(train_list, str):
            train_list = [train_list]
        train_list = [x + '.jpg' for x in train_list]

    elif args.dataset == "coco14":
        file_list = tuple(open(args.split_file, "r"))
        file_list = [id_.rstrip().split(" ") for id_ in file_list]
        train_list = [x[0] + '.jpg' for x in file_list]
        all_label_list = [x[1:] for x in file_list]

    elif args.dataset in ("gta5", "cityscapes", "synthia"):
        with open(args.split_file, "r") as f:
            train_list = [line.strip() for line in f if line.strip()]

    print(f"Total images: {len(train_list)}")

    dataset_list = split_dataset(train_list, n_splits=args.num_workers)

    if all_label_list is not None:
        label_splits = split_dataset(all_label_list, n_splits=args.num_workers)
    else:
        label_splits = None

    if len(dataset_list) == 1:
        perform(0, dataset_list, args, all_label_list=label_splits)
    else:
        print(f"Launching {len(dataset_list)} workers")
        multiprocessing.spawn(
            perform,
            nprocs=len(dataset_list),
            args=(dataset_list, args, label_splits),
        )

    print("Done.")


if __name__ == "__main__":
    main()
