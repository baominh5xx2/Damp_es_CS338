"""Build multilabel.json for Cityscapes from label masks.

Cityscapes labels are stored as RAW label IDs (7, 8, 11, ..., 33).
This script remaps them to Cityscapes-19 train IDs (0..18), then
filters to the SYNTHIA-16 valid set for use in SYNTHIA->Cityscapes UDA.

Usage:
    python tools/build_cityscapes_multilabel.py \
        --split-file /path/to/data/raw/cityscapes/splits/train.txt \
        --label-dir /path/to/data/raw/cityscapes/labels \
        --output-dir /path/to/data/processed/cityscapes_multilabel \
        --output-file train_multilabel.json

    python tools/build_cityscapes_multilabel.py \
        --split-file /path/to/data/raw/cityscapes/splits/val.txt \
        --label-dir /path/to/data/raw/cityscapes/labels \
        --output-dir /path/to/data/processed/cityscapes_multilabel \
        --output-file val_multilabel.json
"""

import argparse
import json
import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

RAW_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}

SYNTHIA_VALID_CITYSCAPES_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18}


def read_split(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def extract_labels(mask_path):
    if not osp.isfile(mask_path):
        return []
    mask = np.array(Image.open(mask_path), dtype=np.int64)
    raw_ids = np.unique(mask)

    train_ids = set()
    for raw_id in raw_ids:
        raw_id = int(raw_id)
        if raw_id in RAW_TO_TRAINID:
            train_ids.add(RAW_TO_TRAINID[raw_id])

    labels = sorted(tid for tid in train_ids if tid in SYNTHIA_VALID_CITYSCAPES_IDS)
    return labels


def _extract_one(args_tuple):
    filename, label_dir = args_tuple
    mask_path = osp.join(label_dir, filename)
    labels = extract_labels(mask_path)
    return filename, labels


def main(args):
    filenames = read_split(args.split_file)
    num_workers = args.num_workers

    multilabel = {}
    missing = []

    work_items = [
        (osp.basename(entry), args.label_dir) for entry in filenames
    ]

    if num_workers <= 1 or len(work_items) <= 10:
        for filename, label_dir in work_items:
            _, labels = _extract_one((filename, label_dir))
            multilabel[filename] = labels
            if not labels:
                missing.append(filename)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_extract_one, item): item[0]
                for item in work_items
            }
            done = 0
            for fut in as_completed(futures):
                filename, labels = fut.result()
                multilabel[filename] = labels
                if not labels:
                    missing.append(filename)
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{len(work_items)} labels extracted",
                          flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    multilabel_path = osp.join(args.output_dir, args.output_file)

    with open(multilabel_path, "w") as f:
        json.dump(multilabel, f, indent=2)

    print(f"Saved: {multilabel_path}")
    print(f"Total files: {len(filenames)}")
    print(f"Empty-label files: {len(missing)}")
    if num_workers > 1:
        print(f"Workers used: {num_workers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file", type=str, required=True,
        help="Path to split txt file (e.g., train.txt or val.txt)",
    )
    parser.add_argument(
        "--label-dir", type=str, required=True,
        help="Path to Cityscapes labels directory (raw label IDs)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for multilabel JSON",
    )
    parser.add_argument(
        "--output-file", type=str, default="train_multilabel.json",
        help="Output filename (use train_multilabel.json or val_multilabel.json)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Parallel workers for label extraction (default: 8)",
    )

    main(parser.parse_args())
