"""Build multilabel.json for SYNTHIA from label masks.

Usage:
    python tools/build_synthia_multilabel.py \
        --split-file /path/to/data/raw/synthia/splits/train.txt \
        --label-dir /path/to/data/raw/synthia/labels \
        --output-dir /path/to/data/processed/synthia_multilabel
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
from PIL import Image

SYNTHIA_VALID_CITYSCAPES_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18}

SYNTHIA_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "sky", "person", "rider", "car",
    "bus", "motorcycle", "bicycle",
]


def read_split(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def extract_labels(mask_path):
    if not osp.isfile(mask_path):
        return []
    mask = np.array(Image.open(mask_path), dtype=np.int64)
    raw_ids = np.unique(mask)
    labels = [int(v) for v in raw_ids if int(v) in SYNTHIA_VALID_CITYSCAPES_IDS]
    return sorted(set(labels))


def main(args):
    filenames = read_split(args.split_file)

    multilabel = {}
    missing = []
    for entry in filenames:
        filename = osp.basename(entry)
        mask_path = osp.join(args.label_dir, filename)
        labels = extract_labels(mask_path)
        multilabel[filename] = labels
        if not labels:
            missing.append(filename)

    os.makedirs(args.output_dir, exist_ok=True)

    multilabel_path = osp.join(args.output_dir, args.output_file)
    class_names_path = osp.join(args.output_dir, "class_names.json")

    with open(multilabel_path, "w") as f:
        json.dump(multilabel, f, indent=2)

    with open(class_names_path, "w") as f:
        json.dump(SYNTHIA_CLASS_NAMES, f, indent=2)

    print(f"Saved: {multilabel_path}")
    print(f"Saved: {class_names_path}")
    print(f"Total files: {len(filenames)}")
    print(f"Empty-label files: {len(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file", type=str, required=True,
        help="Path to split txt file (e.g., train.txt)",
    )
    parser.add_argument(
        "--label-dir", type=str, required=True,
        help="Path to labels directory (Cityscapes 19-class train IDs)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for multilabel.json and class_names.json",
    )
    parser.add_argument(
        "--output-file", type=str, default="multilabel.json",
        help="Output multilabel json filename",
    )

    main(parser.parse_args())
