import argparse
import json
import os
import os.path as osp

import numpy as np
from PIL import Image


DEFAULT_CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def read_split(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def to_filename(entry):
    return osp.basename(entry)


def extract_labels(mask_path, num_classes):
    if not osp.isfile(mask_path):
        return []

    mask = np.array(Image.open(mask_path), dtype=np.int64)
    labels = np.unique(mask)
    
    # Cityscapes/GTA5 raw ID to train ID mapping (19 classes)
    id_to_trainid = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
    }
    
    # Detect if dataset is using raw IDs (>18 are present) or TrainIDs (all valid max 18)
    if any(int(v) > 18 and int(v) != 255 for v in labels):
        # Raw IDs
        mapped_labels = [
            id_to_trainid[int(v)] for v in labels 
            if int(v) in id_to_trainid
        ]
    else:
        # Pre-mapped train IDs [0..18]
        mapped_labels = [int(v) for v in labels if 0 <= int(v) < num_classes]
        
    return sorted(list(set(mapped_labels)))


def main(args):
    filenames = read_split(args.split_file)
    num_classes = args.num_classes

    multilabel = {}
    missing = []
    for entry in filenames:
        filename = to_filename(entry)
        mask_path = osp.join(args.label_dir, filename)
        labels = extract_labels(mask_path, num_classes)
        multilabel[filename] = labels
        if not labels:
            missing.append(filename)

    os.makedirs(args.output_dir, exist_ok=True)

    multilabel_path = osp.join(args.output_dir, args.output_file)
    class_names_path = osp.join(args.output_dir, "class_names.json")

    with open(multilabel_path, "w") as f:
        json.dump(multilabel, f, indent=2)

    class_names = DEFAULT_CLASS_NAMES[:num_classes]
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved: {multilabel_path}")
    print(f"Saved: {class_names_path}")
    print(f"Total files: {len(filenames)}")
    print(f"Empty-label files: {len(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Path to split txt file (e.g., train_full.txt)",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        required=True,
        help="Path to labels directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for multilabel.json and class_names.json",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=19,
        help="Number of valid classes; pixel IDs outside range are ignored",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="multilabel.json",
        help="Output multilabel json filename",
    )

    main(parser.parse_args())
