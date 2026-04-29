"""COCO 2014 dataset (Dassl wrapper) using CLIP-ES split format.

Split files (``Damp_es/datasets/splits/coco14/{train,val}.txt``) contain one
line per image:

    COCO_train2014_000000000009 45 49 50

Each (image, class) pair is emitted as a separate ``Datum``; the DAMP trainer
re-builds multi-hot vectors per batch using the multilabel lookup helper.

Dataset layout expected on disk::

    $DATA/coco14/
        train2014/  COCO_train2014_*.jpg
        val2014/    COCO_val2014_*.jpg
"""

import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


_SPLITS_DIR = osp.abspath(osp.join(osp.dirname(__file__), "splits", "coco14"))


@DATASET_REGISTRY.register()
class COCO14(DatasetBase):
    dataset_dir = "coco14"
    domains = ["train", "val"]

    # Vanilla MS-COCO 80 class names (order matches CLIP-ES `class_names_coco`).
    CLASS_NAMES = [
        "person", "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    # CLIP-ES enriched prompt names (`new_class_names_coco`).
    PROMPT_CLASS_NAMES = [
        "person with clothes,people,human", "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird avian",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack,bag",
        "umbrella,parasol", "handbag,purse", "necktie", "suitcase", "frisbee",
        "skis", "sknowboard", "sports ball", "kite", "baseball bat",
        "glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "dessertspoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair seat", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor screen", "laptop", "mouse",
        "remote control", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hairdrier,blowdrier", "toothbrush",
    ]

    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        val = self._read_data(["val"])
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    # ------------------------------------------------------------------ paths
    def _image_dir(self, split_name):
        # COCO 2014: train2014/ and val2014/
        return osp.join(self.dataset_dir, f"{split_name}2014")

    def _split_file(self, split_name):
        path = osp.join(_SPLITS_DIR, f"{split_name}.txt")
        if not osp.isfile(path):
            raise FileNotFoundError(f"COCO14 split not found: {path}")
        return path

    # ----------------------------------------------------------------- labels
    @staticmethod
    def _parse_split_line(line):
        toks = line.strip().split()
        if not toks:
            return None, []
        image_id = toks[0]
        labels = sorted({int(t) for t in toks[1:]})
        return image_id, labels

    def _read_data(self, input_domains):
        items = []
        for domain, split_name in enumerate(input_domains):
            image_dir = self._image_dir(split_name)
            with open(self._split_file(split_name), "r") as f:
                for raw in f:
                    image_id, labels = self._parse_split_line(raw)
                    if image_id is None:
                        continue
                    impath = osp.join(image_dir, f"{image_id}.jpg")
                    if not osp.isfile(impath):
                        continue
                    if not labels:
                        # Keep iterator alive so multilabel lookup can hit it.
                        labels = [0]
                    for label in labels:
                        if not (0 <= label < self.NUM_CLASSES):
                            continue
                        items.append(Datum(
                            impath=impath,
                            label=int(label),
                            domain=domain,
                            classname=self.PROMPT_CLASS_NAMES[label],
                        ))
        return items
