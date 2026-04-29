"""PASCAL VOC2012 dataset (Dassl wrapper).

Adapted from DAMP/custom_datasets/voc12.py — split files now live in
``Damp_es/datasets/splits/voc12/``. Each (image, class) pair is emitted as a
separate ``Datum`` so it plugs into Dassl's single-label pipeline; the DAMP
trainer rebuilds multi-hot vectors per batch via the multilabel lookup.
"""

import os.path as osp
import xml.etree.ElementTree as ET

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


_SPLITS_DIR = osp.abspath(osp.join(osp.dirname(__file__), "splits", "voc12"))


@DATASET_REGISTRY.register()
class VOC12(DatasetBase):
    dataset_dir = "VOC2012"
    domains = ["train", "train_aug", "trainval", "val"]

    CLASS_NAMES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    # CLIP-ES style enriched class prompts (used by both DAMP & CAM stages).
    PROMPT_CLASS_NAMES = [
        "aeroplane", "bicycle", "bird avian", "boat", "bottle",
        "bus", "car", "cat", "chair seat", "cow",
        "diningtable", "dog", "horse", "motorbike",
        "person with clothes,people,human",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor screen",
    ]

    CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    CLASS_TO_PROMPT = dict(zip(CLASS_NAMES, PROMPT_CLASS_NAMES))

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "JPEGImages")
        self.annotation_dir = osp.join(self.dataset_dir, "Annotations")
        self.split_dirs = self._find_split_dirs(root)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        val = self._read_data(["val"])
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    # ------------------------------------------------------------------ paths
    def _find_split_dirs(self, root):
        candidates = [
            _SPLITS_DIR,
            osp.join(self.dataset_dir, "voc12"),
            osp.join(self.dataset_dir, "ImageSets", "Segmentation"),
            osp.join(root, "voc12"),
        ]

        split_dirs, seen = [], set()
        for path in candidates:
            path = osp.abspath(path)
            if path in seen:
                continue
            seen.add(path)
            if osp.isdir(path):
                split_dirs.append(path)

        if not split_dirs:
            raise FileNotFoundError(
                "No VOC12 split directory found. Expected one of:\n  "
                + "\n  ".join(candidates)
            )
        return split_dirs

    def _resolve_split_file(self, split_name):
        filename = f"{split_name}.txt"
        for split_dir in self.split_dirs:
            split_file = osp.join(split_dir, filename)
            if osp.isfile(split_file):
                return split_file
        searched = "\n  ".join(osp.join(p, filename) for p in self.split_dirs)
        raise FileNotFoundError(
            f"Split file '{filename}' not found. Searched:\n  {searched}"
        )

    # ----------------------------------------------------------------- labels
    @staticmethod
    def _read_split_file(split_file):
        with open(split_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _parse_voc_labels(self, image_id):
        xml_path = osp.join(self.annotation_dir, f"{image_id}.xml")
        if not osp.isfile(xml_path):
            return []

        labels = []
        for obj in ET.parse(xml_path).getroot().findall("object"):
            name_tag = obj.find("name")
            if name_tag is None:
                continue
            class_name = name_tag.text
            if class_name in self.CLASS_TO_LABEL and class_name not in labels:
                labels.append(class_name)
        return labels

    def _read_data(self, input_domains):
        items = []
        for domain, split_name in enumerate(input_domains):
            split_file = self._resolve_split_file(split_name)
            for image_id in self._read_split_file(split_file):
                impath = osp.join(self.image_dir, f"{image_id}.jpg")
                if not osp.isfile(impath):
                    continue
                for class_name in self._parse_voc_labels(image_id):
                    items.append(Datum(
                        impath=impath,
                        label=self.CLASS_TO_LABEL[class_name],
                        domain=domain,
                        classname=self.CLASS_TO_PROMPT[class_name],
                    ))
        return items
