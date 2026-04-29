import json
import os.path as osp

import numpy as np
from PIL import Image

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class GTA5(DatasetBase):
    """GTA5/Cityscapes-style segmentation dataset for DAMP classification stage.

    This dataset reads split files and converts image-level multi-label
    annotations into single-label Datum items (one image can produce multiple
    Datum records).
    """

    dataset_dir = "data"
    domains = ["gta5_train", "cityscapes_train", "cityscapes_val"]

    CLASS_NAMES = [
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
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.raw_gta5_dir = self._first_existing_dir(
            [
                osp.join(self.dataset_dir, "raw", "gta5_kaggle_full"),
                osp.join(root, "gta5-kaggle-full"),
            ],
            default=osp.join(self.dataset_dir, "raw", "gta5_kaggle_full"),
        )
        self.raw_city_dir = self._first_existing_dir(
            [
                osp.join(self.dataset_dir, "raw", "cityscapes"),
                osp.join(root, "cityscapes"),
            ],
            default=osp.join(self.dataset_dir, "raw", "cityscapes"),
        )
        self.processed_dir = osp.join(self.dataset_dir, "processed")

        self.gta5_multilabel_file = osp.join(
            self.processed_dir, "gta5_multilabel", "multilabel.json"
        )

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        # For UDA, evaluate on cityscapes val when available.
        test_domains = ["cityscapes_val"] if self._has_cityscapes_val() else cfg.DATASET.TARGET_DOMAINS
        test = self._read_data(test_domains)

        train_x, train_u, test = self._remap_labels_to_contiguous(
            train_x, train_u, test
        )

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    @staticmethod
    def _first_existing_dir(candidates, default):
        for path in candidates:
            if osp.isdir(path):
                return path
        return default

    def _domain_config(self, domain_name):
        if domain_name == "gta5_train":
            return {
                "image_dir": osp.join(self.raw_gta5_dir, "images"),
                "label_dir": osp.join(self.raw_gta5_dir, "labels"),
                "split_file": osp.join(self.raw_gta5_dir, "splits", "train_full.txt"),
                "multilabel_file": self.gta5_multilabel_file,
            }

        if domain_name == "cityscapes_train":
            return {
                "image_dir": osp.join(self.raw_city_dir, "images"),
                "label_dir": osp.join(self.raw_city_dir, "labels"),
                "split_file": osp.join(self.raw_city_dir, "splits", "train.txt"),
                "multilabel_file": osp.join(
                    self.processed_dir,
                    "cityscapes_multilabel",
                    "train_multilabel.json",
                ),
            }

        if domain_name == "cityscapes_val":
            return {
                "image_dir": osp.join(self.raw_city_dir, "images"),
                "label_dir": osp.join(self.raw_city_dir, "labels"),
                "split_file": osp.join(self.raw_city_dir, "splits", "val.txt"),
                "multilabel_file": osp.join(
                    self.processed_dir,
                    "cityscapes_multilabel",
                    "val_multilabel.json",
                ),
            }

        raise ValueError(f"Unsupported domain: {domain_name}")

    def _has_cityscapes_val(self):
        return osp.isfile(osp.join(self.raw_city_dir, "splits", "val.txt"))

    @staticmethod
    def _read_split_file(split_file):
        if not osp.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        return lines

    @staticmethod
    def _to_filename(split_entry):
        # Allow filename-only or relative path in split file.
        return osp.basename(split_entry)

    def _load_multilabel_map(self, multilabel_file):
        if not osp.isfile(multilabel_file):
            return {}

        with open(multilabel_file, "r") as f:
            data = json.load(f)

        output = {}
        for key, value in data.items():
            filename = self._to_filename(key)
            if isinstance(value, list):
                if value and all(isinstance(v, (int, np.integer)) for v in value):
                    labels = sorted(
                        int(v)
                        for v in value
                        if 0 <= int(v) < self.NUM_CLASSES
                    )
                    output[filename] = labels
                    continue

                # Multi-hot vector case.
                labels = [
                    idx for idx, flag in enumerate(value)
                    if int(flag) > 0 and idx < self.NUM_CLASSES
                ]
                output[filename] = labels

        return output

    def _extract_labels_from_mask(self, mask_path):
        if not osp.isfile(mask_path):
            return []

        mask = np.array(Image.open(mask_path), dtype=np.int64)
        labels = np.unique(mask)
        
        id_to_trainid = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
            25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }
        
        if any(int(v) > 18 and int(v) != 255 for v in labels):
            mapped_labels = [
                id_to_trainid[int(v)] for v in labels 
                if int(v) in id_to_trainid
            ]
        else:
            mapped_labels = [int(v) for v in labels if 0 <= int(v) < self.NUM_CLASSES]
            
        return sorted(list(set(mapped_labels)))


    def _read_data(self, input_domains):
        items = []

        for domain_index, domain_name in enumerate(input_domains):
            cfg = self._domain_config(domain_name)
            split_entries = self._read_split_file(cfg["split_file"])
            multilabel_map = self._load_multilabel_map(cfg["multilabel_file"])

            for split_entry in split_entries:
                filename = self._to_filename(split_entry)
                impath = osp.join(cfg["image_dir"], filename)
                if not osp.isfile(impath):
                    continue

                labels = multilabel_map.get(filename, None)
                if labels is None:
                    labels = self._extract_labels_from_mask(
                        osp.join(cfg["label_dir"], filename)
                    )

                # Keep pipeline alive even when target labels are unavailable.
                if not labels:
                    labels = [0]

                for label in labels:
                    item = Datum(
                        impath=impath,
                        label=int(label),
                        domain=domain_index,
                        classname=self.CLASS_NAMES[int(label)],
                    )
                    items.append(item)

        return items

    def _remap_labels_to_contiguous(self, train_x, train_u, test):
        all_items = train_x + train_u + test
        if not all_items:
            return train_x, train_u, test

        raw_labels = sorted({int(item.label) for item in all_items})
        label_map = {raw_label: idx for idx, raw_label in enumerate(raw_labels)}

        def _remap(items):
            remapped = []
            for item in items:
                raw_label = int(item.label)
                new_label = label_map[raw_label]
                remapped.append(
                    Datum(
                        impath=item.impath,
                        label=new_label,
                        domain=item.domain,
                        classname=self.CLASS_NAMES[raw_label],
                    )
                )
            return remapped

        return _remap(train_x), _remap(train_u), _remap(test)
