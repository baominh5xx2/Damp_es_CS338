import json
import os.path as osp

import numpy as np
from PIL import Image

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from cam.clip_text import SYNTHIA_16_TO_CITYSCAPES_19


@DATASET_REGISTRY.register()
class SYNTHIA(DatasetBase):
    """SYNTHIA-RAND-CITYSCAPES 16-class dataset for DAMP (SYNTHIA->Cityscapes UDA).

    16 classes = Cityscapes 19 minus terrain, truck, train.
    Label masks are stored in Cityscapes-19 train IDs after preparation.
    """

    dataset_dir = "data"
    domains = ["synthia_train", "cityscapes_train", "cityscapes_val"]

    CLASS_NAMES = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic light", "traffic sign", "vegetation",
        "sky", "person", "rider", "car",
        "bus", "motorcycle", "bicycle",
    ]
    NUM_CLASSES = len(CLASS_NAMES)

    CITYSCAPES_19_IDS = sorted(SYNTHIA_16_TO_CITYSCAPES_19.values())
    _CIT19_TO_LOCAL = {cid: i for i, cid in enumerate(CITYSCAPES_19_IDS)}

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.raw_synthia_dir = self._first_existing_dir(
            [
                osp.join(self.dataset_dir, "raw", "synthia"),
                osp.join(root, "synthia"),
            ],
            default=osp.join(self.dataset_dir, "raw", "synthia"),
        )
        self.raw_city_dir = self._first_existing_dir(
            [
                osp.join(self.dataset_dir, "raw", "cityscapes"),
                osp.join(root, "cityscapes"),
            ],
            default=osp.join(self.dataset_dir, "raw", "cityscapes"),
        )
        self.processed_dir = osp.join(self.dataset_dir, "processed")

        self.synthia_multilabel_file = osp.join(
            self.processed_dir, "synthia_multilabel", "multilabel.json"
        )

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        test_domains = (
            ["cityscapes_val"]
            if self._has_cityscapes_val()
            else cfg.DATASET.TARGET_DOMAINS
        )
        test = self._read_data(test_domains)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

        self._num_classes = self.NUM_CLASSES
        self._lab2cname = {i: self.CLASS_NAMES[i] for i in range(self.NUM_CLASSES)}
        self._classnames = list(self.CLASS_NAMES)

    @staticmethod
    def _first_existing_dir(candidates, default):
        for path in candidates:
            if osp.isdir(path):
                return path
        return default

    def _domain_config(self, domain_name):
        if domain_name == "synthia_train":
            return {
                "image_dir": osp.join(self.raw_synthia_dir, "images"),
                "label_dir": osp.join(self.raw_synthia_dir, "labels"),
                "split_file": osp.join(
                    self.raw_synthia_dir, "splits", "train.txt"
                ),
                "multilabel_file": self.synthia_multilabel_file,
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
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _to_filename(split_entry):
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
                if value and all(isinstance(v, (int, float)) for v in value):
                    labels = sorted(
                        int(v)
                        for v in value
                        if int(v) in self._CIT19_TO_LOCAL
                    )
                    output[filename] = labels
        return output

    _RAW_TO_TRAINID = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
        22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
        32: 17, 33: 18,
    }

    def _extract_labels_from_mask(self, mask_path):
        if not osp.isfile(mask_path):
            return []

        mask = np.array(Image.open(mask_path), dtype=np.int64)
        raw_ids = np.unique(mask)

        if any(int(v) > 18 and int(v) != 255 for v in raw_ids):
            train_ids = [
                self._RAW_TO_TRAINID[int(v)]
                for v in raw_ids
                if int(v) in self._RAW_TO_TRAINID
            ]
        else:
            train_ids = [int(v) for v in raw_ids if 0 <= int(v) <= 18]

        labels = [v for v in train_ids if v in self._CIT19_TO_LOCAL]
        return sorted(set(labels))

    def _read_data(self, input_domains):
        items = []

        for domain_index, domain_name in enumerate(input_domains):
            cfg = self._domain_config(domain_name)
            split_entries = self._read_split_file(cfg["split_file"])
            multilabel_map = self._load_multilabel_map(cfg["multilabel_file"])

            print(f"[SYNTHIA] domain={domain_name}, "
                  f"split={len(split_entries)} entries, "
                  f"multilabel_map={len(multilabel_map)} entries")
            if not multilabel_map:
                print(f"[SYNTHIA]   multilabel file: {cfg['multilabel_file']} "
                      f"(exists={osp.isfile(cfg['multilabel_file'])})")
                print(f"[SYNTHIA]   label_dir: {cfg['label_dir']} "
                      f"(exists={osp.isdir(cfg['label_dir'])})")

            missing_img = 0
            empty_labels = 0
            all_labels = set()

            for split_entry in split_entries:
                filename = self._to_filename(split_entry)
                impath = osp.join(cfg["image_dir"], filename)
                if not osp.isfile(impath):
                    missing_img += 1
                    continue

                labels = multilabel_map.get(filename, None)
                if labels is None:
                    labels = self._extract_labels_from_mask(
                        osp.join(cfg["label_dir"], filename)
                    )

                if not labels:
                    empty_labels += 1
                    labels = [0]

                all_labels.update(labels)

                for cit19_id in labels:
                    local_id = self._CIT19_TO_LOCAL.get(cit19_id, None)
                    if local_id is None:
                        continue
                    item = Datum(
                        impath=impath,
                        label=local_id,
                        domain=domain_index,
                        classname=self.CLASS_NAMES[local_id],
                    )
                    items.append(item)

            print(f"[SYNTHIA]   items={len(items)}, missing_img={missing_img}, "
                  f"empty_labels={empty_labels}, "
                  f"unique_cit19_ids={sorted(all_labels)}")

        return items

