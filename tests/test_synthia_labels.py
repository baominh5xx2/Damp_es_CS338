"""Unit tests for SYNTHIA label mappings and consistency across the pipeline.

These tests avoid importing heavy dependencies (clip, cv2, torch, dassl)
by testing the pure-Python / numpy logic directly.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))


# ========================================================================
# 1. clip_text.py constants (no heavy deps)
# ========================================================================

class TestClipTextConstants:
    def test_synthia_class_names_count(self):
        from cam.clip_text import SYNTHIA_CLASS_NAMES
        assert len(SYNTHIA_CLASS_NAMES) == 16

    def test_synthia_prompt_names_count(self):
        from cam.clip_text import SYNTHIA_PROMPT_NAMES
        assert len(SYNTHIA_PROMPT_NAMES) == 16

    def test_mapping_keys_are_0_to_15(self):
        from cam.clip_text import SYNTHIA_16_TO_CITYSCAPES_19
        assert sorted(SYNTHIA_16_TO_CITYSCAPES_19.keys()) == list(range(16))

    def test_mapping_values_are_valid_cityscapes_ids(self):
        from cam.clip_text import SYNTHIA_16_TO_CITYSCAPES_19
        valid_cit_ids = set(range(19))
        for v in SYNTHIA_16_TO_CITYSCAPES_19.values():
            assert v in valid_cit_ids, f"Invalid Cityscapes ID: {v}"

    def test_mapping_values_are_unique(self):
        from cam.clip_text import SYNTHIA_16_TO_CITYSCAPES_19
        values = list(SYNTHIA_16_TO_CITYSCAPES_19.values())
        assert len(values) == len(set(values))

    def test_missing_classes_are_terrain_truck_train(self):
        from cam.clip_text import SYNTHIA_16_TO_CITYSCAPES_19, CITYSCAPES_CLASS_NAMES
        all_19 = set(range(19))
        synthia_cit_ids = set(SYNTHIA_16_TO_CITYSCAPES_19.values())
        missing = all_19 - synthia_cit_ids
        missing_names = {CITYSCAPES_CLASS_NAMES[i] for i in missing}
        assert missing_names == {"terrain", "truck", "train"}

    def test_class_names_match_cityscapes_subset(self):
        from cam.clip_text import (
            SYNTHIA_CLASS_NAMES,
            SYNTHIA_16_TO_CITYSCAPES_19,
            CITYSCAPES_CLASS_NAMES,
        )
        for synthia_idx, cit_idx in SYNTHIA_16_TO_CITYSCAPES_19.items():
            assert SYNTHIA_CLASS_NAMES[synthia_idx] == CITYSCAPES_CLASS_NAMES[cit_idx], (
                f"Mismatch at synthia[{synthia_idx}]={SYNTHIA_CLASS_NAMES[synthia_idx]} "
                f"vs cityscapes[{cit_idx}]={CITYSCAPES_CLASS_NAMES[cit_idx]}"
            )

    def test_prompt_names_match_cityscapes_subset(self):
        from cam.clip_text import (
            SYNTHIA_PROMPT_NAMES,
            SYNTHIA_16_TO_CITYSCAPES_19,
            CITYSCAPES_PROMPT_NAMES,
        )
        for synthia_idx, cit_idx in SYNTHIA_16_TO_CITYSCAPES_19.items():
            assert SYNTHIA_PROMPT_NAMES[synthia_idx] == CITYSCAPES_PROMPT_NAMES[cit_idx], (
                f"Mismatch at synthia[{synthia_idx}]={SYNTHIA_PROMPT_NAMES[synthia_idx]} "
                f"vs cityscapes[{cit_idx}]={CITYSCAPES_PROMPT_NAMES[cit_idx]}"
            )


# ========================================================================
# 2. prepare_synthia_hf.py — parquet label remap (no heavy deps)
# ========================================================================

class TestPrepareSynthiaRemap:
    def test_remap_label_basic(self):
        from prepare_synthia_hf import remap_label
        label = np.array([[0, 1, 8, 9, 15]], dtype=np.uint8)
        result = remap_label(label)
        assert result[0, 0] == 0   # road → 0
        assert result[0, 1] == 1   # sidewalk → 1
        assert result[0, 2] == 8   # vegetation → 8
        assert result[0, 3] == 10  # sky → 10 (skips terrain=9)
        assert result[0, 4] == 18  # bicycle → 18

    def test_remap_label_ignore_stays_255(self):
        from prepare_synthia_hf import remap_label
        label = np.array([[255]], dtype=np.uint8)
        result = remap_label(label)
        assert result[0, 0] == 255

    def test_remap_label_unmapped_becomes_255(self):
        from prepare_synthia_hf import remap_label
        label = np.array([[16, 17, 18, 19, 200]], dtype=np.uint8)
        result = remap_label(label)
        np.testing.assert_array_equal(result[0], [255, 255, 255, 255, 255])

    def test_remap_all_16_classes(self):
        from prepare_synthia_hf import remap_label, SYNTHIA_16_TO_CITYSCAPES_19
        label = np.arange(16, dtype=np.uint8).reshape(1, -1)
        result = remap_label(label)
        expected = np.array(
            [SYNTHIA_16_TO_CITYSCAPES_19[i] for i in range(16)],
            dtype=np.uint8,
        ).reshape(1, -1)
        np.testing.assert_array_equal(result, expected)

    def test_remap_is_injective(self):
        """Each SYNTHIA class maps to a unique Cityscapes class."""
        from prepare_synthia_hf import SYNTHIA_16_TO_CITYSCAPES_19
        values = list(SYNTHIA_16_TO_CITYSCAPES_19.values())
        assert len(values) == len(set(values))


# ========================================================================
# 3. Inline label logic tests (avoid importing cam.generate/cam.evaluate)
# ========================================================================

SYNTHIA_VALID_CIT19_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18}


def extract_synthia_labels_from_mask_standalone(mask_path):
    """Standalone copy of cam.generate._extract_synthia_labels_from_mask."""
    if not os.path.isfile(mask_path):
        return []
    img = Image.open(mask_path)
    mask = np.array(img, dtype=np.int64)
    img.close()
    raw_ids = np.unique(mask)
    return sorted(int(v) for v in raw_ids if int(v) in SYNTHIA_VALID_CIT19_IDS)


def map_mask_to_synthia16_standalone(mask):
    """Standalone copy of cam.evaluate.map_mask_to_synthia16."""
    from cam.clip_text import ID_TO_TRAINID
    mask = mask.astype(np.int64)
    if np.any((mask > 18) & (mask != 255)):
        out = np.full(mask.shape, 255, dtype=np.uint8)
        for raw_id, train_id in ID_TO_TRAINID.items():
            out[mask == raw_id] = train_id
        mask = out
    out = mask.copy()
    for v in np.unique(out):
        v = int(v)
        if v != 255 and v not in SYNTHIA_VALID_CIT19_IDS:
            out[out == v] = 255
    return out.astype(np.uint8)


class TestExtractSynthiaLabels:
    def _make_mask(self, ids, shape=(64, 64)):
        mask = np.full(shape, 255, dtype=np.uint8)
        for i, cls_id in enumerate(ids):
            row = i * (shape[0] // max(len(ids), 1))
            mask[row : row + 5, :] = cls_id
        return mask

    def test_extract_valid_synthia_ids(self):
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
                mask = self._make_mask([0, 8, 13, 18])  # road, veg, car, bicycle
                Image.fromarray(mask, mode="L").save(path)
            result = extract_synthia_labels_from_mask_standalone(path)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        assert result == [0, 8, 13, 18]

    def test_extract_filters_invalid_ids(self):
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
                mask = self._make_mask([0, 9, 14, 16, 255])
                Image.fromarray(mask, mode="L").save(path)
            result = extract_synthia_labels_from_mask_standalone(path)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        assert result == [0], f"Expected only road (0), got {result}"

    def test_extract_all_16_valid_ids(self):
        valid = sorted(SYNTHIA_VALID_CIT19_IDS)
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
                mask = self._make_mask(valid, shape=(200, 64))
                Image.fromarray(mask, mode="L").save(path)
            result = extract_synthia_labels_from_mask_standalone(path)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        assert result == valid

    def test_extract_returns_cityscapes19_ids(self):
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
                mask = self._make_mask([10, 15, 17])
                Image.fromarray(mask, mode="L").save(path)
            result = extract_synthia_labels_from_mask_standalone(path)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        assert 10 in result  # sky
        assert 15 in result  # bus
        assert 17 in result  # motorcycle

    def test_missing_file_returns_empty(self):
        result = extract_synthia_labels_from_mask_standalone("/nonexistent/path.png")
        assert result == []


class TestMapMaskToSynthia16:
    def test_keeps_valid(self):
        mask = np.array([[0, 1, 8], [10, 13, 18]], dtype=np.uint8)
        result = map_mask_to_synthia16_standalone(mask)
        np.testing.assert_array_equal(result, mask)

    def test_ignores_terrain_truck_train(self):
        mask = np.array([[9, 14, 16]], dtype=np.uint8)
        result = map_mask_to_synthia16_standalone(mask)
        np.testing.assert_array_equal(result, np.array([[255, 255, 255]], dtype=np.uint8))

    def test_preserves_ignore(self):
        mask = np.array([[255, 0, 255]], dtype=np.uint8)
        result = map_mask_to_synthia16_standalone(mask)
        assert result[0, 0] == 255
        assert result[0, 1] == 0
        assert result[0, 2] == 255

    def test_handles_raw_cityscapes_ids(self):
        mask = np.array([[7, 23, 26]], dtype=np.uint8)  # road=7, sky=23, car=26
        result = map_mask_to_synthia16_standalone(mask)
        assert result[0, 0] == 0   # road
        assert result[0, 1] == 10  # sky
        assert result[0, 2] == 13  # car

    def test_mixed_valid_and_invalid(self):
        mask = np.array([[0, 9, 13, 14, 18, 255]], dtype=np.uint8)
        result = map_mask_to_synthia16_standalone(mask)
        expected = np.array([[0, 255, 13, 255, 18, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_all_16_valid_ids_preserved(self):
        valid = sorted(SYNTHIA_VALID_CIT19_IDS)
        mask = np.array([valid], dtype=np.uint8)
        result = map_mask_to_synthia16_standalone(mask)
        np.testing.assert_array_equal(result[0], np.array(valid, dtype=np.uint8))


# ========================================================================
# 4. Dataset wrapper label logic (inline, no dassl import)
# ========================================================================

class TestDatasetWrapperLabels:
    def setup_method(self):
        from cam.clip_text import SYNTHIA_16_TO_CITYSCAPES_19
        cit19_ids = sorted(SYNTHIA_16_TO_CITYSCAPES_19.values())
        self.cit19_to_local = {cid: i for i, cid in enumerate(cit19_ids)}
        self.class_names = [
            "road", "sidewalk", "building", "wall", "fence",
            "pole", "traffic light", "traffic sign", "vegetation",
            "sky", "person", "rider", "car",
            "bus", "motorcycle", "bicycle",
        ]

    def test_cit19_to_local_has_16_entries(self):
        assert len(self.cit19_to_local) == 16

    def test_cit19_to_local_keys_are_valid(self):
        expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18}
        assert set(self.cit19_to_local.keys()) == expected

    def test_cit19_to_local_produces_contiguous_0_to_15(self):
        values = sorted(self.cit19_to_local.values())
        assert values == list(range(16))

    def test_class_names_indexed_by_local(self):
        from cam.clip_text import CITYSCAPES_CLASS_NAMES
        for cit_id, local_id in self.cit19_to_local.items():
            assert self.class_names[local_id] == CITYSCAPES_CLASS_NAMES[cit_id], (
                f"local[{local_id}]={self.class_names[local_id]} "
                f"!= cityscapes[{cit_id}]={CITYSCAPES_CLASS_NAMES[cit_id]}"
            )

    def test_extract_from_mask(self):
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
                mask = np.array([[0, 10, 18, 255]], dtype=np.uint8)
                Image.fromarray(mask, mode="L").save(path)
            img = Image.open(path)
            mask_read = np.array(img, dtype=np.int64)
            img.close()
            raw_ids = np.unique(mask_read)
            labels = sorted(int(v) for v in raw_ids if int(v) in self.cit19_to_local)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        assert labels == [0, 10, 18]

    def test_extract_filters_non_synthia_ids(self):
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
                mask = np.array([[9, 14, 16]], dtype=np.uint8)
                Image.fromarray(mask, mode="L").save(path)
            img = Image.open(path)
            mask_read = np.array(img, dtype=np.int64)
            img.close()
            raw_ids = np.unique(mask_read)
            labels = sorted(int(v) for v in raw_ids if int(v) in self.cit19_to_local)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        assert labels == []


# ========================================================================
# 5. DAMP trainer _normalize_label_ids (inline, no dassl/torch import)
# ========================================================================

class TestDampNormalizeLabelIds:
    def setup_method(self):
        self.synthia_cit19_to_local = {
            cid: i for i, cid in enumerate(sorted(
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18}
            ))
        }

    def _normalize_for_synthia(self, labels, num_classes=16):
        id_to_trainid = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
            22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
            32: 17, 33: 18,
        }
        labels = [int(v) for v in labels]
        if any(v > 18 and v != 255 for v in labels):
            mapped = [id_to_trainid[v] for v in labels if v in id_to_trainid]
        else:
            mapped = labels
        mapped = [
            self.synthia_cit19_to_local[v]
            for v in mapped
            if v in self.synthia_cit19_to_local
        ]
        return sorted(set(mapped))

    def test_basic_cit19_ids(self):
        result = self._normalize_for_synthia([0, 8, 13, 18])
        assert result == [0, 8, 12, 15]  # local indices

    def test_filters_terrain_truck_train(self):
        result = self._normalize_for_synthia([0, 9, 14, 16])
        assert result == [0]  # only road maps

    def test_all_16_valid_ids(self):
        valid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        result = self._normalize_for_synthia(valid)
        assert result == list(range(16))

    def test_sky_maps_to_local_9(self):
        result = self._normalize_for_synthia([10])  # sky in Cityscapes-19
        assert result == [9]  # local index 9

    def test_bicycle_maps_to_local_15(self):
        result = self._normalize_for_synthia([18])  # bicycle in Cityscapes-19
        assert result == [15]  # local index 15

    def test_motorcycle_maps_to_local_14(self):
        result = self._normalize_for_synthia([17])  # motorcycle in Cityscapes-19
        assert result == [14]  # local index 14

    def test_bus_maps_to_local_13(self):
        result = self._normalize_for_synthia([15])  # bus in Cityscapes-19
        assert result == [13]  # local index 13

    def test_empty_input(self):
        result = self._normalize_for_synthia([])
        assert result == []


# ========================================================================
# 6. Cross-module consistency
# ========================================================================

class TestCrossModuleConsistency:
    def test_clip_text_mapping_values_match_valid_set(self):
        from cam.clip_text import SYNTHIA_16_TO_CITYSCAPES_19
        assert set(SYNTHIA_16_TO_CITYSCAPES_19.values()) == SYNTHIA_VALID_CIT19_IDS

    def test_all_modules_agree_on_16_classes(self):
        from cam.clip_text import SYNTHIA_CLASS_NAMES, SYNTHIA_PROMPT_NAMES
        assert len(SYNTHIA_CLASS_NAMES) == 16
        assert len(SYNTHIA_PROMPT_NAMES) == 16

    def test_prepare_remap_covers_all_16_source_ids(self):
        from prepare_synthia_hf import SYNTHIA_16_TO_CITYSCAPES_19
        assert sorted(SYNTHIA_16_TO_CITYSCAPES_19.keys()) == list(range(16))

    def test_prepare_remap_matches_clip_text_mapping(self):
        from prepare_synthia_hf import SYNTHIA_16_TO_CITYSCAPES_19 as prep_map
        from cam.clip_text import SYNTHIA_16_TO_CITYSCAPES_19 as clip_map
        assert prep_map == clip_map

    def test_build_multilabel_valid_ids_match(self):
        from build_synthia_multilabel import SYNTHIA_VALID_CITYSCAPES_IDS
        assert SYNTHIA_VALID_CITYSCAPES_IDS == SYNTHIA_VALID_CIT19_IDS


# ========================================================================
# 7. End-to-end label flow
# ========================================================================

class TestEndToEndLabelFlow:
    def test_synthia_label_roundtrip(self):
        """parquet [0..15] → remap → PNG → extract → valid Cityscapes-19 IDs."""
        from prepare_synthia_hf import remap_label
        from cam.clip_text import CITYSCAPES_PROMPT_NAMES

        synthia_16_label = np.array([
            [0, 1, 2, 3],
            [9, 12, 14, 15],
        ], dtype=np.uint8)

        remapped = remap_label(synthia_16_label)
        assert remapped[0, 0] == 0   # road
        assert remapped[1, 0] == 10  # sky
        assert remapped[1, 2] == 17  # motorcycle

        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
                Image.fromarray(remapped, mode="L").save(path)
            cam_labels = extract_synthia_labels_from_mask_standalone(path)

            for lid in cam_labels:
                _ = CITYSCAPES_PROMPT_NAMES[lid]

            img = Image.open(path)
            gt_mask = np.array(img, dtype=np.uint8)
            img.close()
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

        eval_mask = map_mask_to_synthia16_standalone(gt_mask)
        unique_eval = set(np.unique(eval_mask))
        assert 9 not in unique_eval
        assert 14 not in unique_eval
        assert 16 not in unique_eval

    def test_cityscapes_gt_through_synthia16_filter(self):
        """Cityscapes GT with all 19 classes → synthia16 filter keeps only 16."""
        gt = np.arange(19, dtype=np.uint8).reshape(1, -1)
        result = map_mask_to_synthia16_standalone(gt)

        assert result[0, 9] == 255   # terrain → ignore
        assert result[0, 14] == 255  # truck → ignore
        assert result[0, 16] == 255  # train → ignore

        kept = [int(v) for v in result[0] if v != 255]
        assert len(kept) == 16

    def test_remap_roundtrip_with_extract(self):
        """After remap, extracting labels should give valid Cityscapes-19 IDs."""
        from prepare_synthia_hf import remap_label

        label = np.array([[0, 5, 9, 12, 15]], dtype=np.uint8)
        remapped = remap_label(label)

        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
                Image.fromarray(remapped, mode="L").save(path)
            extracted = extract_synthia_labels_from_mask_standalone(path)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

        expected_cit_ids = sorted({0, 5, 10, 13, 18})
        assert extracted == expected_cit_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
