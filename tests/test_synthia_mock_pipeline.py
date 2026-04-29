#!/usr/bin/env python
"""End-to-end integration test for SYNTHIA→Cityscapes pipeline.

Creates a tiny mock dataset (2 images per domain) and runs every pipeline
stage on CPU to verify no crashes and correct label flow.

Requires: torch, dassl, clip, numpy, PIL  (run on server, not local dev)

Usage:
    python tests/test_synthia_mock_pipeline.py
    # or via pytest (each stage is a test):
    python -m pytest tests/test_synthia_mock_pipeline.py -v -s
"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

# ── project root on sys.path ──
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, osp.join(PROJECT_ROOT, "tools"))

# Cityscapes-19 train IDs used by SYNTHIA (16 classes)
VALID_CIT19 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18}

# ── shared temp dir ──
_TMPDIR = None


def _get_tmpdir():
    global _TMPDIR
    if _TMPDIR is None:
        _TMPDIR = tempfile.mkdtemp(prefix="synthia_mock_")
    return _TMPDIR


def _cleanup():
    global _TMPDIR
    if _TMPDIR and osp.isdir(_TMPDIR):
        shutil.rmtree(_TMPDIR, ignore_errors=True)
        _TMPDIR = None


# ===========================================================================
# Mock data creation
# ===========================================================================

def _make_mock_data():
    """Create a tiny dataset structure for 2 SYNTHIA + 2 Cityscapes images."""
    root = _get_tmpdir()
    rng = np.random.RandomState(42)

    # ---- SYNTHIA (labels in Cityscapes-19 trainID space) ----
    syn_img_dir = osp.join(root, "data", "raw", "synthia", "images")
    syn_lbl_dir = osp.join(root, "data", "raw", "synthia", "labels")
    syn_split_dir = osp.join(root, "data", "raw", "synthia", "splits")
    os.makedirs(syn_img_dir, exist_ok=True)
    os.makedirs(syn_lbl_dir, exist_ok=True)
    os.makedirs(syn_split_dir, exist_ok=True)

    syn_names = []
    syn_cit19_ids = [0, 1, 8, 10, 13, 18]  # road, sidewalk, veg, sky, car, bicycle
    for i in range(2):
        fname = f"train_{i:06d}.png"
        syn_names.append(fname)
        img = Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(osp.join(syn_img_dir, fname))

        lbl = np.full((64, 64), 255, dtype=np.uint8)
        for j, cid in enumerate(syn_cit19_ids):
            row = j * 10
            lbl[row:row+9, :] = cid
        Image.fromarray(lbl, mode="L").save(osp.join(syn_lbl_dir, fname))

    with open(osp.join(syn_split_dir, "train.txt"), "w") as f:
        f.write("\n".join(syn_names))

    # ---- Cityscapes (labels in RAW Cityscapes IDs) ----
    city_img_dir = osp.join(root, "data", "raw", "cityscapes", "images")
    city_lbl_dir = osp.join(root, "data", "raw", "cityscapes", "labels")
    city_split_dir = osp.join(root, "data", "raw", "cityscapes", "splits")
    os.makedirs(city_img_dir, exist_ok=True)
    os.makedirs(city_lbl_dir, exist_ok=True)
    os.makedirs(city_split_dir, exist_ok=True)

    city_names = []
    raw_city_ids = [7, 23, 26]  # road=7, sky=23, car=26 (raw IDs)
    for i in range(2):
        fname = f"train_{i:06d}.png"
        city_names.append(fname)
        img = Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(osp.join(city_img_dir, fname))

        lbl = np.full((64, 64), 0, dtype=np.uint8)
        for j, cid in enumerate(raw_city_ids):
            row = j * 20
            lbl[row:row+18, :] = cid
        Image.fromarray(lbl, mode="L").save(osp.join(city_lbl_dir, fname))

    with open(osp.join(city_split_dir, "train.txt"), "w") as f:
        f.write("\n".join(city_names))
    with open(osp.join(city_split_dir, "val.txt"), "w") as f:
        f.write("\n".join(city_names))

    return root


# ===========================================================================
# Stage 1: build_synthia_multilabel
# ===========================================================================

def test_build_multilabel():
    """Run build_synthia_multilabel logic on mock labels."""
    root = _make_mock_data()
    syn_lbl = osp.join(root, "data", "raw", "synthia", "labels")
    syn_split = osp.join(root, "data", "raw", "synthia", "splits", "train.txt")
    out_dir = osp.join(root, "data", "processed", "synthia_multilabel")
    os.makedirs(out_dir, exist_ok=True)

    from build_synthia_multilabel import read_split, extract_labels

    filenames = read_split(syn_split)
    assert len(filenames) == 2, f"Expected 2 files, got {len(filenames)}"

    multilabel = {}
    for entry in filenames:
        fname = osp.basename(entry)
        mask_path = osp.join(syn_lbl, fname)
        labels = extract_labels(mask_path)
        multilabel[fname] = labels
        assert len(labels) > 0, f"No labels found for {fname}"
        for lid in labels:
            assert lid in VALID_CIT19, f"Label {lid} not in valid SYNTHIA Cityscapes-19 IDs"

    ml_path = osp.join(out_dir, "multilabel.json")
    with open(ml_path, "w") as f:
        json.dump(multilabel, f)

    print(f"  [PASS] multilabel.json created: {multilabel}")


# ===========================================================================
# Stage 2: Dataset loading
# ===========================================================================

def test_dataset_loading():
    """Instantiate SYNTHIA dataset, verify classes and data counts."""
    root = _make_mock_data()
    _ensure_multilabel(root)

    from dassl.config import get_cfg_default
    import datasets  # noqa: F401 — register datasets

    cfg = get_cfg_default()
    cfg.DATASET.NAME = "SYNTHIA"
    cfg.DATASET.ROOT = root
    cfg.DATASET.SOURCE_DOMAINS = ["synthia_train"]
    cfg.DATASET.TARGET_DOMAINS = ["cityscapes_train"]

    from dassl.data.datasets import DATASET_REGISTRY
    ds_cls = DATASET_REGISTRY.get("SYNTHIA")
    ds = ds_cls(cfg)

    assert len(ds.classnames) > 1, f"Only {len(ds.classnames)} classes detected (expected 16)"
    print(f"  classnames ({len(ds.classnames)}): {ds.classnames}")
    print(f"  train_x: {len(ds.train_x)}, train_u: {len(ds.train_u)}, test: {len(ds.test)}")

    assert len(ds.train_x) > 0, "train_x is empty"
    assert len(ds.train_u) > 0, "train_u is empty"

    all_labels = set()
    for item in ds.train_x:
        all_labels.add(int(item.label))
    print(f"  train_x labels: {sorted(all_labels)}")
    assert len(all_labels) > 1, f"Only {len(all_labels)} unique labels in train_x"

    print("  [PASS] Dataset loading OK")


# ===========================================================================
# Stage 3: DAMP trainer init
# ===========================================================================

def test_damp_trainer_init():
    """Build DAMP trainer (data loader + model) on CPU."""
    root = _make_mock_data()
    _ensure_multilabel(root)

    import torch
    cfg = _build_cfg(root)

    import trainers  # noqa: F401
    from dassl.engine import build_trainer
    trainer = build_trainer(cfg)

    assert trainer.num_classes == 16, f"num_classes={trainer.num_classes}, expected 16"
    assert trainer.n_cls == 16, f"n_cls={trainer.n_cls}, expected 16"
    assert trainer.n_cls == trainer.num_classes, (
        f"n_cls={trainer.n_cls} != num_classes={trainer.num_classes}"
    )
    print(f"  num_classes={trainer.num_classes}, n_cls={trainer.n_cls}")
    print(f"  multi_label_lookup size: {len(trainer.multi_label_lookup)}")

    def _is_synthia_path(p):
        norm = p.replace("\\", "/").lower()
        return "/synthia/images/" in norm

    has_synthia = any(_is_synthia_path(k) for k in trainer.multi_label_lookup)
    print(f"  has synthia entries in lookup: {has_synthia}")
    assert has_synthia, "No SYNTHIA entries in multi_label_lookup"

    for k, v in trainer.multi_label_lookup.items():
        if _is_synthia_path(k):
            active = (v > 0).nonzero(as_tuple=True)[0].tolist()
            assert len(active) == 6, (
                f"SYNTHIA {osp.basename(k)}: expected 6 active labels, got {len(active)}: {active}"
            )
            assert active == [0, 1, 8, 9, 12, 15], (
                f"SYNTHIA {osp.basename(k)}: wrong label IDs {active}"
            )
            print(f"    {osp.basename(k)}: {len(active)} active labels, ids={active}")

    print("  [PASS] DAMP trainer init OK")
    return trainer


# ===========================================================================
# Stage 4: One training step
# ===========================================================================

def test_damp_forward_backward():
    """Run 1 forward-backward step to verify no crashes."""
    root = _make_mock_data()
    _ensure_multilabel(root)

    import torch
    cfg = _build_cfg(root)

    import trainers  # noqa: F401
    from dassl.engine import build_trainer
    trainer = build_trainer(cfg)

    trainer.set_model_mode("train")
    it_x = iter(trainer.train_loader_x)
    it_u = iter(trainer.train_loader_u)
    batch_x = next(it_x)
    batch_u = next(it_u)

    loss_summary = trainer.forward_backward(batch_x, batch_u)
    print(f"  loss_summary: {loss_summary}")

    assert "loss" in loss_summary
    assert np.isfinite(loss_summary["loss"]), f"Loss is not finite: {loss_summary['loss']}"
    print("  [PASS] forward_backward OK")


# ===========================================================================
# Stage 5: CAM generation
# ===========================================================================

def test_cam_generation():
    """Generate CAM for 1 mock image."""
    root = _make_mock_data()
    _ensure_multilabel(root)

    cam_dir = osp.join(root, "cams")
    os.makedirs(cam_dir, exist_ok=True)

    syn_img = osp.join(root, "data", "raw", "synthia", "images")
    syn_lbl = osp.join(root, "data", "raw", "synthia", "labels")
    syn_split = osp.join(root, "data", "raw", "synthia", "splits", "train.txt")

    mock_args = argparse.Namespace(
        dataset="synthia",
        img_root=syn_img,
        label_root=syn_lbl,
        split_file=syn_split,
        cam_out_dir=cam_dir,
        model="ViT-B/16",
        num_workers=1,
        box_threshold=0.4,
        image_scale=1.0,
        max_long_side=1024,
        max_images=1,
        skip_existing=False,
        no_refine=True,
        damp_prompt_ckpt="",
        damp_n_ctx=-1,
        damp_mix_alpha=1.0,
        damp_use_plain_names=True,
    )

    with open(syn_split, "r") as f:
        train_list = [line.strip() for line in f if line.strip()]
    train_list = train_list[:1]

    from cam.generate import perform, split_dataset
    dataset_list = split_dataset(train_list, n_splits=1)
    perform(0, dataset_list, mock_args)

    npy_files = [f for f in os.listdir(cam_dir) if f.endswith(".npy")]
    assert len(npy_files) > 0, f"No .npy CAM files generated in {cam_dir}"

    cam_data = np.load(osp.join(cam_dir, npy_files[0]), allow_pickle=True).item()
    print(f"  CAM file keys: {list(cam_data.keys())}")
    assert "attn_highres" in cam_data or "highres" in cam_data, (
        f"Expected 'attn_highres' or 'highres' key, got: {list(cam_data.keys())}"
    )

    if "attn_highres" in cam_data:
        cam = cam_data["attn_highres"]
        print(f"  attn_highres shape: {cam.shape}")
        keys = cam_data.get("keys", None)
        if keys is not None:
            print(f"  keys: {keys}")
            for k in keys:
                assert int(k) in VALID_CIT19, f"CAM key {k} not in valid SYNTHIA IDs"

    print("  [PASS] CAM generation OK")


# ===========================================================================
# Stage 6: CAM evaluation
# ===========================================================================

def test_cam_evaluation():
    """Create mock CAMs + GT, run evaluation, verify mIoU."""
    root = _make_mock_data()
    cam_dir = osp.join(root, "mock_cams")
    os.makedirs(cam_dir, exist_ok=True)

    syn_lbl = osp.join(root, "data", "raw", "synthia", "labels")
    syn_split = osp.join(root, "data", "raw", "synthia", "splits", "train.txt")

    with open(syn_split, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    for fname in filenames:
        gt = np.array(Image.open(osp.join(syn_lbl, fname)), dtype=np.uint8)
        h, w = gt.shape
        n_class = 19
        cam_arr = np.random.rand(n_class, h, w).astype(np.float32)
        for cid in range(n_class):
            if cid in VALID_CIT19:
                mask_pixels = (gt == cid)
                if mask_pixels.any():
                    cam_arr[cid][mask_pixels] = 1.0

        stem = osp.splitext(fname)[0]
        np.save(osp.join(cam_dir, stem + ".npy"), {
            "attn_highres": cam_arr,
            "keys": np.array(list(range(n_class))),
        })

    from cam.evaluate import run_eval_cam

    result = run_eval_cam(
        eval_list=filenames,
        cam_dir=cam_dir,
        gt_root=syn_lbl,
        cam_type="attn_highres",
        cam_eval_thres=0.5,
        n_class=19,
        dataset="synthia",
        use_bg_channel=True,
    )

    print(f"  mIoU: {result['Mean IoU']:.4f}")
    print(f"  Pixel Acc: {result['Pixel Accuracy']:.4f}")
    iu = result["IoU Array"]
    for i, v in enumerate(iu):
        if v > 0:
            print(f"    class {i}: IoU={v:.4f}")

    assert result["Mean IoU"] >= 0, "mIoU should be non-negative"
    print("  [PASS] CAM evaluation OK")


# ===========================================================================
# Helpers
# ===========================================================================

def _ensure_multilabel(root):
    """Build multilabel.json if it doesn't exist yet."""
    out_dir = osp.join(root, "data", "processed", "synthia_multilabel")
    ml_path = osp.join(out_dir, "multilabel.json")
    if osp.isfile(ml_path):
        return

    os.makedirs(out_dir, exist_ok=True)
    syn_lbl = osp.join(root, "data", "raw", "synthia", "labels")
    syn_split = osp.join(root, "data", "raw", "synthia", "splits", "train.txt")

    from build_synthia_multilabel import read_split, extract_labels
    filenames = read_split(syn_split)
    multilabel = {}
    for entry in filenames:
        fname = osp.basename(entry)
        labels = extract_labels(osp.join(syn_lbl, fname))
        multilabel[fname] = labels
    with open(ml_path, "w") as f:
        json.dump(multilabel, f)


def _build_cfg(root):
    """Build config for SYNTHIA by reusing train.py's setup_cfg + yaml."""
    import torch
    import datasets  # noqa: F401

    sys.path.insert(0, PROJECT_ROOT)
    from train import setup_cfg

    config_file = osp.join(PROJECT_ROOT, "configs", "trainers", "damp_synthia.yaml")
    out_dir = osp.join(root, "output_test")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mock_args = argparse.Namespace(
        config_file=config_file,
        opts=[
            "DATASET.ROOT", root,
            "OUTPUT_DIR", out_dir,
            "DATALOADER.TRAIN_X.BATCH_SIZE", "2",
            "DATALOADER.TRAIN_U.BATCH_SIZE", "2",
            "DATALOADER.TEST.BATCH_SIZE", "2",
            "DATALOADER.NUM_WORKERS", "0",
            "OPTIM.MAX_EPOCH", "1",
            "OPTIM.WARMUP_EPOCH", "0",
            "TRAIN.CHECKPOINT_FREQ", "0",
            "TEST.NO_TEST", "True",
            "USE_CUDA", str(device == "cuda"),
        ],
    )
    return setup_cfg(mock_args)


# ===========================================================================
# Main (for running as script)
# ===========================================================================

def main():
    stages = [
        ("Stage 1: build_multilabel", test_build_multilabel),
        ("Stage 2: dataset_loading", test_dataset_loading),
        ("Stage 3: damp_trainer_init", test_damp_trainer_init),
        ("Stage 4: damp_forward_backward", test_damp_forward_backward),
        ("Stage 5: cam_generation", test_cam_generation),
        ("Stage 6: cam_evaluation", test_cam_evaluation),
    ]

    passed = 0
    failed = 0
    for name, func in stages:
        print(f"\n{'='*60}")
        print(f" {name}")
        print(f"{'='*60}")
        try:
            func()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] {name}: {e}")

    print(f"\n{'='*60}")
    print(f" Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    _cleanup()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
