"""Convert SYNTHIA parquet dataset to raw images/labels/splits format.

Usage:
    python tools/prepare_synthia_hf.py \
        --parquet-dir /path/to/synthia-parquet/parquet \
        --output-root /path/to/data/raw/synthia

Output layout:
    <output-root>/
        images/train_000000.png ...
        labels/train_000000.png ...  (Cityscapes 19-class train IDs)
        splits/train.txt
"""

import argparse
import os
import os.path as osp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

SYNTHIA_16_TO_CITYSCAPES_19 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    5: 5, 6: 6, 7: 7, 8: 8,
    9: 10, 10: 11, 11: 12, 12: 13,
    13: 15, 14: 17, 15: 18,
}

_REMAP_LUT = np.full(256, 255, dtype=np.uint8)
for _src, _dst in SYNTHIA_16_TO_CITYSCAPES_19.items():
    _REMAP_LUT[_src] = _dst


def remap_label(label_np):
    """Remap SYNTHIA 16-class IDs [0..15] to Cityscapes 19-class train IDs."""
    return _REMAP_LUT[label_np.astype(np.uint8)]


def _process_row(args_tuple):
    """Worker function: decode image+label bytes, remap, save to disk."""
    global_idx, img_bytes, lbl_bytes, images_dir, labels_dir, skip_existing = args_tuple

    fname = f"train_{global_idx:06d}.png"
    image_out = osp.join(images_dir, fname)
    label_out = osp.join(labels_dir, fname)

    if skip_existing and osp.isfile(image_out) and osp.isfile(label_out):
        return global_idx, fname

    img = _to_pil(img_bytes).convert("RGB")
    img.save(image_out)

    lbl = _to_pil(lbl_bytes)
    lbl_np = np.array(lbl)
    if lbl_np.ndim == 3:
        lbl_np = lbl_np[:, :, 0]
    lbl_remapped = remap_label(lbl_np)
    Image.fromarray(lbl_remapped, mode="L").save(label_out)

    return global_idx, fname


def main(args):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("pip install pyarrow")

    parquet_dir = osp.abspath(args.parquet_dir)
    out_root = osp.abspath(args.output_root)

    images_dir = osp.join(out_root, "images")
    labels_dir = osp.join(out_root, "labels")
    splits_dir = osp.join(out_root, "splits")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    parquet_files = sorted(
        osp.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    print(f"Found {len(parquet_files)} parquet shards")

    num_workers = args.num_workers
    filenames = {}
    global_idx = 0
    start = time.time()
    total_submitted = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for shard_idx, pf in enumerate(parquet_files):
            table = pq.read_table(pf)
            n_rows = table.num_rows
            print(f"Shard {shard_idx+1}/{len(parquet_files)}: "
                  f"{osp.basename(pf)} ({n_rows} rows)")

            img_col = table.column("image")
            lbl_col = table.column("label")

            for row_idx in range(n_rows):
                img_val = img_col[row_idx].as_py()
                lbl_val = lbl_col[row_idx].as_py()

                fut = executor.submit(
                    _process_row,
                    (global_idx, img_val, lbl_val,
                     images_dir, labels_dir, args.skip_existing),
                )
                futures[fut] = global_idx
                global_idx += 1
                total_submitted += 1

        done = 0
        for fut in as_completed(futures):
            idx, fname = fut.result()
            filenames[idx] = fname
            done += 1
            if done % 500 == 0 or done == total_submitted:
                elapsed = time.time() - start
                rate = done / max(elapsed, 1e-6)
                print(f"  {done}/{total_submitted} images ({rate:.1f} img/s)",
                      flush=True)

    sorted_names = [filenames[i] for i in sorted(filenames.keys())]

    split_file = osp.join(splits_dir, "train.txt")
    with open(split_file, "w") as f:
        f.write("\n".join(sorted_names))

    elapsed = time.time() - start
    print(f"\nDone. {len(sorted_names)} images saved to {out_root} "
          f"in {elapsed:.1f}s ({len(sorted_names)/max(elapsed,1e-6):.1f} img/s)")
    print(f"Split file: {split_file}")


def _to_pil(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        return Image.fromarray(value)
    if isinstance(value, dict):
        if value.get("path") and osp.isfile(value["path"]):
            return Image.open(value["path"])
        if value.get("bytes"):
            from io import BytesIO
            return Image.open(BytesIO(value["bytes"]))
    raise TypeError(f"Unsupported image value type: {type(value)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-dir",
        type=str,
        required=True,
        help="Directory containing train-*.parquet files",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root: images/ labels/ splits/ will be created here",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist on disk",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel workers for image processing (default: 8)",
    )

    main(parser.parse_args())
