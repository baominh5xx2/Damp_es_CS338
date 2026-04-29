import argparse
import os
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image


def to_pil_image(value):
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, np.ndarray):
        return Image.fromarray(value)

    # HuggingFace image feature can be dict with path/bytes.
    if isinstance(value, dict):
        if value.get("path") and osp.isfile(value["path"]):
            return Image.open(value["path"])
        if value.get("bytes"):
            from io import BytesIO

            return Image.open(BytesIO(value["bytes"]))

    raise TypeError(f"Unsupported image value type: {type(value)}")


def detect_column(sample, candidates, required=True):
    for key in candidates:
        if key in sample:
            return key

    if required:
        raise KeyError(f"Cannot find required column among: {candidates}")

    return None


def save_split(
    dataset_split,
    split_name,
    image_key,
    label_key,
    out_root,
    log_every=200,
    skip_existing=True,
    num_workers=1,
):
    images_dir = osp.join(out_root, "images")
    labels_dir = osp.join(out_root, "labels")
    splits_dir = osp.join(out_root, "splits")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    if split_name in ("validation", "val"):
        split_file = osp.join(splits_dir, "val.txt")
    elif split_name == "test":
        split_file = osp.join(splits_dir, "test.txt")
    else:
        split_file = osp.join(splits_dir, "train.txt")

    total = len(dataset_split)
    start = time.time()
    print(f"Converting split={split_name} total={total} ...", flush=True)

    # If an old export was created with a different sampling ratio, the
    # filename pattern can collide (e.g., train_000123.png) and silently keep
    # stale content. In that case, force rewrite this split.
    effective_skip_existing = skip_existing
    if effective_skip_existing and osp.isfile(split_file):
        with open(split_file, "r") as f:
            existing_names = [line.strip() for line in f if line.strip()]
        if len(existing_names) != total:
            print(
                f"[{split_name}] Existing split size {len(existing_names)} != target {total}; rewriting split files",
                flush=True,
            )
            effective_skip_existing = False

    def process_item(i):
        sample = dataset_split[i]
        fname = f"{split_name}_{i:06d}.png"
        image_out = osp.join(images_dir, fname)
        label_out = osp.join(labels_dir, fname)

        if effective_skip_existing and osp.isfile(image_out):
            if label_key is None:
                return i, fname
            if osp.isfile(label_out):
                return i, fname

        image = to_pil_image(sample[image_key]).convert("RGB")
        image.save(image_out)

        if label_key is not None and label_key in sample:
            label = to_pil_image(sample[label_key])
            label_np = np.array(label)
            if label_np.ndim == 3:
                label_np = label_np[:, :, 0]
            label = Image.fromarray(label_np.astype(np.uint8), mode="L")
            label.save(label_out)

        return i, fname

    file_names = [None] * total
    done = 0
    if num_workers <= 1:
        for i in range(total):
            idx, fname = process_item(i)
            file_names[idx] = fname
            done += 1
            if done % log_every == 0 or done == total:
                elapsed = time.time() - start
                rate = done / max(elapsed, 1e-6)
                print(f"[{split_name}] {done}/{total} ({rate:.2f} items/s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(process_item, i) for i in range(total)]
            for fut in as_completed(futures):
                idx, fname = fut.result()
                file_names[idx] = fname
                done += 1
                if done % log_every == 0 or done == total:
                    elapsed = time.time() - start
                    rate = done / max(elapsed, 1e-6)
                    print(f"[{split_name}] {done}/{total} ({rate:.2f} items/s)", flush=True)

    file_names = [fname for fname in file_names if fname is not None]

    with open(split_file, "w") as f:
        f.write("\n".join(file_names))

    print(f"Saved split {split_name} with {len(file_names)} samples -> {split_file}", flush=True)


def main(args):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Please install datasets package first: pip install datasets"
        ) from exc

    out_root = osp.abspath(args.output_root)
    os.makedirs(out_root, exist_ok=True)

    print(f"Loading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id)

    split_names = list(ds.keys())
    if not split_names:
        raise RuntimeError("No splits found in dataset")

    sample = ds[split_names[0]][0]
    image_key = detect_column(
        sample,
        ["image", "img", "leftImg8bit", "pixel_values", "input_image"],
        required=True,
    )
    label_key = detect_column(
        sample,
        ["label", "labels", "mask", "segmentation_mask", "annotation", "semantic_segmentation"],
        required=False,
    )

    print(f"Detected columns: image={image_key}, label={label_key}")

    wanted_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    wanted_lower = {s.lower() for s in wanted_splits}

    sample_ratio = 1.0
    sample_seed = 42

    for split_name in split_names:
        if wanted_lower and split_name.lower() not in wanted_lower:
            print(f"Skipping split={split_name} (not in --splits)", flush=True)
            continue

        full_split = ds[split_name]
        split_len = len(full_split)
        if sample_ratio < 1.0:
            sample_size = max(1, int(split_len * sample_ratio))
            rng = np.random.default_rng(sample_seed)
            sampled_indices = rng.choice(split_len, size=sample_size, replace=False)
            split_ds = full_split.select(sorted(sampled_indices.tolist()))
            print(
                f"Sampling split={split_name}: {sample_size}/{split_len} ({sample_ratio*100:.0f}%)",
                flush=True,
            )
        else:
            split_ds = full_split
            print(
                f"Sampling split={split_name}: {split_len}/{split_len} (100%)",
                flush=True,
            )

        if args.max_per_split > 0:
            split_ds = split_ds.select(range(min(args.max_per_split, len(split_ds))))
        save_split(
            split_ds,
            split_name,
            image_key,
            label_key,
            out_root,
            log_every=args.log_every,
            skip_existing=not args.no_skip_existing,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="Chris1/cityscapes",
        help="HuggingFace dataset id",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output folder in cityscapes raw format: images/ labels/ splits/",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=-1,
        help="Optional cap for debugging; -1 means all",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print progress every N samples",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel workers for decoding/saving samples",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="Comma-separated splits to export (e.g., train,validation,test)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rewrite existing files instead of resuming",
    )

    main(parser.parse_args())
