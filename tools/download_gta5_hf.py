import argparse
import os
import os.path as osp


def main(args):
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Please install huggingface_hub first: pip install huggingface_hub"
        ) from exc

    out_dir = osp.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading dataset {args.repo_id} to {out_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=args.max_workers,
    )

    required = [
        osp.join(out_dir, "images"),
        osp.join(out_dir, "labels"),
        osp.join(out_dir, "splits", "train_full.txt"),
    ]
    missing = [p for p in required if not osp.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Download finished but required files are missing:\n" + "\n".join(missing)
        )

    print("GTA5 dataset is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Minhbao5xx2/gta5-kaggle-full",
        help="HuggingFace dataset repo id",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local output directory, e.g. <PROJECT_ROOT>/gta5-kaggle-full",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers for snapshot download",
    )

    main(parser.parse_args())
