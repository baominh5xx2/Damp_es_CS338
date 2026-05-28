"""Train a lightweight downstream segmentation model from exported pseudo masks.

This is intentionally self-contained so it can run from Colab after
``pipeline_main.ipynb`` exports ``image_path mask_path`` pairs.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cam.clip_text import CITYSCAPES_CLASS_NAMES
from cam.evaluate import map_mask_to_trainid, resolve_label_path


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def resolve_image_path(root, entry):
    root = Path(root)
    base = Path(entry).name
    stem = Path(base).stem
    candidates = [
        root / entry,
        root / base,
        root / f"{stem}.png",
        root / f"{stem}.jpg",
        root / f"{stem}.jpeg",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return root / base


def sanitize_mask(mask, n_class, ignore_index):
    mask = mask.astype(np.int64)
    mask[(mask < 0) | ((mask >= n_class) & (mask != ignore_index))] = ignore_index
    return mask.astype(np.int64)


class PairSegDataset(Dataset):
    def __init__(self, pair_file, image_size, n_class=19, ignore_index=255,
                 max_items=None, train=True):
        self.items = []
        with open(pair_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, mask_path = line.split(maxsplit=1)
                if os.path.isfile(image_path) and os.path.isfile(mask_path):
                    self.items.append((image_path, mask_path))
                if max_items is not None and len(self.items) >= max_items:
                    break
        if not self.items:
            raise RuntimeError(f"No valid image/mask pairs found in {pair_file}")
        self.image_size = tuple(image_size)
        self.n_class = n_class
        self.ignore_index = ignore_index
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, mask_path = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        h, w = self.image_size
        image = image.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)

        image = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        mask = sanitize_mask(np.asarray(mask), self.n_class, self.ignore_index)
        mask = torch.from_numpy(mask).long()
        return image, mask


class ValSegDataset(Dataset):
    def __init__(self, image_root, label_root, split_file, image_size,
                 n_class=19, ignore_index=255, max_items=None):
        with open(split_file, "r") as f:
            entries = [line.strip() for line in f if line.strip()]
        if max_items is not None:
            entries = entries[:max_items]

        self.items = []
        for entry in entries:
            image_path = resolve_image_path(image_root, entry)
            label_path = Path(resolve_label_path(str(label_root), entry))
            if image_path.is_file() and label_path.is_file():
                self.items.append((image_path, label_path))
        if not self.items:
            raise RuntimeError(
                f"No valid validation images/labels found from {split_file}"
            )
        self.image_size = tuple(image_size)
        self.n_class = n_class
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label_path = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(Image.open(label_path), dtype=np.uint8)
        mask = map_mask_to_trainid(mask)

        h, w = self.image_size
        image = image.resize((w, h), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((w, h), Image.NEAREST)

        image = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        mask = sanitize_mask(np.asarray(mask), self.n_class, self.ignore_index)
        mask = torch.from_numpy(mask).long()
        return image, mask


def build_model(n_class, pretrained_backbone):
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    weights_backbone = None
    if pretrained_backbone:
        try:
            from torchvision.models import MobileNet_V3_Large_Weights
            weights_backbone = MobileNet_V3_Large_Weights.DEFAULT
        except Exception as exc:
            print(f"WARNING: pretrained backbone unavailable ({exc}); using random init")
            weights_backbone = None

    try:
        return lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=n_class,
        )
    except Exception as exc:
        if weights_backbone is None:
            raise
        print(f"WARNING: failed to load pretrained backbone ({exc}); retry random init")
        return lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=n_class,
        )


def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    lt = label_true[mask].astype(int)
    lp = label_pred[mask].astype(int)
    lp[(lp < 0) | (lp >= n_class)] = n_class
    return np.bincount(
        (n_class + 1) * lt + lp,
        minlength=n_class * (n_class + 1),
    ).reshape(n_class, n_class + 1)


def scores_from_hist(hist, n_class):
    tp = np.diag(hist[:, :n_class])
    gt_count = hist.sum(axis=1)
    pred_count = hist[:, :n_class].sum(axis=0)
    acc = tp.sum() / max(gt_count.sum(), 1.0)
    mean_acc = np.nanmean(tp / np.maximum(gt_count, 1.0))
    iu = tp / np.maximum(gt_count + pred_count - tp, 1.0)
    valid = gt_count > 0
    mean_iu = np.nanmean(iu[valid])
    freq = gt_count / max(gt_count.sum(), 1.0)
    fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()
    return {
        "Pixel Accuracy": float(acc),
        "Mean Accuracy": float(mean_acc),
        "Mean IoU": float(mean_iu),
        "FW IoU": float(fw_iou),
        "Class IoU": {str(i): float(iu[i]) for i in range(n_class)},
        "GT Pixels": {str(i): int(gt_count[i]) for i in range(n_class)},
    }


@torch.no_grad()
def evaluate(model, loader, device, n_class):
    model.eval()
    hist = np.zeros((n_class, n_class + 1), dtype=np.float64)
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)["out"]
        preds = logits.argmax(dim=1).cpu().numpy()
        gts = masks.numpy()
        for gt, pred in zip(gts, preds):
            hist += fast_hist(gt.flatten(), pred.flatten(), n_class)
    return scores_from_hist(hist, n_class)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    image_size = (args.image_height, args.image_width)
    train_set = PairSegDataset(
        args.train_pairs,
        image_size=image_size,
        n_class=args.num_classes,
        ignore_index=args.ignore_index,
        max_items=args.max_train,
        train=True,
    )
    val_set = ValSegDataset(
        args.val_image_root,
        args.val_label_root,
        args.val_split_file,
        image_size=image_size,
        n_class=args.num_classes,
        ignore_index=args.ignore_index,
        max_items=args.max_val,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"Device: {device}")
    print(f"Train pairs: {len(train_set)}")
    print(f"Val images : {len(val_set)}")
    print(f"Image size : {image_size}")

    model = build_model(args.num_classes, args.pretrained_backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    best_miou = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, masks in tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)["out"]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        metrics = evaluate(model, val_loader, device, args.num_classes)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses)) if losses else 0.0
        print(
            f"epoch {epoch}: loss={metrics['train_loss']:.4f} "
            f"mIoU={metrics['Mean IoU']:.4f} PA={metrics['Pixel Accuracy']:.4f}"
        )

        if metrics["Mean IoU"] > best_miou:
            best_miou = metrics["Mean IoU"]
            best_metrics = metrics
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "args": vars(args),
                },
                output_dir / "model_best.pth",
            )

    if best_metrics is None:
        best_metrics = evaluate(model, val_loader, device, args.num_classes)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print("\nDownstream segmentation result")
    print(f"  best epoch      : {best_metrics.get('epoch', 'n/a')}")
    print(f"  Pixel Accuracy  : {best_metrics['Pixel Accuracy']:.4f}")
    print(f"  Mean Accuracy   : {best_metrics['Mean Accuracy']:.4f}")
    print(f"  Mean IoU        : {best_metrics['Mean IoU']:.4f}")
    print(f"  FW IoU          : {best_metrics['FW IoU']:.4f}")
    print("\nPer-class IoU")
    print(f"  {'id':>2s} {'class':<16s} {'IoU':>8s} {'gt_pixels':>10s}")
    print(f"  {'--':>2s} {'-' * 16:<16s} {'-' * 8:>8s} {'-' * 10:>10s}")
    for cid, cname in enumerate(CITYSCAPES_CLASS_NAMES):
        print(
            f"  {cid:>2d} {cname:<16s} "
            f"{best_metrics['Class IoU'][str(cid)]:>8.4f} "
            f"{best_metrics['GT Pixels'][str(cid)]:>10d}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--val_image_root", required=True)
    parser.add_argument("--val_label_root", required=True)
    parser.add_argument("--val_split_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
