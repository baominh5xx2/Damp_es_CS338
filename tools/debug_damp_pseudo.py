"""Audit DAMP target-domain pseudo-label distributions.

This script is diagnostic only. It uses the target-domain image-level labels
that are already built from masks to score pseudo labels, but those labels are
not fed back into training here.
"""

import argparse
import os.path as osp
import sys
from types import SimpleNamespace

import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dassl.config import get_cfg_default  # noqa: E402
from dassl.engine import build_trainer  # noqa: E402
from dassl.utils import set_random_seed  # noqa: E402

from train import extend_cfg  # noqa: E402
import datasets  # noqa: F401,E402
import trainers  # noqa: F401,E402


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def _quantiles(x):
    qs = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
                      device=x.device)
    vals = torch.quantile(x.float().flatten(), qs).detach().cpu().tolist()
    return " ".join(
        f"q{int(q * 100):02d}={v:.4f}" for q, v in zip(qs.cpu().tolist(), vals)
    )


def _score_probs(name, probs, labels, thresholds):
    print(f"\n[{name}] prob quantiles: {_quantiles(probs)}")
    print("  thres   pos%  cls/img  conf%  prec   recall micro_f1")
    eps = 1e-12
    for threshold in thresholds:
        pred = (probs >= threshold).float()
        tp = ((pred == 1) & (labels == 1)).sum().double()
        fp = ((pred == 1) & (labels == 0)).sum().double()
        fn = ((pred == 0) & (labels == 1)).sum().double()
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        pos_rate = pred.mean().item() * 100.0
        cls_per_img = pred.sum(dim=1).float().mean().item()
        conf_rate = (pred.sum(dim=1) > 0).float().mean().item() * 100.0
        print(
            f"  {threshold:5.2f} {pos_rate:6.2f} {cls_per_img:8.3f} "
            f"{conf_rate:6.2f} {prec.item():6.3f} {rec.item():7.3f} "
            f"{f1.item():8.3f}"
        )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--mix-lambda", type=float, default=0.0,
                        help="Blend weight for DAMP output_u vs naive pseudo logits")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.85])
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    trainer = build_trainer(cfg)
    if args.model_dir:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.set_model_mode("eval")

    device = trainer.device
    logit_scale = trainer.model.logit_scale.exp().detach()
    pseudo_temp_cfg = float(getattr(cfg.TRAINER.DAMP, "PSEUDO_TEMP", 0.0))
    pseudo_temp = (
        torch.tensor(pseudo_temp_cfg, device=device)
        if pseudo_temp_cfg > 0
        else logit_scale.to(device)
    ).clamp_min(1.0)

    print("*** Pseudo Diagnostic Config ***")
    print(f"dataset={cfg.DATASET.NAME}")
    print(f"target_domains={cfg.DATASET.TARGET_DOMAINS}")
    print(f"tau={float(cfg.TRAINER.DAMP.TAU):.4f}")
    print(f"mix_lambda={args.mix_lambda:.4f}")
    print(f"logit_scale={logit_scale.item():.4f}")
    print(f"pseudo_temp={pseudo_temp.item():.4f}")
    print(f"pixel_mean={cfg.INPUT.PIXEL_MEAN}")
    print(f"pixel_std={cfg.INPUT.PIXEL_STD}")

    all_labels = []
    all_output_logits = []
    all_naive_logits = []

    it_u = iter(trainer.train_loader_u)
    for batch_idx in range(args.num_batches):
        try:
            batch = next(it_u)
        except StopIteration:
            break

        image = batch["img"].to(device)
        impaths = batch.get("impath", [])
        label_fallback = batch.get("label", None)
        if label_fallback is not None:
            label_fallback = label_fallback.to(device)
        labels = trainer._build_multihot_labels(impaths, label_fallback).detach().cpu()

        output_u, _, pseudo_label_logits = trainer.model(
            image, ind=True, pse=True
        )

        all_labels.append(labels)
        all_output_logits.append(output_u.detach().cpu())
        all_naive_logits.append(pseudo_label_logits.detach().cpu())

    if not all_labels:
        raise RuntimeError("No target-domain batches were read")

    labels = torch.cat(all_labels, dim=0)
    output_logits = torch.cat(all_output_logits, dim=0)
    naive_logits = torch.cat(all_naive_logits, dim=0)

    print("\n*** Target Labels ***")
    print(f"samples={labels.shape[0]} classes={labels.shape[1]}")
    print(f"true cls/img={labels.sum(dim=1).float().mean().item():.3f}")
    print(f"true pos%={labels.float().mean().item() * 100.0:.2f}")
    print(f"per-class true positives={labels.sum(dim=0).int().tolist()}")

    pseudo_temp_cpu = pseudo_temp.detach().cpu()
    mix_lambda = float(args.mix_lambda)
    naive_uncal = torch.sigmoid(naive_logits)
    output_uncal = torch.sigmoid(output_logits)
    naive_cal = torch.sigmoid(naive_logits / pseudo_temp_cpu)
    output_cal = torch.sigmoid(output_logits / pseudo_temp_cpu)
    blend_cal = mix_lambda * output_cal + (1.0 - mix_lambda) * naive_cal

    _score_probs("naive_uncalibrated_sigmoid", naive_uncal, labels, args.thresholds)
    _score_probs("naive_calibrated_sigmoid", naive_cal, labels, args.thresholds)
    _score_probs("damp_output_calibrated_sigmoid", output_cal, labels, args.thresholds)
    _score_probs("blended_calibrated_pseudo", blend_cal, labels, args.thresholds)


if __name__ == "__main__":
    main()
