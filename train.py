"""DAMP training entry point (Dassl-based).

Usage:
    python train.py --config-file configs/trainers/damp_voc12.yaml \
        DATASET.ROOT /path/to/data OUTPUT_DIR output/damp/voc12

    python train.py --config-file configs/trainers/damp_gta5.yaml \
        DATASET.ROOT /path/to/data OUTPUT_DIR output/damp/gta5
"""

import argparse

import torch

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import setup_logger, set_random_seed, collect_env_info

import datasets  # noqa: F401 — register custom datasets
import trainers  # noqa: F401 — register custom trainers

# ---------------------------------------------------------------------------
# Compat: PyTorch ≥2.4 removed the `verbose` arg from LRScheduler.__init__.
# Dassl's _LRSchedulerBase still passes it, causing TypeError. Patch it.
# ---------------------------------------------------------------------------
import dassl.optim.lr_scheduler as _lr_mod

_base_name = "_BaseWarmupScheduler" if hasattr(_lr_mod, "_BaseWarmupScheduler") else "_LRSchedulerBase"
if hasattr(_lr_mod, _base_name):
    _Base = getattr(_lr_mod, _base_name)
    _orig_base_init = _Base.__init__

    def _patched_base_init(self, optimizer, successor, warmup_epoch,
                           last_epoch=-1, verbose=False):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        try:
            _lr_mod._LRScheduler.__init__(self, optimizer, last_epoch, verbose)
        except TypeError:
            _lr_mod._LRScheduler.__init__(self, optimizer, last_epoch)

    _Base.__init__ = _patched_base_init


def extend_cfg(cfg):
    """Register DAMP-specific config keys (must be called before merge)."""
    from yacs.config import CfgNode as CN

    cfg.MODEL.BACKBONE.PATH = "./assets"
    cfg.MODEL.INIT_WEIGHTS_CTX = None
    cfg.MODEL.INIT_WEIGHTS_PRO = None

    cfg.TRAINER.DAMP = CN()
    cfg.TRAINER.DAMP.N_CTX = 16
    cfg.TRAINER.DAMP.N_CLS = 2
    cfg.TRAINER.DAMP.CSC = False
    cfg.TRAINER.DAMP.PREC = "fp16"
    cfg.TRAINER.DAMP.TAU = 0.5
    cfg.TRAINER.DAMP.U = 1.0
    cfg.TRAINER.DAMP.IND = 1.0
    cfg.TRAINER.DAMP.IM = 1.0
    cfg.TRAINER.DAMP.STRONG_TRANSFORMS = []

    cfg.OPTIM_C = cfg.OPTIM.clone()


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="DAMP Training")
    parser.add_argument("--config-file", type=str, default="",
                        help="Path to Dassl YAML config")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--model-dir", type=str, default="",
                        help="Load model from this directory (for eval)")
    parser.add_argument("--load-epoch", type=int, default=None,
                        help="Load checkpoint from this epoch (None=best)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Override config keys (e.g. DATASET.ROOT /data)")
    args = parser.parse_args()

    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("*** Config ***")
    print(cfg)
    print("Environment:")
    print(collect_env_info())

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    trainer.train()


if __name__ == "__main__":
    main()
