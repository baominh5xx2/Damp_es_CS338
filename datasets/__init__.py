"""Unified dataset module for Damp_es.

All datasets are registered with Dassl's ``DATASET_REGISTRY`` on import so the
trainer can resolve them by name (``cfg.DATASET.NAME``).

Built-in Dassl datasets (Office-Home, VisDA17, MiniDomainNet) are imported via
``dassl.data.datasets`` automatically; we only need to register the custom ones
here.
"""

from .voc12 import VOC12
from .coco14 import COCO14
from .gta5 import GTA5
from .synthia import SYNTHIA

__all__ = ["VOC12", "COCO14", "GTA5", "SYNTHIA"]
