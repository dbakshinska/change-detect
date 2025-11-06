# src/cd_pipeline/types.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional


@dataclass
class PatchRec:
    """
    Metadata for a single patch pair extracted from a tile.

    Bounding boxes use (x1, y1, x2, y2) in three coordinate frames:
      • bbox_aligned – pixel coords in the *aligned* after-image
      • bbox_before  – pixel coords in the *before* image (raw before-tif)
      • bbox_after   – pixel coords in the *original* after-image (raw after-tif)
    """

    # identity & paths 
    patch_name: str
    path_t0: Path
    path_t1: Path
    tile_id: str

    # geometry 
    bbox_aligned: Tuple[int, int, int, int]
    bbox_before: Tuple[int, int, int, int]
    bbox_after: Tuple[int, int, int, int]

    # stage flags / metrics 
    pass_dino: bool = False
    pass_ssim: bool = False
    pass_yolo: bool = False

    dino_change: Optional[float] = None
    ssim: Optional[float] = None
    l1: Optional[float] = None
    yolo_detections: Optional[List[Dict]] = None


@dataclass
class OrthoPair:
    """Paths of the two orthophotos to be compared."""
    before_image_path: Path
    after_image_path: Path
