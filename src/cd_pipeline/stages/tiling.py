# src/cd_pipeline/stages/tiling.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from cd_pipeline.pipeline import Stage
from cd_pipeline.types import PatchRec
from cd_pipeline.utils.tmp import tmp_dir

log = logging.getLogger(__name__)


class Tiling(Stage):
    """
    Generate overlapping tile pairs from the CHOSEN alignment (tightest-overlap).
    Requires:
      ctx["orthos"].before_image_path  (reference image for grid)
      ctx["aligned_image_path"]        (partner warped to reference)
      ctx["H_inv"] (optional)          (aligned -> original-partner)
    """

    def __init__(self, tile_size: int = 224, stride: int | None = None, valid_ratio: float = 0.9):
        self.tile_size = int(tile_size)
        self.stride = int(stride if stride is not None else (tile_size * 0.7))  # legacy stride
        self.valid_ratio = float(valid_ratio)

    def run(self, ctx: Dict) -> Dict:
        # Ignore any in-memory arrays / original_image to avoid path-dependent behavior
        if "before_window" in ctx or "after_window" in ctx or "original_image" in ctx:
            log.debug("Tiling: ignoring in-memory arrays; using disk images from chosen alignment.")

        before_p = ctx["orthos"].before_image_path
        aligned_p = ctx["aligned_image_path"]
        log.info("Tiling using reference=%s  aligned=%s", before_p.name, Path(aligned_p).name)

        img0 = cv2.imread(str(before_p), cv2.IMREAD_COLOR)   # reference (t0)
        img1 = cv2.imread(str(aligned_p), cv2.IMREAD_COLOR)  # aligned partner (t1)
        if img0 is None or img1 is None:
            raise IOError(f"Failed to read images:\n  before={before_p}\n  aligned={aligned_p}")

        # Crop to common canvas (so grid is identical no matter the original CLI order)
        Hc = min(img0.shape[0], img1.shape[0])
        Wc = min(img0.shape[1], img1.shape[1])
        img0 = img0[:Hc, :Wc]
        img1 = img1[:Hc, :Wc]
        h, w = Hc, Wc

        out_dir: Path = Path(ctx["out_dir"])
        tile_dir: Path = out_dir / "tiles"
        ctx["tiles_directory"] = tile_dir

        if h < self.tile_size or w < self.tile_size:
            log.warning("Image smaller than tile_size (%dx%d vs %dx%d) → no tiles.",
                        w, h, self.tile_size, self.tile_size)
            ctx["tiles"] = []
            ctx["tile_offsets"] = pd.DataFrame(columns=["x_offset", "y_offset"])
            return ctx

        # Valid-overlap mask (both sides nonzero)
        mask0 = (img0.sum(axis=2) > 0).astype(np.uint8)
        mask1 = (img1.sum(axis=2) > 0).astype(np.uint8)
        valid_overlap = cv2.bitwise_and(mask0, mask1)

        ny = (h - self.tile_size) // self.stride + 1
        nx = (w - self.tile_size) // self.stride + 1
        log.info("Tiling: %dx%d, tile_size=%d, stride=%d → up to %d tiles",
                 w, h, self.tile_size, self.stride, nx * ny)

        H_inv = ctx.get("H_inv", np.eye(3, dtype=np.float32)).astype(np.float32)

        def to_bgr3(img: np.ndarray) -> np.ndarray:
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 3:
                return img
            if img.shape[2] >= 4:
                return img[:, :, :3]
            out = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
            out[:, :, :img.shape[2]] = img
            return out

        tiles: List[PatchRec] = []
        for yi in range(ny):
            y_off = yi * self.stride
            y1_ = y_off + self.tile_size
            for xi in range(nx):
                x_off = xi * self.stride
                x1_ = x_off + self.tile_size

                roi = valid_overlap[y_off:y1_, x_off:x1_]
                if roi.size == 0 or int(roi.sum()) < self.valid_ratio * (self.tile_size * self.tile_size):
                    continue

                patch0 = img0[y_off:y1_, x_off:x1_]
                patch1 = img1[y_off:y1_, x_off:x1_]

                name = f"tile_{y_off}_{x_off}"
                p0 = tile_dir / f"{name}_t0.jpg"
                p1 = tile_dir / f"{name}_t1.jpg"
                p0.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p0), to_bgr3(patch0))
                cv2.imwrite(str(p1), to_bgr3(patch1))

                ba = (x_off, y_off, x1_, y1_)
                try:
                    corners = np.float32([[ba[0], ba[1]],[ba[2], ba[1]],[ba[2], ba[3]],[ba[0], ba[3]]]).reshape(-1,1,2)
                    transformed = cv2.perspectiveTransform(corners, H_inv).squeeze()
                    mn = transformed.min(axis=0).astype(int)
                    mx = transformed.max(axis=0).astype(int)
                    bc = (int(mn[0]), int(mn[1]), int(mx[0]), int(mx[1]))
                except Exception:
                    bc = ba

                tiles.append(PatchRec(
                    patch_name=name,
                    path_t0=p0, path_t1=p1,
                    tile_id=name,
                    bbox_aligned=ba,
                    bbox_before=ba,
                    bbox_after=bc,
                    pass_dino=False, pass_ssim=False, pass_yolo=False,
                ))

        ctx["tiles"] = tiles
        ctx["tile_offsets"] = pd.DataFrame(
            [{"tile_id": p.tile_id, "x_offset": p.bbox_aligned[0], "y_offset": p.bbox_aligned[1]} for p in tiles]
        ).set_index("tile_id")

        log.info("Tiling: generated %d tile-pairs → %d images", len(tiles), len(tiles) * 2)
        return ctx
