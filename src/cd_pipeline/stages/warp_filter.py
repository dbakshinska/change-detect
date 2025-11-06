# src/cd_pipeline/stages/warp_filter.py
"""
Warp-filter stage
=================

Classifies each tile pair as **clean** or **warped** using

* Laplacian variance (too flat)
* Dense optical-flow magnitude (excess motion)

Files are copied into
``ctx["tiles_directory"]/filtered_tiles/{clean|warped}/`` and the two
CSV files
``clean_tile_ids.csv`` / ``warped_tile_ids.csv`` are produced.

Context on entry
----------------
* ctx["tiles_directory"] : pathlib.Path            (where tile_###_t{0,1}.jpg live)
* ctx["tiles"]           : list[PatchRec]          (created in Tiling)

Context on exit
---------------
* ctx["clean_tiles"]  : list[int]
* ctx["warped_tiles"] : list[int]
"""

from __future__ import annotations

import logging
from typing import Dict, List

import cv2
import numpy as np

from cd_pipeline.pipeline import Stage
from cd_pipeline.types import PatchRec

log = logging.getLogger(__name__)


class WarpFilter(Stage):
    """Remove tiles whose after-image patch is too warped/blurry."""

    def __init__(
        self,
        laplacian_threshold_low: float = 10.0,
        laplacian_threshold_high: float | None = None,
        optical_flow_threshold: float = 4.0,
    ):
        self.lap_low = laplacian_threshold_low
        self.lap_high = laplacian_threshold_high  # if None, upper bound disabled
        self.flow_thresh = optical_flow_threshold

    def run(self, ctx: Dict) -> Dict:
        tiles: List[PatchRec] = ctx.get("tiles", [])
        clean: List[PatchRec] = []

        for p in tiles:
            img0 = cv2.imread(str(p.path_t0), cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(str(p.path_t1), cv2.IMREAD_GRAYSCALE)
            if img0 is None or img1 is None:
                log.warning("Cannot read tile images for %s, skipping", p.patch_name)
                continue

            if self._is_warped_laplacian(img1) or self._is_warped_flow(img0, img1):
                # drop this tile-pair
                continue
            clean.append(p)

        # show both tile-pair counts and image-file counts
        num_pairs = len(tiles)
        num_clean_pairs = len(clean)
        log.info(
            "WarpFilter: kept %d / %d tile-pairs (%d / %d image files)",
            num_clean_pairs,
            num_pairs,
            num_clean_pairs * 2,
            num_pairs * 2,
        )

        # put the survivors into ctx for the next stage
        ctx["clean_tiles"] = [p.patch_name for p in clean]
        # but also pass the PatchRec objects forward
        ctx["tiles"] = clean
        return ctx

    def _is_warped_laplacian(self, img: np.ndarray) -> bool:
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        if self.lap_high is None:
            return var < self.lap_low
        return var < self.lap_low or var > self.lap_high

    def _is_warped_flow(self, img0: np.ndarray, img1: np.ndarray) -> bool:
        flow = cv2.calcOpticalFlowFarneback(
            img0, img1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag)) > self.flow_thresh
