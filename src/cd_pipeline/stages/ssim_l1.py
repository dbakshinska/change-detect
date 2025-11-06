# src/cd_pipeline/stages/ssim_l1.py
"""
SSIM + L1 filtering stage
=========================

For every patch that *passed DINO*::

    - compute L1 pixel difference
    - compute grayscale SSIM
    - flag ``patch.pass_ssim`` if
        (SSIM < ssim_thresh) **and** (L1 > l1_thresh)

"""

from __future__ import annotations

import logging
from typing import Dict, List

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from cd_pipeline.pipeline import Stage
from cd_pipeline.types import PatchRec

log = logging.getLogger(__name__)


class SSIML1Filter(Stage):
    """Flag patches that survive SSIM + L1 thresholds."""

    def __init__(self, ssim_thresh: float = 0.90, l1_thresh: float = 22.0):
        self.ssim_thresh = ssim_thresh
        self.l1_thresh = l1_thresh

    def run(self, ctx: Dict) -> Dict:
        patches: List[PatchRec] = ctx.get("patches", [])
        kept = 0

        for patch in patches:
            if not patch.pass_dino:
                continue  # DINO already rejected

            # These paths were stored by DINOStage when saving PNGs
            img1 = cv2.imread(str(patch.path_t0))
            img2 = cv2.imread(str(patch.path_t1))
            if img1 is None or img2 is None:
                log.warning("Could not read patch PNGs for %s", patch)
                continue

            #  L1 + SSIM 
            l1_val = float(np.sum(np.abs(img1.astype("float32") - img2.astype("float32"))) / img1.size)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ssim_val = float(ssim(gray1, gray2, data_range=255))

            patch.l1 = l1_val
            patch.ssim = ssim_val
            patch.pass_ssim = (ssim_val < self.ssim_thresh) and (l1_val > self.l1_thresh)

            if patch.pass_ssim:
                kept += 1

        log.info("SSIM/L1 filter kept %s / %s patches", kept, len(patches))
        return ctx
