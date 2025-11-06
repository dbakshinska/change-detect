# src/cd_pipeline/stages/alignment.py
"""
Alignment stage
===============

Aligns *after* orthophoto (`T₁`) to the *before* orthophoto (`T₀`)
with SIFT + FLANN and writes the warped image to a run-scoped temp
directory created via ``tmp_dir("align")``.

Context keys on entry
---------------------
* ``ctx["orthos"]`` : cd_pipeline.types.OrthoPair

Context keys produced
---------------------
* ``ctx["aligned_image_path"]`` : pathlib.Path
* ``ctx["H"]``                  : np.ndarray 3×3  (T₁ → T₀)
* ``ctx["H_inv"]``              : np.ndarray 3×3  (T₀ → T₁)
* ``ctx["original_image"]``     : np.ndarray (T₀ RGB image)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from cd_pipeline.pipeline import Stage
from cd_pipeline.utils.tmp import tmp_dir

log = logging.getLogger(__name__)


class Alignment(Stage):
    """Align two images using SIFT keypoints and FLANN matching."""

    def run(self, ctx: Dict) -> Dict:
        """
        Run SIFT + FLANN alignment and update *ctx* with aligned path & matrices.

        Parameters
        ----------
        ctx : dict
            Pipeline context.  Requires ``ctx["orthos"]`` to be set.

        Returns
        -------
        dict
            Updated context.
        """
        pair = ctx["orthos"]
        img1_path = str(pair.before_image_path)
        img2_path = str(pair.after_image_path)

        out_dir: Path = ctx["out_dir"]
        align_dir: Path = out_dir / "align"
        align_dir.mkdir(parents=True, exist_ok=True)
        aligned_img_path = align_dir / "aligned.tiff"

        img1_full, img2_aligned, H = self.align_images_sift(
            img1_path,
            img2_path,
            str(aligned_img_path),
        )

        ctx["aligned_image_path"] = aligned_img_path
        ctx["H"] = H
        ctx["H_inv"] = np.linalg.inv(H)
        ctx["original_image"] = img1_full  # reference (before)
        return ctx

    @staticmethod
    def align_images_sift(
        img1_path: str,
        img2_path: str,
        output_path: str,
        scale_factor: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align *img2* onto *img1* using SIFT + FLANN (+ mutual check) and USAC/MAGSAC.
        """
        img1_full = cv2.imread(img1_path)
        img2_full = cv2.imread(img2_path)

        # keypoint detection at reduced resolution 
        img1_small = cv2.resize(img1_full, (0, 0), fx=scale_factor, fy=scale_factor)
        img2_small = cv2.resize(img2_full, (0, 0), fx=scale_factor, fy=scale_factor)

        sift = cv2.SIFT_create()  # keep your defaults
        kp1, des1 = sift.detectAndCompute(img1_small, None)
        kp2, des2 = sift.detectAndCompute(img2_small, None)
        log.info("[✓] SIFT keypoints: %s vs %s", len(kp1), len(kp2))

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            raise RuntimeError("Not enough keypoints/descriptors to match.")

        # FLANN matching (KDTree for float)
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=128))

        # 1) A->B 2-NN ratio
        m12 = flann.knnMatch(des1, des2, k=2)
        good12 = [
            m[0] for m in m12 if len(m) == 2 and m[0].distance < 0.75 * m[1].distance
        ]

        # 2) B->A 2-NN ratio
        m21 = flann.knnMatch(des2, des1, k=2)
        good21 = [
            m[0] for m in m21 if len(m) == 2 and m[0].distance < 0.75 * m[1].distance
        ]

        # 3) Mutual check (cross-consistency)
        idx21 = {(m.trainIdx, m.queryIdx) for m in good21}
        good = [m for m in good12 if (m.queryIdx, m.trainIdx) in idx21]

        log.info("[✓] Good matches after mutual check: %s", len(good))
        if len(good) < 10:
            raise RuntimeError("Not enough good matches to estimate homography.")

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]) / scale_factor
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]) / scale_factor

        # Robust homography 
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, inliers = cv2.findHomography(
            pts2,
            pts1,
            method=method,
            ransacReprojThreshold=3.0,  # tighter than 5.0
            confidence=0.999,
            maxIters=10000,
        )
        if H is None:
            raise RuntimeError("Homography computation failed.")
        log.info(
            "Homography inliers: %s / %s",
            int(inliers.sum()) if inliers is not None else 0,
            len(good),
        )
        log.info("Homography:\n%s", H)

        # Use BORDER_CONSTANT so padding doesn’t create fake valid overlap
        img2_aligned = cv2.warpPerspective(
            img2_full,
            H,
            (img1_full.shape[1], img1_full.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        cv2.imwrite(output_path, img2_aligned)
        log.info("Aligned image saved → %s", output_path)
        return img1_full, img2_aligned, H

    @staticmethod
    def compute_overlap_area_between_reference_and_aligned(
        reference: np.ndarray, aligned_smaller: np.ndarray
    ) -> int:
        """
        Helper to compute overlap area (in pixels) between reference and warped image.
        Smaller result implies a tighter overlap region; used by hybrid selection logic.
        """
        mask_ref = (reference.sum(axis=2) > 10).astype(np.uint8)
        mask_aligned = (aligned_smaller.sum(axis=2) > 10).astype(np.uint8)
        overlap = cv2.bitwise_and(mask_ref, mask_aligned)
        return int(np.sum(overlap))
