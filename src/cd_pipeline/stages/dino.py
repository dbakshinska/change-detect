# src/cd_pipeline/stages/dino.py
"""
DINO change-detection stage
===========================

Runs DINOv2 embeddings on each *clean* tile pair and extracts connected
components whose *change score* exceeds the threshold.  Patches are saved
to ``tmp_dir("dino")`` and appended to ``ctx["patches"]`` as `PatchRec`
objects.

Context on entry
----------------
* ctx["clean_tiles"]      : list[str]      (tile indices from WarpFilter)
* ctx["tiles_directory"]  : pathlib.Path   (where tile_<idx>_t{0,1}.jpg live)
* ctx["tile_offsets"]     : pandas.DataFrame (id → x_offset, y_offset)
  – optional; if present we compute global coords

Context on exit
---------------
* ctx["patches"] : list[PatchRec]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import morphology
from skimage.metrics import structural_similarity as ssim
from transformers import AutoImageProcessor, Dinov2Model
from tqdm.auto import tqdm

from cd_pipeline.pipeline import Stage
from cd_pipeline.types import PatchRec
from cd_pipeline.utils.tmp import tmp_dir

log = logging.getLogger(__name__)


class DINOStage(Stage):
    """Extract semantic-change patches with DINOv2."""

    def __init__(
        self,
        change_threshold: float = 0.6,
        min_cc_area: int = 300,
        black_frac_max: float = 0.05,  # drop if >5% pixels are black
        black_val_thr: int = 5,        # channel sum < 5 → consider pixel black
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base", use_fast=True
        )
        self.model = (
            Dinov2Model.from_pretrained("facebook/dinov2-base")
            .to(self.device)
            .eval()
        )
        self.change_threshold = change_threshold
        self.min_cc_area = min_cc_area
        self.black_frac_max = black_frac_max
        self.black_val_thr = black_val_thr

    def run(self, ctx: Dict) -> Dict:
        clean_tiles: List[str] = ctx.get("clean_tiles", [])
        tile_dir: Path = ctx["tiles_directory"]

        out_dir: Path = ctx["out_dir"]
        dino_dir: Path = out_dir / "dino"
        dino_dir.mkdir(parents=True, exist_ok=True)
        patch_records: List[PatchRec] = []

        log.info("🔍 DINO processing %d clean tile pairs", len(clean_tiles))
        for tile_id in tqdm(clean_tiles, desc="DINO tiles", unit="tile"):
            t0_path = tile_dir / f"{tile_id}_t0.jpg"
            t1_path = tile_dir / f"{tile_id}_t1.jpg"
            out_dir = dino_dir / tile_id
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                img1, img2 = self._load_images(t0_path, t1_path)
                change_map = self._compute_change_map(img1, img2)
                mask = self._cleanup_mask(change_map)
                patch_records += self._extract_patches(
                    mask, img1, img2, tile_id, out_dir, ctx
                )
            except Exception as exc:
                log.warning("DINO failed on %s: %s", tile_id, exc)

        # save metadata CSV
        df = pd.DataFrame([p.__dict__ for p in patch_records])
        df.to_csv(dino_dir / "dino_patch_coords.csv", index=False)
        log.info("Patch metadata → %s", dino_dir / "dino_patch_coords.csv")

        ctx["patches"] = patch_records
        return ctx

    @staticmethod
    def _load_images(t0: Path, t1: Path) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.array(Image.open(t0).convert("RGB")),
            np.array(Image.open(t1).convert("RGB")),
        )

    def _compute_change_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        f1 = self._get_features(img1)
        f2 = self._get_features(img2)
        f1n = torch.nn.functional.normalize(f1, dim=1)
        f2n = torch.nn.functional.normalize(f2, dim=1)

        change_vec = 1 - torch.sum(f1n * f2n, dim=1)
        grid = int(np.sqrt(change_vec.shape[0]))
        change_map = (
            change_vec[: grid**2]
            .view(grid, grid)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return cv2.resize(
            change_map,
            (img1.shape[1], img1.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    def _cleanup_mask(self, change_map: np.ndarray) -> np.ndarray:
        binary = (change_map > self.change_threshold).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = morphology.remove_small_objects(
            closed.astype(bool), min_size=100
        ).astype(np.uint8) * 255
        return cleaned

    def _extract_patches(
        self,
        mask: np.ndarray,
        img1: np.ndarray,
        img2: np.ndarray,
        tile_id: str,
        out_dir: Path,
        ctx: Dict,
    ) -> List[PatchRec]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        records: List[PatchRec] = []
        offset_df = ctx.get("tile_offsets")

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_cc_area:
                continue

            patch1 = img1[y : y + h, x : x + w]
            patch2 = img2[y : y + h, x : x + w]

            # black-border filter 
            # Treat very dark pixels as black to account for JPEG noise
            black1 = (patch1.sum(axis=2) < self.black_val_thr)
            black2 = (patch2.sum(axis=2) < self.black_val_thr)
            frac_black = max(black1.mean(), black2.mean())
            if frac_black > self.black_frac_max:
                continue

            # SSIM to weed out trivial differences
            gray1 = cv2.cvtColor(patch1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(patch2, cv2.COLOR_RGB2GRAY)
            score = ssim(gray1, gray2, data_range=255)
            if score >= 0.97:
                continue

            name = f"{tile_id}_p{len(records):03d}"
            path_t0 = out_dir / f"{name}_t0.png"
            path_t1 = out_dir / f"{name}_t1.png"
            cv2.imwrite(str(path_t0), cv2.cvtColor(patch1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(path_t1), cv2.cvtColor(patch2, cv2.COLOR_RGB2BGR))

            # global coords (AFTER space)
            if offset_df is not None and tile_id in offset_df.index:
                xo, yo = offset_df.loc[tile_id]
                xg, yg = int(x + xo), int(y + yo)
            else:
                xg, yg = x, y

            records.append(
                PatchRec(
                    patch_name=name,
                    path_t0=path_t0,
                    path_t1=path_t1,
                    tile_id=tile_id,
                    bbox_aligned=(x, y, x + w, y + h),
                    bbox_before=(x, y, x + w, y + h),
                    bbox_after=(xg, yg, xg + w, yg + h),
                    pass_dino=True,
                    dino_change=float(mask[y + h // 2, x + w // 2]),
                    ssim=score,
                )
            )
        return records

    def _get_features(self, img: np.ndarray) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # drop CLS token then squeeze
        return outputs.last_hidden_state[:, 1:, :].squeeze(0)

