# src/cd_pipeline/stages/patch_saver.py
"""
PatchSaver stage
================

Copies (optionally) PNG snapshots for each surviving patch, applies patch‐level
NMS, and writes a JSON manifest containing both pixel‐ and geographic‐coordinate
quads (in WGS84) plus metric area in meters² (via UTM). The manifest format
matches the single-image mode, so chip mode outputs remain consistent.

Key options:
- save_tile_pngs: bool (default False) – whether to copy p.path_t0/p.path_t1 PNGs
- tile_png_subdir: optional subfolder name to place PNGs under out_dir
"""

from __future__ import annotations

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from pyproj import CRS, Transformer
import cv2  # noqa: F401  (used indirectly by upstream types; safe to keep)

from cd_pipeline.pipeline import Stage
from cd_pipeline.types import PatchRec
from cd_pipeline.utils.tmp import tmp_dir

log = logging.getLogger(__name__)

# IoU threshold used for per-chip patch NMS
PATCH_NMS_IOU = 0.30



def highest_mode(p: PatchRec) -> int:
    """Return 3 if YOLO passed, 2 if SSIM passed, 1 if DINO passed, else 0."""
    if getattr(p, "pass_yolo", False):
        return 3
    if getattr(p, "pass_ssim", False):
        return 2
    if getattr(p, "pass_dino", False):
        return 1
    return 0


def _iou_box(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / union


def nms_patches(patches: List[PatchRec], thr: float = PATCH_NMS_IOU) -> List[PatchRec]:
    """Simple NMS by IoU on p.bbox_after (descending area)."""
    if not patches:
        return []
    boxes = np.array([p.bbox_after for p in patches], dtype=float)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        ious = np.array([_iou_box(tuple(boxes[i]), tuple(boxes[j])) for j in rest], dtype=float)
        order = rest[ious < thr]
    return [patches[i] for i in keep]


def quad_from_bbox(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """bbox -> 4x2 quad (tl, tr, br, bl) in pixel coords."""
    x1, y1, x2, y2 = bbox
    return np.array(
        [[x1, y1],
         [x2, y1],
         [x2, y2],
         [x1, y2]],
        dtype=np.float64,
    )


def warp_quad_via_hinv(quad: np.ndarray, Hinv: np.ndarray) -> np.ndarray:
    """Warp quad corners via inverse homography; returns 4x2 float pixels."""
    pts = quad + 0.5  # center convention
    homo = np.concatenate([pts, np.ones((4, 1), dtype=np.float64)], axis=1).T  # 3x4
    warped = Hinv @ homo  # 3x4
    warped /= warped[2:3, :]
    return warped[:2, :].T  # 4x2


def quad_pixels_to_native(quad_px: np.ndarray, src) -> Dict[str, Tuple[float, float]]:
    """
    Convert 4x2 pixel quad to native geographic coordinates via rasterio src.xy.
    Assumes quad order is tl, tr, br, bl.
    """
    def pc(x: float, y: float) -> Tuple[float, float]:
        # rasterio.xy uses (row, col) == (y, x)
        lon, lat = src.xy(int(round(y)), int(round(x)), offset="center")
        return float(lon), float(lat)

    tl, tr, br, bl = quad_px
    return {
        "left_top": pc(*tl),
        "right_top": pc(*tr),
        "right_bottom": pc(*br),
        "left_bottom": pc(*bl),
    }


def to_wgs84(quad: Dict[str, Tuple[float, float]], src_crs) -> Dict[str, Tuple[float, float]]:
    """Transform native coords to WGS84 (EPSG:4326)."""
    xs, ys = zip(*quad.values())
    lons, lats = rio_transform(src_crs, "EPSG:4326", xs, ys)
    return {k: (float(lo), float(la)) for k, lo, la in zip(quad.keys(), lons, lats)}


def compute_utm_from_wgs84_quad(quad_wgs84: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Pick a UTM zone from the quad centroid and transform all corners."""
    lons, lats = zip(*quad_wgs84.values())
    centroid_lon = float(np.mean(lons))
    centroid_lat = float(np.mean(lats))

    zone_number = int((centroid_lon + 180) / 6) + 1
    is_south = centroid_lat < 0
    utm_epsg = 32700 + zone_number if is_south else 32600 + zone_number
    utm_crs = CRS.from_epsg(utm_epsg)
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    out: Dict[str, Tuple[float, float]] = {}
    for k, (lon, lat) in quad_wgs84.items():
        ux, uy = transformer.transform(lon, lat)
        out[k] = (float(ux), float(uy))
    return out


def quad_area_m2_from_utm(quad_utm: Dict[str, Tuple[float, float]]) -> float:
    """Shoelace on ordered polygon: left_top, right_top, right_bottom, left_bottom."""
    poly = [
        quad_utm["left_top"],
        quad_utm["right_top"],
        quad_utm["right_bottom"],
        quad_utm["left_bottom"],
    ]
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    area = 0.0
    for i in range(4):
        area += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    return abs(area) / 2.0


# Stage


class PatchSaver(Stage):
    def __init__(
        self,
        mode: int | None,
        keep_temp: bool = False,
        patch_nms_iou: float = PATCH_NMS_IOU,
        save_tile_pngs: bool = False,          # NEW: default off
        tile_png_subdir: str | None = None,    # optional subfolder for PNGs
    ):
        self.mode = mode
        self.keep_temp = keep_temp
        self.patch_nms_iou = patch_nms_iou
        self.save_tile_pngs = save_tile_pngs
        self.tile_png_subdir = tile_png_subdir

    def run(self, ctx: Dict) -> Dict:
        patches: List[PatchRec] = ctx.get("patches", [])
        out_dir: Path = Path(ctx["out_dir"])
        work_dir: Path = out_dir / "patches"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Filter by requested minimum mode, then NMS
        selected = patches if self.mode is None else [p for p in patches if highest_mode(p) >= self.mode]
        selected = nms_patches(selected, thr=self.patch_nms_iou)

        # Datasets + homography
        ortho = ctx["orthos"]
        src_ref = rasterio.open(ortho.before_image_path)   # reference (before)
        src_given = rasterio.open(ortho.after_image_path)  # given (after)
        Hinv = ctx["H_inv"]

        # Alignment/order metadata (preserve for manifest parity)
        ai = ctx.get("alignment_info", {})
        alignment_order = {
            "original_before": ai.get("original_before"),
            "original_after": ai.get("original_after"),
            "used_swapped": ai.get("used_swapped", False),
            "reference_is": ai.get("reference_is"),
            "given_is": ai.get("given_is"),
        }

        # If saving PNGs, optionally nest them
        if self.save_tile_pngs and self.tile_png_subdir:
            png_dir = work_dir / self.tile_png_subdir
            png_dir.mkdir(parents=True, exist_ok=True)
        else:
            png_dir = work_dir

        manifest_patches: List[Dict] = []

        # Helper: parse pixel tile offset from typical patch name
        # Example: tile_15600_6864_p001_t0.png -> (y=15600, x=6864)
        def extract_tile_offset(patch_name: str) -> Tuple[int, int]:
            parts = patch_name.split("_")
            # Defensive: require at least like ['tile','y','x',...]
            try:
                return int(parts[1]), int(parts[2])
            except Exception:
                # Fallback to (0,0) if unexpected naming
                log.warning("Unexpected patch_name format for offset: %s", patch_name)
                return 0, 0

        for p in selected:
            # Optionally copy visual PNGs
            if self.save_tile_pngs:
                for src_path in (p.path_t0, p.path_t1):
                    try:
                        shutil.copy2(src_path, png_dir / Path(src_path).name)
                    except Exception as e:
                        log.warning("Skipping PNG copy %s due to: %s", src_path, e)

            mode_found = highest_mode(p)

            la1, ly1, la2, ly2 = map(int, p.bbox_aligned)
            ga1, gy1, ga2, gy2 = map(int, p.bbox_after)

            tile_y, tile_x = extract_tile_offset(p.patch_name)  # row, col offsets

            # Before (reference) quad in original BEFORE pixels
            before_aligned_quad = quad_from_bbox((la1, ly1, la2, ly2))
            before_original_quad = before_aligned_quad.copy()
            before_original_quad[:, 0] += tile_x
            before_original_quad[:, 1] += tile_y

            # After (given) quad in original AFTER pixels (via H^-1)
            after_aligned_quad = quad_from_bbox((ga1, gy1, ga2, gy2))
            after_original_quad = warp_quad_via_hinv(after_aligned_quad, Hinv)

            # Native CRS corner coords
            ref_nat = quad_pixels_to_native(before_original_quad, src_ref)
            giv_nat = quad_pixels_to_native(after_original_quad, src_given)

            # WGS84 corners
            ref_wgs = to_wgs84(ref_nat, src_ref.crs)
            giv_wgs = to_wgs84(giv_nat, src_given.crs)

            # UTM + areas
            ref_utm = compute_utm_from_wgs84_quad(ref_wgs)
            giv_utm = compute_utm_from_wgs84_quad(giv_wgs)
            area_m2_reference = quad_area_m2_from_utm(ref_utm)
            area_m2_given = quad_area_m2_from_utm(giv_utm)

            manifest_patches.append({
                "patch_name": p.patch_name,
                "bbox_aligned": [la1, ly1, la2, ly2],
                "bbox_before_original_quad": before_original_quad.tolist(),
                "bbox_after_original_quad": after_original_quad.tolist(),
                "mode_found": mode_found,
                "area_m2_reference": area_m2_reference,
                "area_m2_given": area_m2_given,
                "reference_image": ref_wgs,
                "given_image": giv_wgs,
                "reference_image_utm": ref_utm,
                "given_image_utm": giv_utm,
            })

        # Close datasets
        src_ref.close()
        src_given.close()

        # Write manifest alongside (and move everything to out_dir)
        full_manifest = {
            "alignment_order": alignment_order,
            "patches": manifest_patches,
            "pngs_saved": bool(self.save_tile_pngs),
        }

        (work_dir / "patches_meta.json").write_text(
            json.dumps(full_manifest, ensure_ascii=False, indent=2)
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        for f in work_dir.iterdir():
            shutil.move(str(f), out_dir / f.name)
        if not self.keep_temp:
            work_dir.rmdir()

        # Return updated context
        ctx["patches"] = selected
        return ctx
