# src/cd_pipeline/cli.py
import os
import random
import json
import logging
import math
import subprocess
import tempfile
from pathlib import Path

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
# Set YOLO config directory to avoid permission warnings
os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics")

import typer
import numpy as np
import rasterio
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)

cv2.setRNGSeed(0)
np.random.seed(0)
random.seed(0)

try:
    import torch
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
except Exception:
    pass

from cd_pipeline.pipeline import Pipeline
from cd_pipeline.stages import (
    Alignment,
    Tiling,
    WarpFilter,
    DINOStage,
    SSIML1Filter,
    YOLOFilter,
    PatchSaver,
)
from cd_pipeline.types import OrthoPair
from cd_pipeline.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = typer.Typer(add_completion=False)
settings = Settings()

DEFAULT_MAX_SIDE_PX = 30000  # planning only

# helpers

def _safe_unlink(p: Path) -> None:
    try:
        Path(p).unlink(missing_ok=True)
    except Exception:
        pass

def _write_minimal_manifest_from_patches(patches, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "patches_meta.json"
    rows = []
    for p in patches or []:
        rows.append({
            "tile_id": getattr(p, "tile_id", getattr(p, "patch_name", "unknown")),
            "patch_name": getattr(p, "patch_name", None),
            "t0": str(getattr(p, "path_t0", "")),
            "t1": str(getattr(p, "path_t1", "")),
            "bbox_aligned": list(map(int, getattr(p, "bbox_aligned", (0, 0, 0, 0)))),
            "bbox_before":  list(map(int, getattr(p, "bbox_before",  (0, 0, 0, 0)))),
            "bbox_after":   list(map(int, getattr(p, "bbox_after",   (0, 0, 0, 0)))),
            "pass_dino": bool(getattr(p, "pass_dino", False)),
            "pass_ssim": bool(getattr(p, "pass_ssim", False)),
            "pass_yolo": bool(getattr(p, "pass_yolo", False)),
        })
    with open(manifest_path, "w") as f:
        json.dump(rows, f, indent=2)
    logging.info("Fallback wrote manifest with %d patches → %s", len(rows), manifest_path)


def _read_manifest_json(path: Path) -> list:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _aggregate_chip_manifests(chip_subdirs: list[Path], out_dir: Path) -> None:
    """
    Accept per-chip manifests that are either:
      - dict style: {"alignment_order": {...}, "patches": [...]}
      - legacy list style: [ {...}, {...}, ... ]
    and emit a single dict identical to single-mode:
      {"alignment_order": {...}, "patches": [...]}
    """
    all_patches: list = []
    alignment_order: dict | None = None

    for subdir in chip_subdirs:
        mpath = subdir / "patches_meta.json"
        try:
            with open(mpath, "r") as f:
                data = json.load(f)
        except Exception as e:
            logging.warning("Failed to read %s: %s", mpath, e)
            continue

        if isinstance(data, dict):
            if alignment_order is None and "alignment_order" in data:
                alignment_order = data["alignment_order"]
            rows = data.get("patches", [])
        elif isinstance(data, list):
            rows = data
        else:
            rows = []

        if rows:
            all_patches.extend(rows)
        else:
            logging.warning("No patches found in %s; skipping.", subdir.name)

    out_payload = {
        "alignment_order": alignment_order or {},
        "patches": all_patches,
    }
    agg_path = Path(out_dir) / "patches_meta.json"
    with open(agg_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    logging.info("Wrote aggregated manifest with %d patches → %s", len(all_patches), agg_path)


def _cleanup_tmpdirs(tmpdirs: list) -> None:
    for td in tmpdirs:
        try:
            td.cleanup()
        except Exception:
            pass


def gdal_translate_projwin(src: Path, dst: Path, ulx: float, uly: float, lrx: float, lry: float) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal_translate", "-q",
        "-projwin", str(ulx), str(uly), str(lrx), str(lry),
        str(src), str(dst),
    ]
    subprocess.run(cmd, check=True)


def overlap_bounds(before: Path, after: Path):
    with rasterio.open(before) as A, rasterio.open(after) as B:
        if A.crs != B.crs:
            raise RuntimeError(f"CRS mismatch: {A.crs} vs {B.crs}")
        left   = max(A.bounds.left,   B.bounds.left)
        right  = min(A.bounds.right,  B.bounds.right)
        bottom = max(A.bounds.bottom, B.bounds.bottom)
        top    = min(A.bounds.top,    B.bounds.top)
        if left >= right or bottom >= top:
            raise RuntimeError("No overlap between before/after")
        return left, bottom, right, top


def _pixel_sizes(before: Path, after: Path):
    with rasterio.open(before) as A, rasterio.open(after) as B:
        ax = abs(A.transform.a); ay = abs(A.transform.e)
        bx = abs(B.transform.a); by = abs(B.transform.e)
    return min(ax, bx), min(ay, by)


def choose_grid_for_overlap(before: Path, after: Path,
                            left: float, bottom: float, right: float, top: float,
                            max_side_px: int):
    px_w, px_h = _pixel_sizes(before, after)
    overlap_w = right - left
    overlap_h = top - bottom
    overlap_px_w = overlap_w / px_w
    overlap_px_h = overlap_h / px_h
    safe_side = max_side_px * 0.98
    nx = max(1, math.ceil(overlap_px_w / safe_side))
    ny = max(1, math.ceil(overlap_px_h / safe_side))
    return nx, ny


def compute_overlap_area(ctx: dict) -> int:
    ref = ctx["original_image"]
    warped = cv2.imread(str(ctx["aligned_image_path"]), cv2.IMREAD_COLOR)
    if warped is None:
        return 0
    mask_ref    = (ref.sum(axis=2) > 10).astype("uint8")
    mask_warped = (warped.sum(axis=2) > 10).astype("uint8")
    overlap     = cv2.bitwise_and(mask_ref, mask_warped)
    return int(overlap.sum())


def build_tail_pipeline(mode: int | None, keep_temp: bool,
                        lap_low: float, lap_high: float | None, flow_thresh: float,
                        save_tile_pngs: bool) -> Pipeline:
    stride = int(settings.tile_size * 0.7)
    # PatchSaver may or may not support save_tile_pngs; be defensive.
    try:
        ps = PatchSaver(mode=mode, keep_temp=keep_temp, save_tile_pngs=save_tile_pngs)
    except TypeError:
        ps = PatchSaver(mode=mode, keep_temp=keep_temp)
        if not save_tile_pngs:
            logging.info("PatchSaver w/out save_tile_pngs arg detected; PNG saving behavior unchanged in this version.")
    return Pipeline([
        Tiling(tile_size=settings.tile_size, stride=stride),
        WarpFilter(laplacian_threshold_low=lap_low,
                   laplacian_threshold_high=lap_high,
                   optical_flow_threshold=flow_thresh),
        DINOStage(),
        SSIML1Filter(),
        YOLOFilter(),
        ps,
    ])


# deterministic alignment

ORIG_SEED  = 1337
SWAP_SEED  = 7331
CANON_SEED = 2024

def _reset_seeds(seed: int) -> None:
    cv2.setRNGSeed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
    except Exception:
        pass


def _align_once(before: Path, after: Path, out_dir: Path, seed: int):
    _reset_seeds(seed)
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)

    align = Alignment()
    ctx = {"orthos": OrthoPair(before, after), "out_dir": out_dir}
    ctx = align.run(ctx)
    area = compute_overlap_area(ctx)
    return ctx, area


def _disk_only_ctx(ctx: dict, out_dir: Path) -> dict:
    return {
        "orthos": ctx["orthos"],
        "aligned_image_path": ctx["aligned_image_path"],
        "H_inv": ctx.get("H_inv", np.eye(3, dtype=float)),
        "out_dir": out_dir,
        "alignment_info": ctx.get("alignment_info", {}),
    }


def _fingerprint_image(path: Path) -> str:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return "unreadable"
        h, w = img.shape[:2]
        arr = img.view(np.uint8).ravel()
        chk = int(arr.sum() % 65536)
        return f"{w}x{h}/chk={chk}"
    except Exception as e:
        return f"error:{e}"


def pick_alignment_and_normalize(before: Path, after: Path, out_dir: Path) -> dict:
    ctx_a = area_a = None
    try:
        ctx_a, area_a = _align_once(before, after, out_dir, seed=ORIG_SEED)
    except Exception as e:
        logging.warning("Alignment A (before→after) failed: %s", e)

    ctx_b = area_b = None
    try:
        ctx_b, area_b = _align_once(after, before, out_dir, seed=SWAP_SEED)
    except Exception as e:
        logging.warning("Alignment B (after→before) failed: %s", e)

    if ctx_a is None and ctx_b is None:
        logging.warning("Both alignment directions failed — falling back to identity.")
        return {
            "orthos": OrthoPair(before, after),
            "aligned_image_path": after,
            "H_inv": np.eye(3, dtype=float),
            "out_dir": out_dir,
            "alignment_info": {
                "original_before": before.name,
                "original_after":  after.name,
                "used_swapped":    False,
                "reference_is":    Path(before).name,
                "given_is":        Path(after).name,
            },
        }

    pick_a = (ctx_b is None) or (ctx_a is not None and area_a <= area_b)
    if pick_a:
        ref_p, mov_p = before, after
        used_swapped = False
    else:
        ref_p, mov_p = after, before
        used_swapped = True
    logging.info("Selected %s order (areas: A=%s, B=%s)", "original" if not used_swapped else "swapped", area_a, area_b)

    ctx_final, _ = _align_once(ref_p, mov_p, out_dir, seed=CANON_SEED)
    ctx_final["alignment_info"] = {
        "original_before": before.name,
        "original_after":  after.name,
        "used_swapped":    used_swapped,
        "reference_is":    ctx_final["orthos"].before_image_path.name,
        "given_is":        ctx_final["orthos"].after_image_path.name,
    }

    ref_pth = ctx_final["orthos"].before_image_path
    ali_pth = ctx_final["aligned_image_path"]
    logging.info("Chosen pair: ref=%s (%s)  aligned=%s (%s)",
                 ref_pth.name, _fingerprint_image(ref_pth),
                 Path(ali_pth).name, _fingerprint_image(ali_pth))

    return _disk_only_ctx(ctx_final, out_dir)


def chip_signal_fraction(tif: Path, sample: int = 512) -> float:
    with rasterio.open(tif) as ds:
        indexes = list(ds.indexes)
        try:
            if len(indexes) > 1 and ds.colorinterp and ds.colorinterp[-1] == rasterio.enums.ColorInterp.alpha:
                indexes = indexes[:-1]
        except Exception:
            pass
        h, w = ds.height, ds.width
        scale = max(h, w) / sample if max(h, w) > sample else 1.0
        out_h = max(1, int(round(h / scale)))
        out_w = max(1, int(round(w / scale)))
        arr = ds.read(indexes=indexes[:1], out_shape=(1, out_h, out_w),
                      resampling=rasterio.enums.Resampling.nearest)
        arr = arr.astype(np.uint8)
        nz = int(np.count_nonzero(arr))
        tot = int(arr.size)
        return (nz / tot) if tot else 0.0


def run_full_pair(before_chip: Path, after_chip: Path, out_dir: Path,
                  mode: int | None, keep_temp: bool,
                  lap_low: float, lap_high: float | None, flow_thresh: float,
                  save_tile_pngs: bool,
                  locked_order: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    tail = build_tail_pipeline(mode, keep_temp, lap_low, lap_high, flow_thresh, save_tile_pngs)

    if locked_order in ("orig", "swap"):
        ref_p, mov_p = (before_chip, after_chip) if locked_order == "orig" else (after_chip, before_chip)
        ctx_final, _ = _align_once(ref_p, mov_p, out_dir, seed=CANON_SEED)
        ctx_final["alignment_info"] = {
            "original_before": before_chip.name,
            "original_after":  after_chip.name,
            "used_swapped":    (locked_order == "swap"),
            "reference_is":    ctx_final["orthos"].before_image_path.name,
            "given_is":        ctx_final["orthos"].after_image_path.name,
        }
        chosen_ctx = _disk_only_ctx(ctx_final, out_dir)
    else:
        chosen_ctx = pick_alignment_and_normalize(before_chip, after_chip, out_dir)

    final_ctx = tail(chosen_ctx)
    return final_ctx


# CLI 

@app.command()
def run(
    before: Path = typer.Argument(...),
    after:  Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    mode: int | None = typer.Option(None, "--mode", "-m"),
    keep_temp: bool = typer.Option(False, "--keep-temp"),
    force_split: bool = typer.Option(False, help="Force splitting the overlap into chips"),
    max_side_px: int = typer.Option(DEFAULT_MAX_SIDE_PX, help="Max chip side in pixels (per raster)"),
    min_chip_signal: float = typer.Option(0.002, help="Skip chips when BOTH rasters have < this nonzero fraction"),
    chip_pad_pct: float = typer.Option(0.10, help="Pad each chip window by this fraction of its width/height, clamped to the overlap"),
    lock_orientation: bool = typer.Option(True, help="Pick best alignment orientation once and reuse for every chip"),
    save_chip_rasters: bool = typer.Option(False, help="Keep per-chip GeoTIFFs under out_dir/chips"),
    save_probe_rasters: bool = typer.Option(False, help="Keep orientation-probe TIFFs under out_dir/orientation_probe"),
    save_tile_pngs: bool = typer.Option(False, help="Save per-patch PNG tiles (default: do not save)"),
    # WarpFilter tuning
    lap_low: float = typer.Option(10.0, "--lap-low"),
    lap_high: float | None = typer.Option(None, "--lap-high"),
    flow_thresh: float = typer.Option(4.0, "--flow"),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    have_overlap = True
    try:
        left, bottom, right, top = overlap_bounds(before, after)
    except Exception as e:
        logging.warning("Overlap computation failed (%s) — proceeding without chip split.", e)
        have_overlap = False

    if have_overlap:
        overlap_w = right - left
        overlap_h = top - bottom
        nx, ny = choose_grid_for_overlap(before, after, left, bottom, right, top, max_side_px)
        planned_chips = nx * ny
        logging.info(
            "Overlap extent: W=%.2f, H=%.2f; planned %d chips (%dx%d grid, max_side_px=%d)",
            overlap_w, overlap_h, planned_chips, nx, ny, max_side_px,
        )
        chip_needed = force_split or (planned_chips > 1)
    else:
        chip_needed = False

    locked_order: str | None = None
    lock_start_idx: int | None = None
    chip_windows: list[tuple[int, int, float, float, float, float]] = []
    _tmp_dirs_to_cleanup = []

    if chip_needed:
        for r in range(ny):
            for c in range(nx):
                x0 = left  + overlap_w * (c    / nx)
                x1 = left  + overlap_w * ((c+1) / nx)
                y1 = top   - overlap_h * (r    / ny)
                y0 = top   - overlap_h * ((r+1) / ny)
                if chip_pad_pct > 0:
                    pad_x = (x1 - x0) * chip_pad_pct
                    pad_y = (y1 - y0) * chip_pad_pct
                    x0 = max(left,  x0 - pad_x)
                    x1 = min(right, x1 + pad_x)
                    y0 = max(bottom, y0 - pad_y)
                    y1 = min(top,    y1 + pad_y)
                chip_windows.append((r, c, x0, y0, x1, y1))

    # Orientation probe: lock on first viable chip; do not revisit earlier chips.
    if chip_needed and lock_orientation:
        if save_probe_rasters:
            probe_root = out_dir / "orientation_probe"
            probe_root.mkdir(parents=True, exist_ok=True)
        else:
            tmp = tempfile.TemporaryDirectory(dir=out_dir, prefix="probe_")
            _tmp_dirs_to_cleanup.append(tmp)
            probe_root = Path(tmp.name)

        for idx, (r, c, x0, y0, x1, y1) in enumerate(chip_windows):
            tag = f"r{r}c{c}"
            probe_dir = probe_root / tag
            probe_dir.mkdir(parents=True, exist_ok=True)
            chip_b = probe_dir / f"before_{tag}.tif"
            chip_a = probe_dir / f"after_{tag}.tif"
            gdal_translate_projwin(before, chip_b, x0, y1, x1, y0)
            gdal_translate_projwin(after,  chip_a, x0, y1, x1, y0)

            sb = chip_signal_fraction(chip_b)
            sa = chip_signal_fraction(chip_a)
            if min(sb, sa) < min_chip_signal:
                logging.info("↷ Orientation probe skip %s (signal %.4f / %.4f)", tag, sb, sa)
                if not save_probe_rasters:
                    _safe_unlink(chip_b)
                    _safe_unlink(chip_a)
                continue

            probe_ctx = pick_alignment_and_normalize(chip_b, chip_a, probe_dir)
            used_swapped = bool(probe_ctx.get("alignment_info", {}).get("used_swapped", False))
            locked_order = "swap" if used_swapped else "orig"
            lock_start_idx = idx
            logging.info("Locking chip orientation to: %s (from %s)", locked_order, tag)

            # Probe files no longer needed once orientation is locked
            if not save_probe_rasters:
                _safe_unlink(chip_b)
                _safe_unlink(chip_a)
            break

        if locked_order is None:
            logging.warning("No viable chip found for orientation probe — proceeding without lock.")

    if chip_needed:
        if save_chip_rasters:
            chips_dir = out_dir / "chips"
            chips_dir.mkdir(parents=True, exist_ok=True)
        else:
            tmp = tempfile.TemporaryDirectory(dir=out_dir, prefix="chips_")
            _tmp_dirs_to_cleanup.append(tmp)
            chips_dir = Path(tmp.name)

        start_idx = lock_start_idx if (lock_start_idx is not None) else 0
        windows_to_process = chip_windows[start_idx:]

        manifests_subdirs: list[Path] = []
        for (r, c, x0, y0, x1, y1) in windows_to_process:
            tag = f"r{r}c{c}"
            subdir = out_dir / tag
            subdir.mkdir(parents=True, exist_ok=True)

            chip_b = chips_dir / f"before_{tag}.tif"
            chip_a = chips_dir / f"after_{tag}.tif"

            gdal_translate_projwin(before, chip_b, x0, y1, x1, y0)
            gdal_translate_projwin(after,  chip_a, x0, y1, x1, y0)

            sb = chip_signal_fraction(chip_b)
            sa = chip_signal_fraction(chip_a)
            if min(sb, sa) < min_chip_signal:
                logging.info("↷ Skipping %s (one side empty: signal %.4f / %.4f)", tag, sb, sa)
                if not save_chip_rasters:
                    _safe_unlink(chip_b)
                    _safe_unlink(chip_a)
                continue

            logging.info("→ Running pipeline for %s", tag)
            final_ctx = run_full_pair(
                chip_b, chip_a, subdir, mode, keep_temp,
                lap_low, lap_high, flow_thresh,
                save_tile_pngs=save_tile_pngs,
                locked_order=locked_order
            )

            # chip rasters no longer needed after this chip is processed
            if not save_chip_rasters:
                _safe_unlink(chip_b)
                _safe_unlink(chip_a)

            manifest_path = subdir / "patches_meta.json"
            if not manifest_path.exists():
                patches = final_ctx.get("patches", [])
                _write_minimal_manifest_from_patches(patches, subdir)

            if manifest_path.exists():
                manifests_subdirs.append(subdir)

        _aggregate_chip_manifests(manifests_subdirs, out_dir)
        _cleanup_tmpdirs(_tmp_dirs_to_cleanup)  # cleanup temp probe/chip dirs
        typer.secho(f"\n Chip mode complete — results in {out_dir}", fg=typer.colors.GREEN)
        return

    # ---- Normal full-image path (no splitting) ----
    logging.info("Running full-image pipeline (no splitting)")

    chosen_ctx = pick_alignment_and_normalize(before, after, out_dir)
    tail = build_tail_pipeline(mode, keep_temp, lap_low, lap_high, flow_thresh, save_tile_pngs)
    final_ctx = tail(chosen_ctx)

    _cleanup_tmpdirs(_tmp_dirs_to_cleanup)  # harmless if empty

    if mode is None:
        patches = final_ctx.get("patches", [])
        highest = 3 if any(getattr(p, "pass_yolo", False) for p in patches) \
            else 2 if any(getattr(p, "pass_ssim", False) for p in patches) \
            else 1 if any(getattr(p, "pass_dino", False) for p in patches) \
            else 0
        typer.secho(f"\n  Highest stage reached: {highest}", fg=typer.colors.BLUE)

    typer.secho(f"\n Pipeline finished — outputs in {out_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
