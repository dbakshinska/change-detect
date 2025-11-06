# src/cd_pipeline/stages/yolo_filter.py
"""
YOLO filtering stage
====================

1. Runs YOLOv8 on every patch that passed SSIM/L1.
2. Converts YOLO boxes to *global* coordinates (relative to the full orthophoto).
3. Applies pure‑NumPy Non‑Maximum Suppression (NMS) that ranks by **box area**
   (largest wins) and merges boxes whose IoU ≥ NMS_IOU_THR.
4. **Annotates** each patch with `pass_yolo=True/False` and stores detections,
   but does **not** prune the list; final filtering happens in PatchSaver.

Context on entry
----------------
ctx["patches"] : list[PatchRec]  (each has pass_dino, pass_ssim flags)

Context on exit
---------------
ctx["patches"] : same list, each PatchRec now also has:
    - pass_yolo: bool
    - yolo_boxes: list[tuple[int,int,int,int]]  (local coords)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

from cd_pipeline.pipeline import Stage
from cd_pipeline.types import PatchRec

log = logging.getLogger(__name__)

# IoU threshold for box‑level NMS
NMS_IOU_THR = 0.30


def nms_numpy(boxes: np.ndarray, iou_thr: float = NMS_IOU_THR) -> List[int]:
    """Pure‑NumPy NMS that keeps the largest‑area boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]  # largest first

    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_thr]

    return keep


class YOLOFilter(Stage):
    """
    Run YOLO on BOTH t0 and t1 patches.
    Annotate each patch with pass_yolo and its detections,
    but do not remove any patches here.
    """

    def __init__(self, weights: str = "yolov8m.pt", conf: float = 0.005):
        self.model = YOLO(weights)
        self.conf = conf

    def run(self, ctx: Dict) -> Dict:
        patches: List[PatchRec] = ctx.get("patches", [])

        all_global_boxes: List[List[int]] = []
        box_patch_ids: List[int] = []

        # 1) Run YOLO and flag each patch
        for idx, p in enumerate(patches):
            # Only consider patches that passed SSIM/L1
            if not getattr(p, "pass_ssim", False):
                p.pass_yolo = False
                p.yolo_boxes = []
                continue

            # Inference on AFTER and BEFORE images
            dets_t1 = self._detect(Path(p.path_t1))
            dets_t0 = self._detect(Path(p.path_t0))

            # Annotate
            p.yolo_boxes = dets_t1 + dets_t0
            p.pass_yolo = len(p.yolo_boxes) > 0

            if not p.pass_yolo:
                continue

            # Convert local detections to global coords
            px, py, _, _ = p.bbox_after
            for (x1, y1, x2, y2) in p.yolo_boxes:
                all_global_boxes.append([x1 + px, y1 + py, x2 + px, y2 + py])
                box_patch_ids.append(idx)

        # 2) Box‑level NMS (for logging/demo; pruning deferred to PatchSaver)
        boxes_arr = np.asarray(all_global_boxes, dtype=float)
        keep_box_ids = nms_numpy(boxes_arr, iou_thr=NMS_IOU_THR)
        kept_patches = {box_patch_ids[i] for i in keep_box_ids}

        log.info(
            "YOLOFilter: %d patches had detections, %d total boxes → %d after NMS → %d patches",
            sum(1 for p in patches if p.pass_yolo),
            len(all_global_boxes),
            len(keep_box_ids),
            len(kept_patches),
        )

        # Leave ctx["patches"] intact for PatchSaver to handle mode filtering
        return ctx

    def _detect(self, img_path: Path) -> List[Tuple[int, int, int, int]]:
        """
        Run YOLOv8 on img_path and return list of (x1,y1,x2,y2) in LOCAL patch coords.
        """
        try:
            results = self.model(img_path, conf=self.conf, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            return [tuple(b) for b in boxes]
        except Exception as exc:
            log.warning("YOLO inference failed on %s: %s", img_path, exc)
            return []
