# Change Detection Pipeline

This repository implements a change detection pipeline utilizing various image processing techniques, including alignment, tiling, deep learning-based change detection (DINO), and object detection (YOLO). It allows users to detect differences between two images, utilizing self-supervised learning models and traditional image processing methods.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Stage | Purpose |
|-------|---------|
| **Alignment** | SIFT + RANSAC homography aligns the two orthos. |
| **Tiling** | Cuts the aligned pair into overlapping 224 × 224 tiles (70 % stride). |
| **Warp / Flow filters** | Discards tiles with poor alignment (low texture, large optical flow). |
| **DINO change metric** | ViT (DINOv2) highlights semantic change regions → connected components → patch crops. |
| **SSIM + L1 filter** | Removes patches that still look too similar. |
| **YOLOv8** | Detects objects; **area‑ranked NMS at IoU 0.40** keeps the largest box for each real object. |
| **Patch Saver** | Copies PNG pairs and writes `patches_meta.json` (global & local bboxes, stage info). |

---

## Installation

> Python **≥ 3.10** required. Using [uv](https://github.com/astral-sh/uv) for speed is recommended but not mandatory.

```bash
git clone <repository‑url>
cd change_detection

# 1) create and activate a venv
uv venv
source .venv/bin/activate        # or your shell equivalent

# 2) install the package (editable) + dev tools
uv pip install -e .[dev]

```
## Usage

To run the Change Detection Pipeline, execute it using `uv`.

### Running the Pipeline with UV

You can trigger the pipeline with the following command:

```bash
uv run python -m cd_pipeline.cli \
 <before_image_path>
 <after_image_path>
 <output_directory>
[--mode <mode>] [--keep-temp]
```
| Argument / flag  | Description                                                                                                                                                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `<before_image>` | Path to the “before / T₀” orthophoto (GeoTIFF, JPG, PNG, …).                                                                                                                                                                                                       |
| `<after_image>`  | Path to the “after / T₁” orthophoto.                                                                                                                                                                                                                               |
| `<output_dir>`   | Directory where all results (patches, CSV, JSON, temps) will be written.                                                                                                                                                                                           |
| `--mode`         | *Optional* patch‑retention level:<br>• **omit**  — keep **all** patches (QA / stats).<br>• **1** — patches that **passed DINO**.<br>• **2** — patches that **passed DINO + SSIM/L1**.<br>• **3** — patches that **passed DINO + SSIM/L1 + YOLO** (production set). |
| `--keep-temp`    | Keep intermediate folders (useful for debugging).                                                                                                                                                                                                                  |


**Example Command:**
```bash
uv run python -m cd_pipeline.cli \
    data/before.tif data/after.tif results/ \
    --mode 3 --keep-temp
```

## Pipeline Overview
```text
Alignment  ──►  Tiling  ──►  Warp/Flow filter  ──►  DINO
                                                     │
                                                     ▼
                                             SSIM / L1 filter
                                                     │
                                                     ▼
                                                YOLOv8
                                                     │
                                                     ▼
                                                Patch Saver

```
1. **Alignment**: Aligns the two images for accurate comparison using SIFT.
2. **Tiling**: Divides aligned images into overlapping tiles for localized analysis.
3. **Warping and Optical Flow Analysis**: Analyzes differences using warping techniques.
4. **Change Detection with DINO**: Uses DINO model to identify significant changes and generate patches.
5. **SSIM and L1 Filtering**: Filters patches based on structural similarity and L1 norm metrics.
6. **YOLO Object Detection**: Conducts object detection on patches to identify significant changes.
7. **Patch Saving**: Saves patches and metadata for further analysis.

## Contributing

Contributions are welcome! Please feel free to submit a pull request with improvements or bug fixes. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
