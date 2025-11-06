# src/cd_pipeline/__init__.py
"""
Top-level package API.

* Core objects: Pipeline, Settings, PatchRec, OrthoPair
* Convenient re-exports for stage classes
"""

from .pipeline import Pipeline
from .config import Settings
from .types import PatchRec, OrthoPair

# re-export stage classes for one-line imports
from .stages import (
    Alignment,
    Tiling,
    WarpFilter,
    DINOStage,
    SSIML1Filter,
    YOLOFilter,
    PatchSaver,
)

__all__ = [
    # core
    "Pipeline",
    "Settings",
    "PatchRec",
    "OrthoPair",
    # stages
    "Alignment",
    "Tiling",
    "WarpFilter",
    "DINOStage",
    "SSIML1Filter",
    "YOLOFilter",
    "PatchSaver",
]
