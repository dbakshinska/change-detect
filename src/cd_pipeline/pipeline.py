# src/cd_pipeline/pipeline.py
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

# Shared context type alias
Context = Dict[str, Any]

class Stage(ABC):
    """Abstract base class for every pipeline stage."""

    @abstractmethod
    def run(self, ctx: Context) -> Context: ...


class Pipeline:
    """
    Minimal orchestrator: iterate over an ordered list of ``Stage``
    instances, passing a mutable *ctx* dict to each one.
    """

    def __init__(self, stages: List[Stage]) -> None:
        if not stages:
            raise ValueError("Pipeline requires at least one stage.")
        self.stages = stages

    def __call__(self, ctx: Context) -> Context:
        for stage in self.stages:
            name = stage.__class__.__name__
            logging.info("▶️  Running stage: %s", name)

            start = time.perf_counter()
            try:
                ctx = stage.run(ctx)
            except Exception:  # noqa: BLE001  (we re-raise after logging)
                logging.exception("Stage %s raised an exception", name)
                raise
            elapsed = time.perf_counter() - start

            logging.info("Finished %s in %.2f s", name, elapsed)
        return ctx
