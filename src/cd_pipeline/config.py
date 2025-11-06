# src/cd_pipeline/config.py
from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field          # Field stays in pydantic

class Settings(BaseSettings):
    """
    Central configuration for the change-detection pipeline.

    Any field can be overridden via environment variables prefixed
    with  `CD_PIPELINE_`, e.g.

        export CD_PIPELINE_TILE_SIZE=512
        python -m cd_pipeline.cli …

    See https://docs.pydantic.dev/latest/concepts/settings/ for details.
    """

    #  tiling
    tile_size: int = Field(
        224,
        description="Edge length of each square tile (pixels).",
    )

    stride: int = Field(
        default_factory=lambda: int(224 * 0.7),
        description="Slide distance between tiles (defaults to 70 % of tile_size).",
    )

    # warp/flow filters
    laplacian_threshold_low: float = Field(10.0)
    laplacian_threshold_high: float = Field(2500.0)
    optical_flow_threshold: float = Field(4.0)

    # DINO + metrics
    dino_change_threshold: float = Field(0.6)
    min_cc_area: int = Field(300)

    ssim_threshold: float = Field(0.90)
    l1_threshold: float = Field(5.0)

    # pydantic-v2
    model_config = {"env_prefix": "cd_pipeline_"}

    # helpers
    def __init__(self, **data):
        """
        Re-compute `stride` after environment overrides so it always
        stays at 70 % of `tile_size` unless explicitly set.
        """
        super().__init__(**data)
        if "stride" not in data:  # user did not override stride explicitly
            object.__setattr__(self, "stride", int(self.tile_size * 0.7))
