# change_detection/tests/stages/test_bboxes.py

import pytest
import numpy as np
from cd_pipeline.stages import AlignmentStage, TilingStage
from cd_pipeline.types import PatchRec

def test_bounding_boxes():
    alignment_stage = AlignmentStage()
    tiling_stage = TilingStage()

    # Dummy context with two images
    ctx = {
        'before_image_path': 'test_data/before.tif',
        'after_image_path': 'test_data/after.tif',
        'patches': []
    }

    # Run the alignment stage
    ctx = alignment_stage.run(ctx)
    assert 'aligned_image_path' in ctx
    assert ctx['aligned_image_path'] == 'align/aligned.tiff'
    assert len(ctx['patches']) > 0

    # Capture the bounding boxes from the patches
    for patch in ctx['patches']:
        assert isinstance(patch, PatchRec)
        assert len(patch.bbox_aligned) == 4
        assert len(patch.bbox_before) == 4
        assert len(patch.bbox_after) == 4

    # Now run the tiling stage
    ctx = tiling_stage.run(ctx)

    # Validate that tiles are created and bounding boxes are correct
    assert 'tiles' in ctx
    assert len(ctx['tiles']) > 0

    for tile in ctx['tiles']:
        assert isinstance(tile, PatchRec)
        assert len(tile.bbox_aligned) == 4
        assert len(tile.bbox_before) == 4
        assert len(tile.bbox_after) == 4

        # Bounding boxes should match expectations based on the image dimensions
        assert tile.bbox_before == tile.bbox_aligned
        assert tile.bbox_after == tile.bbox_aligned
