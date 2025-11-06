# change_detection/tests/stages/test_alignment.py

import pytest
import numpy as np
from cd_pipeline.stages.alignment import AlignmentStage
from cd_pipeline.types import PatchRec

def test_alignment_stage():
    stage = AlignmentStage()

    # Dummy context
    ctx = {
        'before_image_path': 'test_data/before.tif',
        'after_image_path': 'test_data/after.tif',
        'patches': []
    }

    # Run the alignment stage
    ctx = stage.run(ctx)

    # Check the resultant aligned image path
    assert 'aligned_image_path' in ctx
    assert ctx['aligned_image_path'] == 'align/aligned.tiff'

    # Assert homography matrix shape
    assert ctx['H'].shape == (3, 3)

    # Check that patches list is updated correctly
    assert len(ctx['patches']) > 0
    for patch in ctx['patches']:
        assert isinstance(patch, PatchRec)
        assert len(patch.bbox_aligned) == 4
        assert len(patch.bbox_before) == 4
        assert len(patch.bbox_after) == 4


# change_detection/tests/utils/test_has_black_pixels.py

import cv2
import numpy as np
import pytest
from cd_pipeline.utils.image_ops import has_black_pixels

def create_test_image(percentage_black: float) -> np.ndarray:
    """Create a test image with a known percentage of black pixels."""
    size = (100, 100)  # 100x100 pixel image
    total_pixels = size[0] * size[1]
    num_black_pixels = int(total_pixels * percentage_black)

    image = np.ones((size[0], size[1]), dtype=np.uint8) * 255  # white image
    for _ in range(num_black_pixels):
        x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
        image[x, y] = 0  # set pixel to black

    return image

def test_has_black_pixels_with_black_image():
    """Test the has_black_pixels function with an image with black pixels."""
    image = create_test_image(0.05)  # 5% black
    assert has_black_pixels(image) is True

def test_has_black_pixels_with_white_image():
    """Test the has_black_pixels function with an entirely white image."""
    image = create_test_image(0.0)  # 0% black
    assert has_black_pixels(image) is False

def test_has_black_pixels_with_full_black_image():
    """Test the has_black_pixels function with an entirely black image."""
    image = np.zeros((100, 100), dtype=np.uint8)  # all black
    assert has_black_pixels(image) is True


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
