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
