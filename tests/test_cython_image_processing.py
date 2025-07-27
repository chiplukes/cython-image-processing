import numpy as np
import pytest

import cython_image_processing

MAX_PIXEL_VALUE = 255


def test_create_sample_image():
    """Test sample image creation."""
    image = cython_image_processing.create_sample_image(100, 100)

    assert image.shape == (100, 100, 3)
    assert image.dtype == np.uint8
    assert np.all(image >= 0) and np.all(image <= MAX_PIXEL_VALUE)


def test_create_sample_image_custom_size():
    """Test sample image creation with custom dimensions."""
    width, height = 256, 128
    image = cython_image_processing.create_sample_image(width, height)

    assert image.shape == (height, width, 3)
    assert image.dtype == np.uint8


def test_process_image_blur():
    """Test blur filter operation."""
    image = cython_image_processing.create_sample_image(50, 50)
    processed = cython_image_processing.process_image(image, "blur")

    assert processed.shape == image.shape
    assert processed.dtype == image.dtype


def test_process_image_sharpen():
    """Test sharpen filter operation."""
    image = cython_image_processing.create_sample_image(50, 50)
    processed = cython_image_processing.process_image(image, "sharpen")

    assert processed.shape == image.shape
    assert processed.dtype == image.dtype


def test_process_image_edge_detect():
    """Test edge detection operation."""
    image = cython_image_processing.create_sample_image(50, 50)
    processed = cython_image_processing.process_image(image, "edge_detect")

    assert processed.shape == image.shape
    assert processed.dtype == image.dtype


def test_process_image_brightness():
    """Test brightness adjustment operation."""
    image = cython_image_processing.create_sample_image(50, 50)
    processed = cython_image_processing.process_image(image, "brightness")

    assert processed.shape == image.shape
    assert processed.dtype == image.dtype


def test_process_image_invalid_operation():
    """Test that invalid operations raise ValueError."""
    image = cython_image_processing.create_sample_image(50, 50)

    with pytest.raises(ValueError, match="Unknown operation"):
        cython_image_processing.process_image(image, "invalid_op")


def test_process_image_invalid_shape():
    """Test that invalid image shapes raise ValueError."""
    # 2D array instead of 3D
    invalid_image = np.zeros((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="Input must be a 3D array"):
        cython_image_processing.process_image(invalid_image, "blur")


def test_process_image_invalid_channels():
    """Test that wrong number of channels raises ValueError."""
    # 4 channels instead of 3
    invalid_image = np.zeros((50, 50, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="Input must be a 3D array"):
        cython_image_processing.process_image(invalid_image, "blur")


def test_process_image_invalid_dtype():
    """Test that wrong dtype raises ValueError."""
    # float64 instead of uint8
    invalid_image = np.zeros((50, 50, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="Input array must have dtype uint8"):
        cython_image_processing.process_image(invalid_image, "blur")


def test_demo_processing():
    """Test the demo processing function."""
    result = cython_image_processing.demo_processing()
    assert result is True
