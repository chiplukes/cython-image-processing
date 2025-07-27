"""
Main image processing module that interfaces with Cython code.
"""

import numpy as np

from . import image_filters


def create_sample_image(width: int = 512, height: int = 512) -> np.ndarray:
    """
    Create a sample RGB image for testing.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        A 3D numpy array representing an RGB image with shape (height, width, 3)
    """
    # Create a gradient pattern for demonstration
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Red channel: horizontal gradient
    for x in range(width):
        image[:, x, 0] = int(255 * x / width)

    # Green channel: vertical gradient
    for y in range(height):
        image[y, :, 1] = int(255 * y / height)

    # Blue channel: diagonal pattern
    for y in range(height):
        for x in range(width):
            image[y, x, 2] = int(255 * ((x + y) % width) / width)

    return image


def process_image(image: np.ndarray, operation: str = "blur") -> np.ndarray:
    """
    Process an RGB image using Cython-accelerated operations.

    Args:
        image: Input RGB image as numpy array with shape (height, width, 3)
        operation: Type of processing ("blur", "sharpen", "edge_detect", "brightness")

    Returns:
        Processed image as numpy array with same shape as input
    """
    if image.ndim != 3 or image.shape[2] != 3:  # noqa: PLR2004
        raise ValueError("Input must be a 3D array with shape (height, width, 3)")

    if not image.dtype == np.uint8:
        raise ValueError("Input array must have dtype uint8")

    # Call the appropriate Cython function based on operation
    if operation == "blur":
        return image_filters.gaussian_blur(image)
    elif operation == "sharpen":
        return image_filters.sharpen_filter(image)
    elif operation == "edge_detect":
        return image_filters.edge_detection(image)
    elif operation == "brightness":
        return image_filters.adjust_brightness(image, 1.2)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def demo_processing():
    """
    Demonstrate the image processing capabilities.
    """
    print("Creating sample image...")
    image = create_sample_image(256, 256)
    print(f"Created image with shape: {image.shape}, dtype: {image.dtype}")

    operations = ["blur", "sharpen", "edge_detect", "brightness"]

    for op in operations:
        print(f"Applying {op} filter...")
        processed = process_image(image, op)
        print(f"Processed image shape: {processed.shape}, dtype: {processed.dtype}")

        # Calculate some basic stats
        original_mean = np.mean(image)
        processed_mean = np.mean(processed)
        print(f"  Original mean intensity: {original_mean:.2f}")
        print(f"  Processed mean intensity: {processed_mean:.2f}")
        print()

    return True
