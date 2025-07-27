"""
Cython Image Processing Package

A high-performance image processing library using Cython and NumPy.
Provides efficient operations on RGB images represented as 2D NumPy arrays.
"""

from .cython_image_processing import create_sample_image, process_image

__version__ = "0.0.1"
__all__ = ["create_sample_image", "process_image"]
