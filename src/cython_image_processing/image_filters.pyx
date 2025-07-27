# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False

"""
High-performance image processing functions implemented in Cython.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp, sqrt

# Define numpy array types
ctypedef cnp.uint8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_blur(cnp.ndarray[DTYPE_t, ndim=3] image):
    """
    Apply Gaussian blur to an RGB image.

    Args:
        image: Input RGB image array with shape (height, width, 3)

    Returns:
        Blurred image array with same shape and dtype
    """
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef int channels = image.shape[2]

    # Create output array
    cdef cnp.ndarray[DTYPE_t, ndim=3] output = np.zeros_like(image)

    # Simple 3x3 Gaussian kernel (approximation)
    cdef double kernel[3][3]
    kernel[0][0] = 1.0/16; kernel[0][1] = 2.0/16; kernel[0][2] = 1.0/16
    kernel[1][0] = 2.0/16; kernel[1][1] = 4.0/16; kernel[1][2] = 2.0/16
    kernel[2][0] = 1.0/16; kernel[2][1] = 2.0/16; kernel[2][2] = 1.0/16

    cdef int y, x, c, ky, kx
    cdef double sum_val
    cdef int ny, nx

    # Apply convolution
    for y in range(1, height-1):
        for x in range(1, width-1):
            for c in range(channels):
                sum_val = 0.0
                for ky in range(3):
                    for kx in range(3):
                        ny = y + ky - 1
                        nx = x + kx - 1
                        sum_val += image[ny, nx, c] * kernel[ky][kx]
                output[y, x, c] = <DTYPE_t>sum_val

    # Copy borders
    output[0, :, :] = image[0, :, :]
    output[height-1, :, :] = image[height-1, :, :]
    output[:, 0, :] = image[:, 0, :]
    output[:, width-1, :] = image[:, width-1, :]

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def sharpen_filter(cnp.ndarray[DTYPE_t, ndim=3] image):
    """
    Apply a sharpening filter to an RGB image.

    Args:
        image: Input RGB image array with shape (height, width, 3)

    Returns:
        Sharpened image array with same shape and dtype
    """
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef int channels = image.shape[2]

    cdef cnp.ndarray[DTYPE_t, ndim=3] output = np.zeros_like(image)

    # Sharpening kernel
    cdef double kernel[3][3]
    kernel[0][0] = 0; kernel[0][1] = -1; kernel[0][2] = 0
    kernel[1][0] = -1; kernel[1][1] = 5; kernel[1][2] = -1
    kernel[2][0] = 0; kernel[2][1] = -1; kernel[2][2] = 0

    cdef int y, x, c, ky, kx
    cdef double sum_val
    cdef int ny, nx

    for y in range(1, height-1):
        for x in range(1, width-1):
            for c in range(channels):
                sum_val = 0.0
                for ky in range(3):
                    for kx in range(3):
                        ny = y + ky - 1
                        nx = x + kx - 1
                        sum_val += image[ny, nx, c] * kernel[ky][kx]

                # Clamp to valid range
                if sum_val < 0:
                    sum_val = 0
                elif sum_val > 255:
                    sum_val = 255

                output[y, x, c] = <DTYPE_t>sum_val

    # Copy borders
    output[0, :, :] = image[0, :, :]
    output[height-1, :, :] = image[height-1, :, :]
    output[:, 0, :] = image[:, 0, :]
    output[:, width-1, :] = image[:, width-1, :]

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def edge_detection(cnp.ndarray[DTYPE_t, ndim=3] image):
    """
    Apply edge detection (Sobel operator) to an RGB image.

    Args:
        image: Input RGB image array with shape (height, width, 3)

    Returns:
        Edge-detected image array with same shape and dtype
    """
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef int channels = image.shape[2]

    cdef cnp.ndarray[DTYPE_t, ndim=3] output = np.zeros_like(image)

    # Sobel X kernel
    cdef double sobel_x[3][3]
    sobel_x[0][0] = -1; sobel_x[0][1] = 0; sobel_x[0][2] = 1
    sobel_x[1][0] = -2; sobel_x[1][1] = 0; sobel_x[1][2] = 2
    sobel_x[2][0] = -1; sobel_x[2][1] = 0; sobel_x[2][2] = 1

    # Sobel Y kernel
    cdef double sobel_y[3][3]
    sobel_y[0][0] = -1; sobel_y[0][1] = -2; sobel_y[0][2] = -1
    sobel_y[1][0] = 0; sobel_y[1][1] = 0; sobel_y[1][2] = 0
    sobel_y[2][0] = 1; sobel_y[2][1] = 2; sobel_y[2][2] = 1

    cdef int y, x, c, ky, kx
    cdef double gx, gy, magnitude
    cdef int ny, nx

    for y in range(1, height-1):
        for x in range(1, width-1):
            for c in range(channels):
                gx = 0.0
                gy = 0.0

                for ky in range(3):
                    for kx in range(3):
                        ny = y + ky - 1
                        nx = x + kx - 1
                        gx += image[ny, nx, c] * sobel_x[ky][kx]
                        gy += image[ny, nx, c] * sobel_y[ky][kx]

                magnitude = sqrt(gx*gx + gy*gy)
                if magnitude > 255:
                    magnitude = 255

                output[y, x, c] = <DTYPE_t>magnitude

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def adjust_brightness(cnp.ndarray[DTYPE_t, ndim=3] image, double factor):
    """
    Adjust brightness of an RGB image.

    Args:
        image: Input RGB image array with shape (height, width, 3)
        factor: Brightness multiplication factor (1.0 = no change, >1.0 = brighter)

    Returns:
        Brightness-adjusted image array with same shape and dtype
    """
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef int channels = image.shape[2]

    cdef cnp.ndarray[DTYPE_t, ndim=3] output = np.zeros_like(image)

    cdef int y, x, c
    cdef double new_val

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                new_val = image[y, x, c] * factor

                # Clamp to valid range
                if new_val > 255:
                    new_val = 255
                elif new_val < 0:
                    new_val = 0

                output[y, x, c] = <DTYPE_t>new_val

    return output
