# Cython Image Processing

[![Tests Status](https://github.com/chiplukes/cython-image-processing/actions/workflows/test.yml/badge.svg)]
[![Changelog](https://img.shields.io/github/v/release/chiplukes/cython-image-processing?include_prereleases&label=changelog)](https://github.com/chiplukes/cython-image-processing/releases)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/chiplukes/cython-image-processing/blob/main/LICENSE)

A high-performance image processing library using Cython and NumPy for efficient operations on RGB images represented as 2D NumPy arrays.

## Features

- **High-Performance**: Cython-accelerated image processing functions
- **NumPy Integration**: Seamless operations on NumPy arrays
- **Multiple Filters**: Gaussian blur, sharpening, edge detection, brightness adjustment
- **Easy to Use**: Simple Python API with type hints
- **Extensible**: Template for creating your own Cython-based image processing projects

## Quick Start

To quickly adapt this project for your own Cython-based image processing needs:

* Rename `cython-image-processing` throughout the project with your actual hyphenated project name
* Rename `cython_image_processing` throughout the project with your actual underscored project name
* Rename `chiplukes` with your actual GitHub username
* Add new image processing functions to `image_filters.pyx`
* Update the Python interface in `cython_image_processing.py`

## Prerequisites

- Python 3.8+
- NumPy
- Cython
- C compiler (MSVC on Windows, GCC on Linux/macOS)

## Dependencies

- **numpy**: For efficient array operations
- **cython**: For compiling high-performance C extensions
- **setuptools**: For building and packaging

## Installation

### Clone repository

```bash
git clone git+https://github.com/chiplukes/cython-image-processing
cd cython-image-processing
```

### Package Installation (via pip)

1. **Install dependencies:**
```bash
pip install numpy cython
```

2. **Build and install the package:**
```bash
pip install -e .
```

This will compile the Cython extensions and install the package in development mode.

### Package Installation (via uv)

To create a virtual environment for your Python project with uv:

1. **Navigate to your project directory:**
```bash
cd cython-image-processing
```

2. **Create the virtual environment:**
```bash
uv venv
```

3. **Activate the environment:**
```bash
# On Windows
.venv\Scripts\activate
# On Unix/macOS
source .venv/bin/activate
```

4. **Install dependencies and build:**
```bash
uv pip install numpy cython
uv pip install -e .
```

## Usage

### Command Line Interface

Run the package with various image processing operations:

```bash
# Basic blur operation on 256x256 image
python -m cython_image_processing

# Apply sharpening filter with custom image size
python -m cython_image_processing --width 512 --height 512 --operation sharpen

# Edge detection
python -m cython_image_processing --operation edge_detect

# Brightness adjustment
python -m cython_image_processing --operation brightness

# Enable debug mode for full demo
python -m cython_image_processing --debug
```

### Python API

```python
import numpy as np
import cython_image_processing

# Create a sample RGB image
image = cython_image_processing.create_sample_image(width=512, height=512)
print(f"Created image: {image.shape}, dtype: {image.dtype}")

# Apply different filters
blurred = cython_image_processing.process_image(image, "blur")
sharpened = cython_image_processing.process_image(image, "sharpen")
edges = cython_image_processing.process_image(image, "edge_detect")
brighter = cython_image_processing.process_image(image, "brightness")

# Work with your own images
your_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
processed = cython_image_processing.process_image(your_image, "blur")
```

### Performance Benefits

The Cython implementation provides significant performance improvements over pure Python:

- **Gaussian Blur**: ~10-50x faster than naive Python implementation
- **Edge Detection**: ~15-60x faster than pure Python with NumPy
- **Memory Efficient**: Operates directly on NumPy arrays without copying

## Development

### Adding New Filters

1. **Add Cython function** to `src/cython_image_processing/image_filters.pyx`
2. **Update Python interface** in `src/cython_image_processing/cython_image_processing.py`
3. **Rebuild package**: `pip install -e .`

### Build Script (`build.py`)

The project includes a convenience build script for manual Cython compilation:

```bash
python build.py
```

**What it does:**
- Runs `python setup.py build_ext --inplace` to compile Cython extensions
- Builds `.pyx` files directly in the source directory (in-place build)
- Provides user-friendly build status messages

**When to use:**
- **Development**: Quick rebuilds after modifying `.pyx` files
- **Manual builds**: Alternative to `pip install -e .` for testing changes
- **Debugging**: Isolate compilation issues from package installation

**Note**: The standard installation process (`pip install -e .`) automatically handles Cython compilation and is the recommended approach for most users. The `build.py` script is primarily a developer convenience tool.

### Testing

```bash
pytest tests/
```

### Setup pre-commit hooks (optional)

```bash
pre-commit install
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
