#!/usr/bin/env python3
"""
Build script for Cython Image Processing package.
This script compiles the Cython extensions.
"""

import os
import subprocess
import sys
from pathlib import Path


def build_extensions():
    """Build the Cython extensions."""
    print("Building Cython extensions...")

    # Change to the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    # Run the build command
    build_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]

    print(f"Running: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, check=False)  # noqa: S603

    if result.returncode == 0:
        print("✅ Cython extensions built successfully!")
        print("You can now import and use the package:")
        print("  import cython_image_processing")
    else:
        print("❌ Build failed!")
        sys.exit(1)
        sys.exit(1)


if __name__ == "__main__":
    build_extensions()
