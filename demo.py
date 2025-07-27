#!/usr/bin/env python3
"""
Demo script for Cython Image Processing package.

This script demonstrates the image processing capabilities with timing comparisons.
"""

import sys
import time

import numpy as np

try:
    import cython_image_processing
except ImportError:
    print("❌ cython_image_processing not found!")
    print("Please install the package first:")
    print("  pip install -e .")
    sys.exit(1)


def benchmark_operation(image, operation, iterations=10):
    """Benchmark an image processing operation."""
    times = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        result = cython_image_processing.process_image(image, operation)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return result, avg_time, std_time


def main():
    """Run the complete demo."""
    print("🚀 Cython Image Processing Demo")
    print("=" * 50)

    # Create test images of different sizes
    sizes = [(128, 128), (256, 256), (512, 512)]
    operations = ["blur", "sharpen", "edge_detect", "brightness"]

    for width, height in sizes:
        print(f"\n📐 Testing with {width}x{height} image:")
        print("-" * 30)

        # Create sample image
        print("Creating sample image...")
        start_time = time.perf_counter()
        image = cython_image_processing.create_sample_image(width, height)
        creation_time = time.perf_counter() - start_time

        print(f"  ✅ Created in {creation_time * 1000:.2f} ms")
        print(f"  📊 Shape: {image.shape}, Size: {image.nbytes / 1024:.1f} KB")
        print(f"  📈 Value range: [{image.min()}, {image.max()}]")

        # Test each operation
        for operation in operations:
            print(f"\n🔧 Testing {operation} operation:")

            try:
                result, avg_time, std_time = benchmark_operation(image, operation, iterations=5)

                print(f"  ⏱️  Average time: {avg_time * 1000:.2f} ± {std_time * 1000:.2f} ms")
                print(f"  📊 Output shape: {result.shape}")
                print(f"  📈 Output range: [{result.min()}, {result.max()}]")

                # Calculate throughput
                pixels_per_sec = (width * height) / avg_time / 1e6
                print(f"  🚄 Throughput: {pixels_per_sec:.1f} Mpixels/sec")

            except Exception as e:
                print(f"  ❌ Error: {e}")

    print("\n" + "=" * 50)
    print("✨ Demo completed successfully!")
    print("\nNext steps:")
    print("• Try running: python -m cython_image_processing --help")
    print("• Check out the API in your Python scripts")
    print("• Modify image_filters.pyx to add new operations")


if __name__ == "__main__":
    main()
