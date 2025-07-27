import argparse
import importlib.metadata

import cython_image_processing

if __name__ == "__main__":
    print("Cython Image Processing Package")
    print(f"{__file__}:__main__")

    parser = argparse.ArgumentParser(description="Cython-accelerated image processing demo")

    # Optional argument flag which defaults to False
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug output")

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity (-v, -vv, etc)")

    # Image size arguments
    parser.add_argument("--width", type=int, default=256, help="Image width (default: 256)")
    parser.add_argument("--height", type=int, default=256, help="Image height (default: 256)")

    # Processing operation
    parser.add_argument(
        "--operation",
        choices=["blur", "sharpen", "edge_detect", "brightness"],
        default="blur",
        help="Image processing operation to perform",
    )

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=importlib.metadata.version("cython_image_processing")),
    )

    args = parser.parse_args()

    print(f"Creating {args.width}x{args.height} sample image...")
    image = cython_image_processing.create_sample_image(args.width, args.height)

    print(f"Applying {args.operation} filter...")
    processed = cython_image_processing.process_image(image, args.operation)

    print(f"Original image - Shape: {image.shape}, Mean intensity: {image.mean():.2f}")
    print(f"Processed image - Shape: {processed.shape}, Mean intensity: {processed.mean():.2f}")

    if args.debug:
        print("Debug: Running full demo...")
        cython_image_processing.demo_processing()
