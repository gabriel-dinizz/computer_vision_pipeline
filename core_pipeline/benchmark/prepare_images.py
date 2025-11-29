#!/usr/bin/env python3
"""
Image Preparation Script for Multi-Resolution Benchmarking
Resizes test images to standard resolutions for comprehensive benchmarking
"""

from PIL import Image
import os
from pathlib import Path
from typing import List, Tuple


class ImagePreparation:
    """Prepare test images at multiple resolutions for benchmarking"""

    # Standard test resolutions (width, height)
    RESOLUTIONS = {
        '480p': (640, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
    }

    def __init__(self, source_dir: str = "../images/dataset",
                 output_dir: str = "../images/benchmark_dataset"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create resolution subdirectories
        for res_name in self.RESOLUTIONS.keys():
            (self.output_dir / res_name).mkdir(exist_ok=True)

    def resize_image(self, image_path: Path, resolution: Tuple[int, int],
                     output_path: Path) -> bool:
        """Resize a single image to specified resolution"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (handles PNG with alpha, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize using high-quality Lanczos resampling
                resized = img.resize(resolution, Image.Resampling.LANCZOS)

                # Save as JPEG with high quality
                resized.save(output_path, 'JPEG', quality=95)

                return True
        except Exception as e:
            print(f"Error resizing {image_path}: {e}")
            return False

    def prepare_all_resolutions(self, source_images: List[str] = None) -> dict:
        """
        Prepare all test images at all resolutions

        Args:
            source_images: List of image filenames to process (without path)
                          If None, processes all images in source directory

        Returns:
            Dictionary with preparation results
        """
        print("=" * 70)
        print("Image Preparation for Multi-Resolution Benchmark")
        print("=" * 70)

        # Find source images
        if source_images is None:
            source_files = list(self.source_dir.glob('*.jpg')) + \
                          list(self.source_dir.glob('*.jpeg')) + \
                          list(self.source_dir.glob('*.png'))
        else:
            source_files = [self.source_dir / img for img in source_images]

        if not source_files:
            print(f"No images found in {self.source_dir}")
            return {}

        print(f"\nFound {len(source_files)} source images")
        print(f"Target resolutions: {', '.join(self.RESOLUTIONS.keys())}\n")

        results = {
            'processed': {},
            'total_images': 0,
            'successful': 0,
            'failed': 0
        }

        for source_path in source_files:
            image_name = source_path.stem  # Filename without extension
            print(f"Processing: {source_path.name}")

            results['processed'][image_name] = {}

            for res_name, (width, height) in self.RESOLUTIONS.items():
                output_filename = f"{image_name}_{res_name}.jpg"
                output_path = self.output_dir / res_name / output_filename

                print(f"  Creating {res_name} ({width}x{height})... ", end='')

                if self.resize_image(source_path, (width, height), output_path):
                    file_size_kb = output_path.stat().st_size / 1024
                    print(f"OK ({file_size_kb:.1f} KB)")

                    results['processed'][image_name][res_name] = str(output_path)
                    results['successful'] += 1
                else:
                    print("FAILED")
                    results['failed'] += 1

                results['total_images'] += 1

            print()

        # Generate summary
        print("=" * 70)
        print("Preparation Summary")
        print("=" * 70)
        print(f"Total images processed: {results['total_images']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"\nOutput directory: {self.output_dir}")

        # List all generated images
        print("\nGenerated test images:")
        for res_name in self.RESOLUTIONS.keys():
            res_dir = self.output_dir / res_name
            images = list(res_dir.glob('*.jpg'))
            print(f"  {res_name}: {len(images)} images")
            for img in sorted(images):
                print(f"    - {img.name}")

        return results

    def verify_images(self) -> bool:
        """Verify all generated images are valid"""
        print("\nVerifying generated images...")
        all_valid = True

        for res_name in self.RESOLUTIONS.keys():
            res_dir = self.output_dir / res_name
            images = list(res_dir.glob('*.jpg'))

            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        expected_width, expected_height = self.RESOLUTIONS[res_name]

                        if width != expected_width or height != expected_height:
                            print(f"  ERROR: {img_path.name} has wrong size: "
                                  f"{width}x{height} (expected {expected_width}x{expected_height})")
                            all_valid = False
                except Exception as e:
                    print(f"  ERROR: Cannot open {img_path.name}: {e}")
                    all_valid = False

        if all_valid:
            print("  All images verified successfully!")

        return all_valid


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare test images at multiple resolutions for benchmarking"
    )
    parser.add_argument(
        '--source-dir',
        default='../images/dataset',
        help='Directory containing source images'
    )
    parser.add_argument(
        '--output-dir',
        default='../images/benchmark_dataset',
        help='Directory for resized images'
    )
    parser.add_argument(
        '--images',
        nargs='+',
        help='Specific images to process (e.g., sample.jpg image.jpg)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify generated images after creation'
    )

    args = parser.parse_args()

    # Create image preparation instance
    prep = ImagePreparation(args.source_dir, args.output_dir)

    # Prepare images
    results = prep.prepare_all_resolutions(args.images)

    # Verify if requested
    if args.verify and results['successful'] > 0:
        prep.verify_images()

    print("\n✅ Image preparation complete!")


if __name__ == "__main__":
    main()
