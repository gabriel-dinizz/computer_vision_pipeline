#!/usr/bin/env python3
"""
Create a severely degraded image to better demonstrate filter effectiveness
"""

import cv2
import numpy as np
import argparse

def create_severely_degraded_image(input_path: str, output_path: str):
    """
    Create a severely degraded image that will benefit from filtering
    """
    # Load the base image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")

    print(f"Creating severely degraded image from: {input_path}")
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")

    # Apply severe degradations

    # 1. Heavy noise
    noise = np.random.normal(0, 40, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Severe contrast reduction
    low_contrast = cv2.convertScaleAbs(noisy_img, alpha=0.4, beta=50)

    # 3. Heavy blur
    blurred = cv2.GaussianBlur(low_contrast, (15, 15), 3.0)

    # 4. Significant darkening
    darkened = cv2.convertScaleAbs(blurred, alpha=0.6, beta=-40)

    # 5. Add motion blur simulation
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    motion_blurred = cv2.filter2D(darkened, -1, kernel)

    # 6. Final noise layer
    final_noise = np.random.normal(0, 15, motion_blurred.shape).astype(np.int16)
    final_degraded = np.clip(motion_blurred.astype(np.int16) + final_noise, 0, 255).astype(np.uint8)

    # Save the severely degraded image
    cv2.imwrite(output_path, final_degraded)

    print(f"✓ Severely degraded image created: {output_path}")
    print("Applied degradations:")
    print("  - Heavy Gaussian noise (σ=40)")
    print("  - Severe contrast reduction (α=0.4)")
    print("  - Heavy Gaussian blur (kernel=15x15, σ=3.0)")
    print("  - Significant darkening")
    print("  - Motion blur simulation")
    print("  - Additional noise layer")

    # Calculate quality metrics
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_degraded = cv2.cvtColor(final_degraded, cv2.COLOR_BGR2GRAY)

    # Calculate variance of Laplacian (blur metric)
    laplacian_orig = cv2.Laplacian(gray_orig, cv2.CV_64F)
    variance_orig = laplacian_orig.var()

    laplacian_degraded = cv2.Laplacian(gray_degraded, cv2.CV_64F)
    variance_degraded = laplacian_degraded.var()

    print(f"\nQuality Assessment:")
    print(f"  Original blur variance: {variance_orig:.1f}")
    print(f"  Degraded blur variance: {variance_degraded:.1f} ({variance_degraded/variance_orig*100:.1f}% of original)")

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create severely degraded test image")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output path for degraded image")

    args = parser.parse_args()

    try:
        create_severely_degraded_image(args.input, args.output)
        print(f"\n✅ Severely degraded image ready for filter effectiveness testing!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())