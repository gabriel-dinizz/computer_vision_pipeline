#!/usr/bin/env python3
"""
Filter Effectiveness Test - Compare YOLO detection with and without preprocessing
Demonstrates the value of image preprocessing for object detection on degraded images
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add python directory to path for imports
sys.path.append(str(Path(__file__).parent / "python"))

from pipeline_integration import CVPipeline
from yolo_detector import YOLODetector


def create_degraded_image(base_image_path: str, output_path: str = "temp/degraded_test.jpg"):
    """
    Create a deliberately degraded test image to demonstrate filter effectiveness
    """
    # Load the base image
    img = cv2.imread(base_image_path)
    if img is None:
        raise ValueError(f"Could not load base image: {base_image_path}")

    print(f"Creating degraded test image from: {base_image_path}")

    # Apply multiple degradations to create a "bad" image
    # 1. Add noise
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Reduce contrast
    low_contrast = cv2.convertScaleAbs(noisy_img, alpha=0.6, beta=30)

    # 3. Add blur
    blurred = cv2.GaussianBlur(low_contrast, (7, 7), 2.0)

    # 4. Darken the image
    darkened = cv2.convertScaleAbs(blurred, alpha=0.8, beta=-20)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the degraded image
    cv2.imwrite(output_path, darkened)

    print(f"✓ Degraded test image created: {output_path}")
    print("  Applied: noise, reduced contrast, blur, darkening")

    return output_path


def run_detection_without_filter(image_path: str, output_dir: str = "temp/no_filter"):
    """
    Run YOLO detection directly on the image without any preprocessing
    """
    print("🔍 Running detection WITHOUT preprocessing filter...")

    os.makedirs(output_dir, exist_ok=True)

    detector = YOLODetector(weights="yolov5su.pt", device="cpu", conf_thresh=0.25)

    start_time = time.time()
    result = detector.detect_objects(
        image_path,
        save_results=True,
        output_dir=output_dir
    )
    total_time = time.time() - start_time

    if "error" in result:
        raise RuntimeError(f"Detection failed: {result['error']}")

    return {
        "detections": result["detection_count"],
        "total_time_ms": total_time * 1000,
        "detection_details": result["detections"],
        "output_dir": output_dir
    }


def run_detection_with_filter(image_path: str, filter_type: str = "auto", output_dir: str = "temp/with_filter"):
    """
    Run full pipeline: preprocessing + YOLO detection
    """
    print(f"🔄 Running detection WITH {filter_type} preprocessing filter...")

    os.makedirs(output_dir, exist_ok=True)

    pipeline = CVPipeline()

    result = pipeline.run_full_pipeline(
        image_path,
        filter_type=filter_type,
        confidence=0.25,
        device="cpu",
        save_results=True,
        output_dir=output_dir
    )

    if "error" in result:
        raise RuntimeError(f"Pipeline failed: {result['error']}")

    return {
        "detections": result["detection"]["detection_count"],
        "total_time_ms": result["timing"]["total"],
        "preprocessing_time_ms": result["timing"]["preprocessing"],
        "detection_time_ms": result["timing"]["detection"],
        "detection_details": result["detection"]["detections"],
        "filter_used": filter_type,
        "output_dir": output_dir
    }


def compare_and_analyze(without_filter_result: dict, with_filter_result: dict, test_image: str):
    """
    Compare results and generate analysis report
    """
    print("\n" + "="*70)
    print("🔬 FILTER EFFECTIVENESS ANALYSIS")
    print("="*70)

    # Extract metrics
    detections_without = without_filter_result["detections"]
    detections_with = with_filter_result["detections"]
    time_without = without_filter_result["total_time_ms"]
    time_with = with_filter_result["total_time_ms"]
    preprocessing_time = with_filter_result["preprocessing_time_ms"]

    # Calculate confidence improvements
    def get_avg_confidence(detection_details):
        if not detection_details or len(detection_details) == 0:
            return 0.0
        return sum(d['confidence'] for d in detection_details) / len(detection_details)

    avg_conf_without = get_avg_confidence(without_filter_result["detection_details"])
    avg_conf_with = get_avg_confidence(with_filter_result["detection_details"])
    confidence_improvement = avg_conf_with - avg_conf_without

    # Calculate improvements
    detection_improvement = detections_with - detections_without
    time_overhead = time_with - time_without
    relative_overhead = (time_overhead / time_without * 100) if time_without > 0 else 0

    # Print analysis
    print(f"📸 Test Image: {test_image}")
    print(f"🎯 Filter Used: {with_filter_result['filter_used']}")
    print()

    print("DETECTION RESULTS:")
    print(f"  ❌ Without Filter: {detections_without} objects detected")
    print(f"  ✅ With Filter:    {detections_with} objects detected")
    print(f"  📊 Improvement:    {detection_improvement:+d} objects")
    print()

    print("CONFIDENCE ANALYSIS:")
    print(f"  ❌ Without Filter: {avg_conf_without:.1%} average confidence")
    print(f"  ✅ With Filter:    {avg_conf_with:.1%} average confidence")
    print(f"  📊 Improvement:    {confidence_improvement:+.1%} confidence boost")
    print()

    print("PERFORMANCE RESULTS:")
    print(f"  ⚡ Without Filter: {time_without:.1f} ms total")
    print(f"  🔄 With Filter:    {time_with:.1f} ms total")
    print(f"     └ Preprocessing: {preprocessing_time:.1f} ms")
    print(f"     └ Detection:     {with_filter_result['detection_time_ms']:.1f} ms")
    print(f"  ⏱️  Time Overhead:  {time_overhead:+.1f} ms ({relative_overhead:+.1f}%)")
    print()

    # Effectiveness verdict
    if detection_improvement > 0 or confidence_improvement > 0.05:  # 5% confidence boost is significant
        print("🎉 FILTERING IS EFFECTIVE!")
        if detection_improvement > 0:
            print(f"   ✓ Preprocessing improved detection by {detection_improvement} object(s)")
        if confidence_improvement > 0.05:
            print(f"   ✓ Preprocessing boosted confidence by {confidence_improvement:.1%}")
        if relative_overhead < 30:
            print(f"   ✓ Time overhead is acceptable ({relative_overhead:.1f}%)")
        else:
            print(f"   ⚠️  Time overhead is significant ({relative_overhead:.1f}%)")
        effectiveness = "EFFECTIVE"
    elif detection_improvement == 0 and abs(confidence_improvement) < 0.05:
        print("⚪ FILTERING HAS NO IMPACT")
        print("   ℹ️  Preprocessing did not change detection results or confidence")
        print("   💡 This image may not benefit from preprocessing")
        effectiveness = "NEUTRAL"
    else:
        print("❌ FILTERING DEGRADED PERFORMANCE")
        if detection_improvement < 0:
            print(f"   ⚠️  Preprocessing reduced detection by {abs(detection_improvement)} object(s)")
        if confidence_improvement < -0.05:
            print(f"   ⚠️  Preprocessing reduced confidence by {abs(confidence_improvement):.1%}")
        print("   💡 Consider using a different filter or no preprocessing")
        effectiveness = "DETRIMENTAL"

    print("="*70)

    # Return summary for saving
    return {
        "test_image": test_image,
        "filter_type": with_filter_result['filter_used'],
        "results": {
            "without_filter": without_filter_result,
            "with_filter": with_filter_result
        },
        "analysis": {
            "detection_improvement": detection_improvement,
            "confidence_improvement": confidence_improvement,
            "avg_confidence_without": avg_conf_without,
            "avg_confidence_with": avg_conf_with,
            "time_overhead_ms": time_overhead,
            "relative_overhead_percent": relative_overhead,
            "effectiveness": effectiveness
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def main():
    parser = argparse.ArgumentParser(description="Test filter effectiveness for object detection")
    parser.add_argument("--image", help="Path to test image (if not provided, uses sample image)")
    parser.add_argument("--create-degraded", action="store_true",
                       help="Create a degraded version of the image for testing")
    parser.add_argument("--filter", default="auto",
                       choices=["auto", "blur", "sharpen", "denoise", "clahe", "edge"],
                       help="Filter type to test")
    parser.add_argument("--output-dir", default="temp/filter_effectiveness_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Determine test image
    test_image = args.image or "images/sample.jpg"

    if args.create_degraded:
        print("Creating degraded test image for demonstration...")
        test_image = create_degraded_image(test_image, "temp/degraded_test.jpg")

    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Available images:")
        for img in Path("images").glob("*.jpg"):
            print(f"  - {img}")
        return 1

    print(f"\n🚀 Starting Filter Effectiveness Test")
    print(f"📸 Test Image: {test_image}")
    print(f"🎯 Filter Type: {args.filter}")
    print()

    try:
        # Run detection without filter
        without_filter_result = run_detection_without_filter(
            test_image,
            f"{args.output_dir}/without_filter"
        )

        # Run detection with filter
        with_filter_result = run_detection_with_filter(
            test_image,
            args.filter,
            f"{args.output_dir}/with_filter"
        )

        # Compare and analyze results
        analysis = compare_and_analyze(without_filter_result, with_filter_result, test_image)

        # Save detailed results
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = f"{args.output_dir}/effectiveness_analysis.json"
        with open(results_file, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"\n📄 Detailed results saved to: {results_file}")
        print(f"🖼️  Visual results available in:")
        print(f"   - {args.output_dir}/without_filter/")
        print(f"   - {args.output_dir}/with_filter/")

        return 0

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())