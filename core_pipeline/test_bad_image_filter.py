#!/usr/bin/env python3
"""
Test script to compare pipeline performance on a "bad" image with and without filtering
This script demonstrates the effectiveness of preprocessing filters on low-quality images
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add python directory to path for imports
sys.path.append(str(Path(__file__).parent / "python"))

from pipeline_integration import CVPipeline


def create_bad_image_test_case(output_path: str = "temp/bad_test_image.jpg"):
    """
    Create a deliberately degraded test image from an existing good image
    This simulates common image quality issues
    """
    import cv2
    import numpy as np
    
    # Use one of the existing images as a base
    base_images = [
        "images/sample.jpg",
        "images/pexels-eberhardgross-1624496.jpg",
        "images/image.jpg"
    ]
    
    base_image = None
    for img_path in base_images:
        if os.path.exists(img_path):
            base_image = img_path
            break
    
    if not base_image:
        print("No base image found. Please ensure you have an image in the images/ directory.")
        return None
    
    # Load the base image
    img = cv2.imread(base_image)
    if img is None:
        print(f"Could not load base image: {base_image}")
        return None
    
    print(f"Creating degraded test image from: {base_image}")
    
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
    
    print(f"Bad test image created: {output_path}")
    print("Applied degradations: noise, reduced contrast, blur, darkening")
    
    return output_path


def run_pipeline_test(image_path: str, use_filter: bool = True, filter_type: str = "auto", 
                     output_dir: str = "temp/filter_test_results", verbose: bool = True):
    """
    Run the pipeline on the test image with or without filtering
    """
    pipeline = EnhancedVisionPipeline()
    
    if use_filter:
        result = pipeline.run_pipeline_with_preprocessing(
            image_path=image_path,
            filter_type=filter_type,
            conf_thresh=0.25,
            device="cpu",
            save_results=True,
            output_dir=f"{output_dir}/with_filter",
            verbose=verbose
        )
        test_type = "with_filter"
    else:
        result = pipeline.run_pipeline_without_preprocessing(
            image_path=image_path,
            conf_thresh=0.25,
            device="cpu",
            save_results=True,
            output_dir=f"{output_dir}/without_filter",
            verbose=verbose
        )
        test_type = "without_filter"
    
    return result, test_type


def compare_results(with_filter_result: dict, without_filter_result: dict, verbose: bool = True):
    """
    Compare the results from filtered vs unfiltered pipeline runs
    """
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_scenario": "bad_image_filter_comparison"
    }
    
    # Extract key metrics
    with_filter_detections = with_filter_result.get("detection", {}).get("detection_count", 0)
    without_filter_detections = without_filter_result.get("detection", {}).get("detection_count", 0)
    
    with_filter_time = with_filter_result.get("total_pipeline_time", 0)
    without_filter_time = without_filter_result.get("total_pipeline_time", 0)
    
    preprocess_time = with_filter_result.get("preprocessing_timing", {}).get("preprocess_total", 0)
    
    # Calculate improvements
    detection_improvement = with_filter_detections - without_filter_detections
    time_overhead = with_filter_time - without_filter_time
    relative_overhead = (time_overhead / without_filter_time * 100) if without_filter_time > 0 else 0
    
    comparison["results"] = {
        "with_filter": {
            "detections": with_filter_detections,
            "total_time_ms": with_filter_time,
            "preprocessing_time_ms": preprocess_time
        },
        "without_filter": {
            "detections": without_filter_detections,
            "total_time_ms": without_filter_time,
            "preprocessing_time_ms": 0
        },
        "analysis": {
            "detection_improvement": detection_improvement,
            "time_overhead_ms": time_overhead,
            "relative_overhead_percent": relative_overhead,
            "preprocessing_effectiveness": "improved" if detection_improvement > 0 else "no_improvement" if detection_improvement == 0 else "degraded"
        }
    }
    
    # Image quality assessments
    if "assessment" in with_filter_result:
        comparison["image_quality"] = {
            "original_quality_score": with_filter_result["assessment"]["quality_score"],
            "quality_issues": with_filter_result["assessment"]["quality_issues"],
            "recommended_filter": with_filter_result["assessment"]["recommended_filter"]
        }
    
    if verbose:
        print("\n" + "="*60)
        print("FILTER EFFECTIVENESS COMPARISON")
        print("="*60)
        print(f"Test Image Quality Issues: {comparison.get('image_quality', {}).get('quality_issues', 'Unknown')}")
        print(f"Recommended Filter: {comparison.get('image_quality', {}).get('recommended_filter', 'Unknown')}")
        print()
        
        print("DETECTION RESULTS:")
        print(f"  With Filter:    {with_filter_detections} objects detected")
        print(f"  Without Filter: {without_filter_detections} objects detected")
        print(f"  Improvement:    {detection_improvement:+d} objects")
        print()
        
        print("PERFORMANCE RESULTS:")
        print(f"  With Filter:    {with_filter_time:.1f} ms total")
        print(f"  Without Filter: {without_filter_time:.1f} ms total")
        print(f"  Preprocessing:  {preprocess_time:.1f} ms")
        print(f"  Time Overhead:  {time_overhead:+.1f} ms ({relative_overhead:+.1f}%)")
        print()
        
        # Verdict
        if detection_improvement > 0:
            print(f"✅ FILTERING IS EFFECTIVE!")
            print(f"   Preprocessing improved detection by {detection_improvement} object(s)")
            if relative_overhead < 20:
                print(f"   Time overhead is acceptable ({relative_overhead:.1f}%)")
            else:
                print(f"   Time overhead is significant ({relative_overhead:.1f}%)")
        elif detection_improvement == 0:
            print(f"⚪ FILTERING HAS NO IMPACT")
            print(f"   Preprocessing did not change detection results")
            print(f"   Consider this image may not benefit from preprocessing")
        else:
            print(f"❌ FILTERING DEGRADED PERFORMANCE")
            print(f"   Preprocessing reduced detection by {abs(detection_improvement)} object(s)")
            print(f"   Consider using a different filter or no preprocessing")
        
        print("="*60)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Test pipeline with bad image and compare filter effectiveness")
    parser.add_argument("--image", help="Path to test image (if not provided, creates a degraded test image)")
    parser.add_argument("--filter", default="auto", choices=["auto", "blur", "sharpen", "denoise", "clahe", "edge"],
                       help="Filter type to test")
    parser.add_argument("--output-dir", default="temp/filter_test_results", help="Output directory for results")
    parser.add_argument("--create-bad-image", action="store_true", 
                       help="Create a degraded test image from existing sample")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Determine test image
    test_image = args.image
    
    if args.create_bad_image or not test_image:
        print("Creating degraded test image...")
        test_image = create_bad_image_test_case("temp/bad_test_image.jpg")
        if not test_image:
            print("Failed to create test image")
            return 1
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return 1
    
    print(f"\nTesting pipeline on: {test_image}")
    print(f"Filter type: {args.filter}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Run pipeline without filter
    print("1. Running pipeline WITHOUT filter...")
    without_filter_result, _ = run_pipeline_test(
        test_image, use_filter=False, output_dir=args.output_dir, verbose=args.verbose
    )
    
    if "error" in without_filter_result:
        print(f"Error in unfiltered pipeline: {without_filter_result['error']}")
        return 1
    
    print("2. Running pipeline WITH filter...")
    with_filter_result, _ = run_pipeline_test(
        test_image, use_filter=True, filter_type=args.filter, 
        output_dir=args.output_dir, verbose=args.verbose
    )
    
    if "error" in with_filter_result:
        print(f"Error in filtered pipeline: {with_filter_result['error']}")
        return 1
    
    # Compare results
    comparison = compare_results(with_filter_result, without_filter_result, verbose=True)
    
    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/comparison_report.json", "w") as f:
        json.dump({
            "test_image": test_image,
            "filter_type": args.filter,
            "with_filter_result": with_filter_result,
            "without_filter_result": without_filter_result,
            "comparison": comparison
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output_dir}/comparison_report.json")
    print(f"Visual results saved to:")
    print(f"  - {args.output_dir}/with_filter/")
    print(f"  - {args.output_dir}/without_filter/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())