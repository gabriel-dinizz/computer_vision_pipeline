#!/usr/bin/env python3
"""
Quick performance test for individual filters
Tests single filter across thread counts to measure optimization impact
"""

import subprocess
import time
import statistics
import os
import re
from pathlib import Path

# Set environment for controlled benchmarking
os.environ['OMP_PROC_BIND'] = 'true'
os.environ['OMP_PLACES'] = 'cores'


def run_single_test(preprocess_bin, image_path, filter_type, thread_count, iterations=10):
    """Run single filter test and return mean time"""
    output_path = f"../../temp/test_{filter_type}.jpg"
    os.environ['OMP_NUM_THREADS'] = str(thread_count)

    times = []

    # Warmup
    for _ in range(3):
        subprocess.run([preprocess_bin, image_path, output_path, filter_type],
                      capture_output=True, text=True, timeout=30)

    # Actual tests
    for _ in range(iterations):
        try:
            result = subprocess.run(
                [preprocess_bin, image_path, output_path, filter_type],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                match = re.search(r'Total processing time:\s+([\d.]+)\s+ms', result.stdout)
                if match:
                    times.append(float(match.group(1)))
        except Exception as e:
            print(f"Error: {e}")
            continue

    if times:
        return {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'samples': len(times)
        }
    return None


def test_filter(filter_name, preprocess_bin="../../bin/preprocess_optimized",
                image_path="../../images/sample.jpg"):
    """Test single filter across thread counts"""

    print(f"\n{'='*60}")
    print(f"Testing {filter_name.upper()} filter")
    print(f"{'='*60}")

    thread_counts = [1, 2, 4, 8]
    results = {}

    for tc in thread_counts:
        print(f"  🧵 {tc} threads: ", end="", flush=True)
        result = run_single_test(preprocess_bin, image_path, filter_name, tc)

        if result:
            results[tc] = result
            print(f"{result['mean']:.2f}ms (±{result['std']:.2f}ms)")
        else:
            print("FAILED")

    # Calculate speedups
    if 1 in results:
        baseline = results[1]['mean']
        print(f"\n  Speedup Analysis:")
        print(f"  {'Threads':<10} {'Time (ms)':<12} {'Speedup':<10} {'Efficiency':<12}")
        print(f"  {'-'*50}")

        for tc in thread_counts:
            if tc in results:
                speedup = baseline / results[tc]['mean']
                efficiency = speedup / tc
                print(f"  {tc:<10} {results[tc]['mean']:<12.2f} {speedup:<10.2f}x {efficiency:<12.1%}")

    return results


def compare_tests(filter_name, baseline_results, new_results):
    """Compare two test runs"""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {filter_name.upper()}")
    print(f"{'='*60}")
    print(f"{'Threads':<10} {'Before':<12} {'After':<12} {'Change':<12} {'Status'}")
    print(f"{'-'*60}")

    for tc in [1, 2, 4, 8]:
        if tc in baseline_results and tc in new_results:
            before = baseline_results[tc]['mean']
            after = new_results[tc]['mean']
            change = ((after - before) / before) * 100

            if change < -5:
                status = "✅ FASTER"
            elif change > 5:
                status = "❌ SLOWER"
            else:
                status = "➖ SIMILAR"

            print(f"{tc:<10} {before:<12.2f} {after:<12.2f} {change:+11.1f}% {status}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick filter performance test")
    parser.add_argument("--filter", default="sharpen",
                       choices=['blur', 'sharpen', 'denoise', 'clahe', 'edge'],
                       help="Filter to test")
    parser.add_argument("--preprocess-bin", default="../../bin/preprocess_optimized",
                       help="Preprocessing binary path")
    parser.add_argument("--image", default="../../images/sample.jpg",
                       help="Test image path")

    args = parser.parse_args()

    # Ensure binary exists
    if not Path(args.preprocess_bin).exists():
        print(f"❌ Binary not found: {args.preprocess_bin}")
        print("Run 'make all' to build the binary first")
        exit(1)

    # Run test
    results = test_filter(args.filter, args.preprocess_bin, args.image)

    print(f"\n✅ Test complete!")
