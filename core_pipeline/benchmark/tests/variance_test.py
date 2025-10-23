#!/usr/bin/env python3
"""
Statistical variance test to determine if performance regressions are real or noise
"""

import subprocess
import statistics
import os
import re
from pathlib import Path

os.environ['OMP_PROC_BIND'] = 'true'
os.environ['OMP_PLACES'] = 'cores'


def run_many_iterations(filter_name, thread_count, iterations=30):
    """Run many iterations to get statistical confidence"""
    preprocess_bin = "../../bin/preprocess_optimized"
    image_path = "../../images/sample.jpg"
    output_path = f"../../temp/test_{filter_name}.jpg"

    os.environ['OMP_NUM_THREADS'] = str(thread_count)

    times = []

    # Warmup
    for _ in range(5):
        subprocess.run([preprocess_bin, image_path, output_path, filter_name],
                      capture_output=True, text=True, timeout=30)

    # Collect data
    for _ in range(iterations):
        try:
            result = subprocess.run(
                [preprocess_bin, image_path, output_path, filter_name],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                match = re.search(r'Total processing time:\s+([\d.]+)\s+ms', result.stdout)
                if match:
                    times.append(float(match.group(1)))
        except:
            continue

    return times


def analyze_variance(filter_name):
    """Analyze if regressions are statistically significant"""
    print(f"\n{'='*70}")
    print(f"Statistical Analysis: {filter_name.upper()}")
    print(f"{'='*70}")
    print(f"Running 30 iterations per thread count...\n")

    results = {}
    for tc in [1, 2, 4, 8]:
        print(f"  Testing {tc} threads... ", end="", flush=True)
        times = run_many_iterations(filter_name, tc, iterations=30)

        if times:
            results[tc] = {
                'times': times,
                'mean': statistics.mean(times),
                'std': statistics.stdev(times),
                'min': min(times),
                'max': max(times),
                'median': statistics.median(times),
                'cv': statistics.stdev(times) / statistics.mean(times) * 100  # coefficient of variation
            }
            print(f"{results[tc]['mean']:.2f}ms (±{results[tc]['std']:.2f}ms, CV={results[tc]['cv']:.1f}%)")

    # Statistical comparison
    print(f"\n  Variance Analysis:")
    print(f"  {'Threads':<10} {'Mean':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10} {'Range':<10}")
    print(f"  {'-'*70}")

    for tc in [1, 2, 4, 8]:
        if tc in results:
            r = results[tc]
            range_val = r['max'] - r['min']
            print(f"  {tc:<10} {r['mean']:<12.2f} {r['std']:<12.2f} {r['min']:<10.2f} {r['max']:<10.2f} {range_val:<10.2f}")

    # Check for significant regressions
    print(f"\n  Regression Analysis (vs previous thread count):")
    prev_tc = None
    for tc in [1, 2, 4, 8]:
        if tc in results and prev_tc and prev_tc in results:
            mean_prev = results[prev_tc]['mean']
            mean_curr = results[tc]['mean']
            change = ((mean_curr - mean_prev) / mean_prev) * 100

            # Check if regression is beyond 2 standard deviations (95% confidence)
            combined_std = (results[prev_tc]['std']**2 + results[tc]['std']**2)**0.5
            significant = abs(mean_curr - mean_prev) > 2 * combined_std

            status = "❌ SIGNIFICANT" if (change > 10 and significant) else ("⚠️ POSSIBLE" if change > 5 else "✅ NORMAL")
            print(f"  {prev_tc}→{tc} threads: {change:+.1f}% {status}")

        prev_tc = tc

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", required=True, choices=['blur', 'edge', 'sharpen', 'clahe', 'denoise'])
    args = parser.parse_args()

    analyze_variance(args.filter)
