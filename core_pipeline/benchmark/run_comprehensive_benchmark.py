#!/usr/bin/env python3
"""
Comprehensive Multi-Variable Benchmark Automation
Tests all combinations of: Filters × Resolutions × Thread Counts

This script automates the complete benchmark matrix for academic analysis.
"""

import subprocess
import json
import time
import os
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import re


class ComprehensiveBenchmark:
    """
    Automated comprehensive benchmarking across all variables:
    - Filters: blur, sharpen, denoise, clahe, edge
    - Resolutions: 480p, 720p, 1080p, 1440p
    - Thread counts: 1, 2, 4, 8
    """

    def __init__(self, preprocess_bin: str = "../bin/preprocess_optimized"):
        self.preprocess_bin = Path(preprocess_bin)
        self.results_dir = Path("results_comprehensive")
        self.results_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

        # Benchmark configuration
        self.filters = ['blur', 'sharpen', 'denoise', 'clahe', 'edge']
        self.resolutions = ['480p', '720p', '1080p', '1440p']
        self.thread_counts = [1, 2, 4, 8]

        # Test parameters
        self.iterations_per_test = 10        # Increased for stability
        self.warmup_iterations = 5           # More warmup for CPU freq stabilization

        # OpenMP configuration for reproducibility
        os.environ['OMP_PROC_BIND'] = 'true'
        os.environ['OMP_PLACES'] = 'cores'

        # Results storage
        self.all_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'filters': self.filters,
                'resolutions': self.resolutions,
                'thread_counts': self.thread_counts,
                'iterations_per_test': self.iterations_per_test,
                'warmup_iterations': self.warmup_iterations
            },
            'results': {}
        }

    def calculate_total_tests(self) -> int:
        """Calculate total number of tests to run"""
        # Each test: filter × resolution × thread_count
        return len(self.filters) * len(self.resolutions) * len(self.thread_counts)

    def run_single_test(self, image_path: str, filter_type: str,
                       thread_count: int) -> Dict:
        """
        Run a single benchmark test

        Returns:
            Dict with timing statistics
        """
        os.environ['OMP_NUM_THREADS'] = str(thread_count)

        output_path = self.temp_dir / f"bench_{filter_type}_{thread_count}.jpg"

        times = []

        # Warmup runs
        for _ in range(self.warmup_iterations):
            self._run_preprocessing(image_path, filter_type, output_path)

        # Actual benchmark runs
        for _ in range(self.iterations_per_test):
            elapsed = self._run_preprocessing(image_path, filter_type, output_path)
            if elapsed > 0:
                times.append(elapsed)

        if not times:
            return {'error': 'All runs failed'}

        # Calculate statistics
        return {
            'times_ms': times,
            'mean_ms': statistics.mean(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_ms': min(times),
            'max_ms': max(times),
            'median_ms': statistics.median(times)
        }

    def _run_preprocessing(self, image_path: str, filter_type: str,
                          output_path: Path) -> float:
        """Execute preprocessing and return time in ms"""
        try:
            cmd = [
                str(self.preprocess_bin),
                image_path,
                str(output_path),
                filter_type
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # Parse processing time from C++ output
                match = re.search(r'Total processing time:\s+([\d.]+)\s+ms',
                                 result.stdout)
                if match:
                    return float(match.group(1))

            return -1

        except (subprocess.TimeoutExpired, Exception):
            return -1

    def run_resolution_benchmark(self, base_image_name: str,
                                 benchmark_dataset_dir: str = "../images/benchmark_dataset"):
        """
        Run comprehensive benchmark for all resolutions of a base image

        Args:
            base_image_name: Base name of image (e.g., 'sample_noisy')
            benchmark_dataset_dir: Directory containing resized images
        """
        dataset_dir = Path(benchmark_dataset_dir)

        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Benchmark dataset not found: {dataset_dir}\n"
                f"Please run prepare_images.py first!"
            )

        total_tests = self.calculate_total_tests()
        current_test = 0

        print("=" * 80)
        print("COMPREHENSIVE BENCHMARK - Multi-Variable Analysis")
        print("=" * 80)
        print(f"Image: {base_image_name}")
        print(f"Filters: {self.filters}")
        print(f"Resolutions: {self.resolutions}")
        print(f"Thread counts: {self.thread_counts}")
        print(f"Total tests: {total_tests}")
        print(f"Iterations per test: {self.iterations_per_test}")
        print("=" * 80)
        print()

        start_time = time.time()

        # Iterate through all combinations
        for resolution in self.resolutions:
            print(f"\n{'='*80}")
            print(f"Resolution: {resolution}")
            print(f"{'='*80}")

            # Find image for this resolution
            image_filename = f"{base_image_name}_{resolution}.jpg"
            image_path = dataset_dir / resolution / image_filename

            if not image_path.exists():
                print(f"  WARNING: Image not found: {image_path}")
                print(f"  Skipping {resolution}...")
                continue

            # Initialize results structure for this resolution
            if resolution not in self.all_results['results']:
                self.all_results['results'][resolution] = {}

            for filter_type in self.filters:
                print(f"\n  Filter: {filter_type}")

                if filter_type not in self.all_results['results'][resolution]:
                    self.all_results['results'][resolution][filter_type] = {}

                for thread_count in self.thread_counts:
                    current_test += 1
                    progress = (current_test / total_tests) * 100

                    print(f"    {thread_count} threads [{current_test}/{total_tests} - {progress:.1f}%]: ",
                          end='', flush=True)

                    # Run test
                    result = self.run_single_test(
                        str(image_path),
                        filter_type,
                        thread_count
                    )

                    # Store result
                    self.all_results['results'][resolution][filter_type][thread_count] = result

                    if 'error' not in result:
                        print(f"{result['mean_ms']:.1f}ms "
                              f"(±{result['std_ms']:.1f}ms)")
                    else:
                        print("FAILED")

                    # Save intermediate results every 10 tests
                    if current_test % 10 == 0:
                        self._save_intermediate_results()

        # Calculate speedup and efficiency metrics
        self._calculate_all_metrics()

        # Save final results
        self._save_final_results()

        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Benchmark completed in {elapsed_time:.1f} seconds")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*80}")

    def _calculate_all_metrics(self):
        """Calculate speedup and efficiency for all tests"""
        print("\n📊 Calculating speedup and efficiency metrics...")

        for resolution, res_data in self.all_results['results'].items():
            for filter_type, filter_data in res_data.items():
                # Get baseline (1 thread) performance
                if 1 in filter_data and 'error' not in filter_data[1]:
                    baseline_time = filter_data[1]['mean_ms']

                    for thread_count, data in filter_data.items():
                        if thread_count > 1 and 'error' not in data:
                            parallel_time = data['mean_ms']

                            # Calculate speedup: T(1) / T(n)
                            speedup = baseline_time / parallel_time

                            # Calculate efficiency: speedup / n
                            efficiency = speedup / thread_count

                            # Store metrics
                            data['speedup'] = speedup
                            data['efficiency'] = efficiency

    def _save_intermediate_results(self):
        """Save intermediate results during benchmarking"""
        intermediate_file = self.results_dir / 'results_intermediate.json'
        with open(intermediate_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)

    def _save_final_results(self):
        """Save final comprehensive results"""
        # Save as JSON
        json_file = self.results_dir / 'results_complete.json'
        with open(json_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        print(f"\n✅ Results saved: {json_file}")

        # Generate CSV export
        self._export_to_csv()

    def _export_to_csv(self):
        """Export results to CSV for easy analysis"""
        import csv

        csv_file = self.results_dir / 'results_complete.csv'

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Resolution', 'Filter', 'Threads',
                'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)', 'Median (ms)',
                'Speedup', 'Efficiency'
            ])

            # Data rows
            for resolution, res_data in self.all_results['results'].items():
                for filter_type, filter_data in res_data.items():
                    for thread_count, data in filter_data.items():
                        if 'error' not in data:
                            writer.writerow([
                                resolution,
                                filter_type,
                                thread_count,
                                f"{data['mean_ms']:.2f}",
                                f"{data['std_ms']:.2f}",
                                f"{data['min_ms']:.2f}",
                                f"{data['max_ms']:.2f}",
                                f"{data['median_ms']:.2f}",
                                f"{data.get('speedup', 0):.3f}",
                                f"{data.get('efficiency', 0):.3f}"
                            ])

        print(f"✅ CSV export: {csv_file}")

    def generate_summary_report(self):
        """Generate markdown summary report"""
        report_file = self.results_dir / 'summary_report.md'

        with open(report_file, 'w') as f:
            f.write("# Comprehensive Benchmark Results\n\n")
            f.write(f"**Timestamp:** {self.all_results['metadata']['timestamp']}\n\n")

            f.write("## Test Configuration\n\n")
            f.write(f"- **Filters:** {', '.join(self.filters)}\n")
            f.write(f"- **Resolutions:** {', '.join(self.resolutions)}\n")
            f.write(f"- **Thread counts:** {', '.join(map(str, self.thread_counts))}\n")
            f.write(f"- **Iterations per test:** {self.iterations_per_test}\n")
            f.write(f"- **Warmup iterations:** {self.warmup_iterations}\n\n")

            f.write("## Results Overview\n\n")

            # Summary table for each resolution
            for resolution in self.resolutions:
                if resolution in self.all_results['results']:
                    f.write(f"\n### {resolution.upper()} Results\n\n")
                    f.write("| Filter | 1T (ms) | 2T (ms) | 4T (ms) | 8T (ms) | Speedup (8T) | Efficiency (8T) |\n")
                    f.write("|--------|---------|---------|---------|---------|--------------|------------------|\n")

                    res_data = self.all_results['results'][resolution]

                    for filter_type in self.filters:
                        if filter_type in res_data:
                            filter_data = res_data[filter_type]

                            t1 = filter_data.get(1, {}).get('mean_ms', 0)
                            t2 = filter_data.get(2, {}).get('mean_ms', 0)
                            t4 = filter_data.get(4, {}).get('mean_ms', 0)
                            t8 = filter_data.get(8, {}).get('mean_ms', 0)
                            speedup = filter_data.get(8, {}).get('speedup', 0)
                            efficiency = filter_data.get(8, {}).get('efficiency', 0)

                            f.write(f"| {filter_type} | {t1:.1f} | {t2:.1f} | {t4:.1f} | {t8:.1f} | {speedup:.2f}x | {efficiency:.2f} |\n")

        print(f"✅ Summary report: {report_file}")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive multi-variable benchmark"
    )
    parser.add_argument(
        '--image',
        default='sample_noisy',
        help='Base image name (without resolution suffix)'
    )
    parser.add_argument(
        '--dataset-dir',
        default='../images/benchmark_dataset',
        help='Directory containing prepared benchmark images'
    )
    parser.add_argument(
        '--preprocess-bin',
        default='../bin/preprocess_optimized',
        help='Path to preprocessing binary'
    )

    args = parser.parse_args()

    # Verify binary exists
    if not Path(args.preprocess_bin).exists():
        print(f"❌ Error: Preprocessing binary not found: {args.preprocess_bin}")
        print("\nPlease build it first:")
        print("  cd /path/to/project")
        print("  make all")
        return

    # Verify dataset exists
    if not Path(args.dataset_dir).exists():
        print(f"❌ Error: Benchmark dataset not found: {args.dataset_dir}")
        print("\nPlease prepare images first:")
        print("  python3 prepare_images.py")
        return

    # Run comprehensive benchmark
    benchmark = ComprehensiveBenchmark(args.preprocess_bin)

    try:
        benchmark.run_resolution_benchmark(args.image, args.dataset_dir)
        benchmark.generate_summary_report()

        print("\n✅ Comprehensive benchmark completed successfully!")
        print(f"\nResults available in: {benchmark.results_dir}/")
        print("  - results_complete.json (detailed JSON)")
        print("  - results_complete.csv (spreadsheet)")
        print("  - summary_report.md (markdown report)")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
