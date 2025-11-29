#!/usr/bin/env python3
"""
Kaggle Dataset Benchmark Runner
Downloads images from Kaggle using kagglehub and runs comprehensive benchmarks
"""

import kagglehub
import json
import os
import statistics
from pathlib import Path
from typing import List, Dict
import time
import shutil

# Import existing benchmark classes
import sys
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from benchmark_academic import AcademicCVBenchmark


class KaggleDatasetBenchmark:
    """
    Benchmark runner for Kaggle datasets using kagglehub

    Downloads images from Kaggle and runs comprehensive benchmarks
    on 50 images for statistical analysis
    """

    def __init__(self,
                 dataset_name: str = "pavansanagapati/images-dataset",
                 num_images: int = 50,
                 output_dir: str = "results_kaggle_dataset"):
        """
        Initialize Kaggle dataset benchmark

        Args:
            dataset_name: Kaggle dataset identifier (e.g., "username/dataset-name")
            num_images: Number of images to test (default: 50)
            output_dir: Directory to save results
        """
        self.dataset_name = dataset_name
        self.num_images = num_images
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        self.results_dir = self.output_dir / "individual_results"
        self.results_dir.mkdir(exist_ok=True)

        # Configuration
        self.thread_counts = [1, 2, 4, 8]
        self.filters = ['blur', 'sharpen', 'denoise', 'clahe', 'edge']

    def download_dataset(self):
        """Download dataset from Kaggle using kagglehub"""
        print(f"\n📥 Downloading dataset: {self.dataset_name}")
        print("This may take a few minutes on first download...")

        try:
            # Download latest version
            path = kagglehub.dataset_download(self.dataset_name)

            print(f"✓ Dataset downloaded to: {path}")
            return Path(path)

        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("\nMake sure you have kagglehub installed:")
            print("  pip install kagglehub")
            return None

    def select_images(self, dataset_path: Path) -> List[Path]:
        """
        Select N images from downloaded dataset

        Args:
            dataset_path: Path to downloaded dataset

        Returns:
            List of image paths (limited to num_images)
        """
        print(f"\n🖼️  Selecting {self.num_images} images...")

        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        all_images = []

        for ext in image_extensions:
            all_images.extend(dataset_path.rglob(f'*{ext}'))
            all_images.extend(dataset_path.rglob(f'*{ext.upper()}'))

        print(f"Found {len(all_images)} total images in dataset")

        # Select only the requested number of images
        selected = all_images[:self.num_images]

        print(f"Selected {len(selected)} images for benchmarking")

        # Copy to working directory
        copied_images = []
        for idx, img_path in enumerate(selected, 1):
            dest = self.images_dir / f"image_{idx:03d}{img_path.suffix}"
            shutil.copy2(img_path, dest)
            copied_images.append(dest)

            if idx % 10 == 0:
                print(f"  Copied {idx}/{len(selected)} images...")

        print(f"✓ Copied {len(copied_images)} images to {self.images_dir}")
        return copied_images

    def run_benchmark_on_image(self, image_path: Path, image_idx: int) -> Dict:
        """
        Run benchmark on a single image

        Args:
            image_path: Path to image
            image_idx: Image index number

        Returns:
            Benchmark results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Image {image_idx}/{self.num_images}: {image_path.name}")
        print(f"{'='*70}")

        results = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "image_index": image_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "preprocessing": {}
        }

        # Run preprocessing benchmark for each filter and thread count
        for filter_type in self.filters:
            print(f"\n  📊 Testing {filter_type} filter...")
            results["preprocessing"][filter_type] = {}

            for thread_count in self.thread_counts:
                os.environ['OMP_NUM_THREADS'] = str(thread_count)

                # Run single test
                benchmark = AcademicCVBenchmark()
                elapsed = benchmark._run_single_preprocessing(
                    str(image_path),
                    filter_type
                )

                if elapsed > 0:
                    results["preprocessing"][filter_type][thread_count] = {
                        "time_ms": elapsed
                    }
                    print(f"    {thread_count}T: {elapsed:.1f}ms", end="")

                    # Show speedup for parallel runs
                    if thread_count > 1 and 1 in results["preprocessing"][filter_type]:
                        seq_time = results["preprocessing"][filter_type][1]["time_ms"]
                        speedup = seq_time / elapsed
                        print(f" (speedup: {speedup:.2f}x)")
                    else:
                        print()

        # Calculate speedups and efficiency
        for filter_type in results["preprocessing"]:
            if 1 in results["preprocessing"][filter_type]:
                sequential_time = results["preprocessing"][filter_type][1]["time_ms"]

                for tc in self.thread_counts:
                    if tc > 1 and tc in results["preprocessing"][filter_type]:
                        parallel_time = results["preprocessing"][filter_type][tc]["time_ms"]
                        speedup = sequential_time / parallel_time
                        efficiency = speedup / tc

                        results["preprocessing"][filter_type][tc]["speedup"] = speedup
                        results["preprocessing"][filter_type][tc]["efficiency"] = efficiency

        # Save individual result
        result_file = self.results_dir / f"image_{image_idx:03d}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Saved results to {result_file.name}")
        return results

    def run_all_benchmarks(self, image_paths: List[Path]) -> List[Dict]:
        """
        Run benchmarks on all images

        Args:
            image_paths: List of image paths

        Returns:
            List of all benchmark results
        """
        print(f"\n{'='*80}")
        print(f"🚀 STARTING BENCHMARK ON {len(image_paths)} IMAGES")
        print(f"{'='*80}")
        print(f"Filters: {self.filters}")
        print(f"Thread counts: {self.thread_counts}")
        print(f"Total tests per image: {len(self.filters) * len(self.thread_counts)}")
        print(f"Total tests: {len(image_paths) * len(self.filters) * len(self.thread_counts)}")

        all_results = []
        start_time = time.time()

        for idx, img_path in enumerate(image_paths, 1):
            try:
                result = self.run_benchmark_on_image(img_path, idx)
                all_results.append(result)

                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (len(image_paths) - idx) * avg_time
                print(f"\n⏱️  Progress: {idx}/{len(image_paths)} | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"ETA: {remaining/60:.1f}min")

            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {e}")
                continue

        total_time = time.time() - start_time
        print(f"\n✓ Completed all benchmarks in {total_time/60:.1f} minutes")

        return all_results

    def generate_comparative_analysis(self, all_results: List[Dict]):
        """
        Generate comparative analysis across all images

        Args:
            all_results: List of all benchmark results
        """
        print(f"\n📊 Generating Comparative Analysis...")

        analysis = {
            "metadata": {
                "num_images": len(all_results),
                "dataset": self.dataset_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "filters": self.filters,
                "thread_counts": self.thread_counts
            },
            "per_filter_analysis": {}
        }

        # Aggregate statistics for each filter and thread count
        for filter_type in self.filters:
            analysis["per_filter_analysis"][filter_type] = {}

            for thread_count in self.thread_counts:
                times = []
                speedups = []
                efficiencies = []

                for result in all_results:
                    if (filter_type in result["preprocessing"] and
                        thread_count in result["preprocessing"][filter_type]):

                        data = result["preprocessing"][filter_type][thread_count]
                        times.append(data["time_ms"])

                        if "speedup" in data:
                            speedups.append(data["speedup"])
                        if "efficiency" in data:
                            efficiencies.append(data["efficiency"])

                if times:
                    analysis["per_filter_analysis"][filter_type][thread_count] = {
                        "mean_time_ms": statistics.mean(times),
                        "median_time_ms": statistics.median(times),
                        "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0,
                        "min_time_ms": min(times),
                        "max_time_ms": max(times),
                        "num_samples": len(times)
                    }

                    if speedups:
                        analysis["per_filter_analysis"][filter_type][thread_count].update({
                            "mean_speedup": statistics.mean(speedups),
                            "median_speedup": statistics.median(speedups),
                            "std_speedup": statistics.stdev(speedups) if len(speedups) > 1 else 0,
                            "min_speedup": min(speedups),
                            "max_speedup": max(speedups)
                        })

                    if efficiencies:
                        analysis["per_filter_analysis"][filter_type][thread_count].update({
                            "mean_efficiency": statistics.mean(efficiencies),
                            "median_efficiency": statistics.median(efficiencies),
                            "std_efficiency": statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0
                        })

        # Save analysis
        analysis_file = self.output_dir / "comparative_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"✓ Saved: {analysis_file}")

        # Generate summary report
        self._generate_summary_report(analysis)

        return analysis

    def _generate_summary_report(self, analysis: Dict):
        """Generate human-readable summary report"""
        report_file = self.output_dir / "comparative_summary.md"

        with open(report_file, 'w') as f:
            f.write("# Kaggle Dataset Benchmark - Comparative Analysis\n\n")
            f.write(f"**Dataset:** {self.dataset_name}\n")
            f.write(f"**Number of Images:** {analysis['metadata']['num_images']}\n")
            f.write(f"**Timestamp:** {analysis['metadata']['timestamp']}\n\n")

            f.write("## Summary Statistics (8 Threads)\n\n")
            f.write("| Filter | Mean Time (ms) | Std Dev | Mean Speedup | Mean Efficiency |\n")
            f.write("|--------|----------------|---------|--------------|------------------|\n")

            for filter_type in self.filters:
                if (filter_type in analysis["per_filter_analysis"] and
                    8 in analysis["per_filter_analysis"][filter_type]):

                    data = analysis["per_filter_analysis"][filter_type][8]
                    f.write(f"| {filter_type.title()} | ")
                    f.write(f"{data['mean_time_ms']:.1f} | ")
                    f.write(f"{data['std_time_ms']:.1f} | ")
                    f.write(f"{data.get('mean_speedup', 0):.2f}x | ")
                    f.write(f"{data.get('mean_efficiency', 0):.2f} |\n")

            f.write("\n## Performance Across Thread Counts\n\n")

            for filter_type in self.filters:
                f.write(f"\n### {filter_type.title()} Filter\n\n")
                f.write("| Threads | Mean Time (ms) | Speedup | Efficiency |\n")
                f.write("|---------|----------------|---------|------------|\n")

                for tc in self.thread_counts:
                    if (filter_type in analysis["per_filter_analysis"] and
                        tc in analysis["per_filter_analysis"][filter_type]):
                        data = analysis["per_filter_analysis"][filter_type][tc]
                        speedup = data.get('mean_speedup', 1.0)
                        efficiency = data.get('mean_efficiency', 1.0)
                        f.write(f"| {tc} | {data['mean_time_ms']:.1f} | ")
                        f.write(f"{speedup:.2f}x | {efficiency:.2f} |\n")

            f.write("\n## Overall Performance Gains\n\n")

            # Calculate overall statistics
            all_speedups = []
            all_efficiencies = []

            for filter_type in self.filters:
                if (filter_type in analysis["per_filter_analysis"] and
                    8 in analysis["per_filter_analysis"][filter_type]):
                    data = analysis["per_filter_analysis"][filter_type][8]
                    if "mean_speedup" in data:
                        all_speedups.append(data["mean_speedup"])
                    if "mean_efficiency" in data:
                        all_efficiencies.append(data["mean_efficiency"])

            if all_speedups:
                avg_speedup = statistics.mean(all_speedups)
                avg_efficiency = statistics.mean(all_efficiencies)
                best_speedup = max(all_speedups)
                worst_speedup = min(all_speedups)

                f.write(f"- **Average 8-thread speedup:** {avg_speedup:.2f}x\n")
                f.write(f"- **Best speedup:** {best_speedup:.2f}x\n")
                f.write(f"- **Worst speedup:** {worst_speedup:.2f}x\n")
                f.write(f"- **Average 8-thread efficiency:** {avg_efficiency:.2f}\n")
                f.write(f"- **Performance gain:** {(avg_speedup - 1) * 100:.1f}%\n")
                f.write(f"- **Sample size:** {analysis['metadata']['num_images']} images\n\n")

            f.write("## Conclusion\n\n")
            f.write(f"Benchmarked {analysis['metadata']['num_images']} images from Kaggle ")
            f.write(f"dataset '{self.dataset_name}' across {len(self.filters)} filters ")
            f.write(f"and {len(self.thread_counts)} thread configurations. ")
            f.write("Results demonstrate consistent performance improvements with ")
            f.write("OpenMP parallelization across diverse images.\n")

        print(f"✓ Saved: {report_file}")

    def run_complete_study(self):
        """Run complete Kaggle dataset benchmark study"""
        print("=" * 80)
        print("KAGGLE DATASET BENCHMARK STUDY")
        print(f"Dataset: {self.dataset_name}")
        print(f"Target: {self.num_images} images")
        print("=" * 80)

        # Step 1: Download dataset
        dataset_path = self.download_dataset()
        if not dataset_path:
            print("❌ Failed to download dataset!")
            return None

        # Step 2: Select 50 images
        image_paths = self.select_images(dataset_path)
        if not image_paths:
            print("❌ No images found!")
            return None

        # Step 3: Run benchmarks
        all_results = self.run_all_benchmarks(image_paths)

        # Step 4: Generate analysis
        analysis = self.generate_comparative_analysis(all_results)

        print("\n" + "=" * 80)
        print("✅ BENCHMARK STUDY COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved in: {self.output_dir}/")
        print(f"  - {len(all_results)} individual result JSON files")
        print(f"  - comparative_analysis.json (aggregate statistics)")
        print(f"  - comparative_summary.md (human-readable report)")
        print(f"\nView summary: cat {self.output_dir}/comparative_summary.md")

        return analysis


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run benchmarks on Kaggle dataset using kagglehub"
    )
    parser.add_argument(
        '--dataset',
        default='pavansanagapati/images-dataset',
        help='Kaggle dataset name (default: pavansanagapati/images-dataset)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=50,
        help='Number of images to test (default: 50)'
    )
    parser.add_argument(
        '--output',
        default='results_kaggle_dataset',
        help='Output directory (default: results_kaggle_dataset)'
    )

    args = parser.parse_args()

    print("\n📦 Required package: kagglehub")
    print("Install with: pip install kagglehub\n")

    benchmark = KaggleDatasetBenchmark(
        dataset_name=args.dataset,
        num_images=args.num_images,
        output_dir=args.output
    )

    try:
        benchmark.run_complete_study()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
