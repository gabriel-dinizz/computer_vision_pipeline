#!/usr/bin/env python3
"""
Comprehensive Results Analysis and Visualization
Generates academic-quality plots and statistical analysis from benchmark results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import statistics


class BenchmarkAnalyzer:
    """
    Comprehensive analysis of multi-variable benchmark results

    Generates:
    - Performance comparison charts
    - Speedup analysis
    - Scaling efficiency plots
    - Resolution impact analysis
    - Statistical summaries
    """

    def __init__(self, results_file: str = "results_comprehensive/results_complete.json"):
        self.results_file = Path(results_file)

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        with open(self.results_file, 'r') as f:
            self.data = json.load(f)

        self.output_dir = self.results_file.parent / "analysis"
        self.output_dir.mkdir(exist_ok=True)

        # Set academic plotting style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

        # Extract metadata
        self.metadata = self.data.get('metadata', {})
        self.results = self.data.get('results', {})

    def generate_all_visualizations(self):
        """Generate all analysis visualizations"""
        print("=" * 80)
        print("Generating Comprehensive Analysis")
        print("=" * 80)

        self.plot_performance_heatmap()
        self.plot_speedup_comparison()
        self.plot_scaling_efficiency()
        self.plot_resolution_impact()
        self.plot_filter_comparison()
        self.plot_thread_scaling()
        self.generate_statistical_report()
        self.export_latex_tables()

        print(f"\n✅ All visualizations generated in: {self.output_dir}/")

    def plot_performance_heatmap(self):
        """Generate heatmap showing performance across all dimensions"""
        print("\n📊 Generating performance heatmap...")

        # Create heatmap for each resolution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Processing Time Heatmap (ms) - All Resolutions',
                     fontsize=16, fontweight='bold')

        resolutions = ['480p', '720p', '1080p', '1440p']

        for idx, resolution in enumerate(resolutions):
            if resolution not in self.results:
                continue

            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            # Prepare data matrix (filters × threads)
            filters = list(self.results[resolution].keys())
            threads = sorted([int(t) for t in list(self.results[resolution][filters[0]].keys())])

            data_matrix = []
            for filter_type in filters:
                row_data = []
                for thread in threads:
                    mean_time = self.results[resolution][filter_type][str(thread)].get('mean_ms', 0)
                    row_data.append(mean_time)
                data_matrix.append(row_data)

            # Create heatmap
            im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

            # Set ticks and labels
            ax.set_xticks(range(len(threads)))
            ax.set_yticks(range(len(filters)))
            ax.set_xticklabels([f"{t}T" for t in threads])
            ax.set_yticklabels(filters)
            ax.set_xlabel('Thread Count')
            ax.set_ylabel('Filter Type')
            ax.set_title(f'{resolution.upper()}')

            # Add values to cells
            for i in range(len(filters)):
                for j in range(len(threads)):
                    text = ax.text(j, i, f'{data_matrix[i][j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=9)

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Time (ms)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'heatmap_performance.svg',
                   format='svg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: heatmap_performance.svg")

    def plot_speedup_comparison(self):
        """Generate speedup comparison across resolutions and filters"""
        print("\n📈 Generating speedup analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Speedup Analysis Across Resolutions',
                     fontsize=16, fontweight='bold')

        resolutions = ['480p', '720p', '1080p', '1440p']

        for idx, resolution in enumerate(resolutions):
            if resolution not in self.results:
                continue

            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            # Plot speedup for each filter
            for filter_type in self.results[resolution].keys():
                threads = []
                speedups = []

                for thread_count, data in self.results[resolution][filter_type].items():
                    if int(thread_count) > 1 and 'speedup' in data:
                        threads.append(int(thread_count))
                        speedups.append(data['speedup'])

                if threads:
                    ax.plot(threads, speedups, marker='o', linewidth=2,
                           markersize=8, label=filter_type.title())

            # Plot ideal speedup line
            max_threads = max([int(t) for t in self.results[resolution][list(self.results[resolution].keys())[0]].keys()])
            ideal_threads = list(range(2, max_threads + 1))
            ax.plot(ideal_threads, ideal_threads, 'k--',
                   alpha=0.5, linewidth=2, label='Ideal Speedup')

            ax.set_xlabel('Thread Count')
            ax.set_ylabel('Speedup')
            ax.set_title(f'{resolution.upper()}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([1, 2, 4, 8])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'speedup_comparison.svg',
                   format='svg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: speedup_comparison.svg")

    def plot_scaling_efficiency(self):
        """Plot parallel efficiency across all scenarios"""
        print("\n⚡ Generating efficiency analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Parallel Efficiency Analysis',
                     fontsize=16, fontweight='bold')

        resolutions = ['480p', '720p', '1080p', '1440p']

        for idx, resolution in enumerate(resolutions):
            if resolution not in self.results:
                continue

            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            # Plot efficiency for each filter
            for filter_type in self.results[resolution].keys():
                threads = []
                efficiencies = []

                for thread_count, data in self.results[resolution][filter_type].items():
                    if int(thread_count) > 1 and 'efficiency' in data:
                        threads.append(int(thread_count))
                        efficiencies.append(data['efficiency'])

                if threads:
                    ax.plot(threads, efficiencies, marker='s', linewidth=2,
                           markersize=8, label=filter_type.title())

            # Plot perfect efficiency line
            ax.axhline(y=1.0, color='k', linestyle='--',
                      alpha=0.5, linewidth=2, label='Perfect Efficiency')

            ax.set_xlabel('Thread Count')
            ax.set_ylabel('Parallel Efficiency')
            ax.set_title(f'{resolution.upper()}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([2, 4, 8])
            ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_analysis.svg',
                   format='svg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: efficiency_analysis.svg")

    def plot_resolution_impact(self):
        """Analyze impact of resolution on performance"""
        print("\n🔍 Analyzing resolution impact...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Resolution Impact on Performance',
                     fontsize=24, fontweight='bold')

        filters = list(self.results['480p'].keys()) if '480p' in self.results else []

        for idx, filter_type in enumerate(filters):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            # Plot performance across resolutions for each thread count
            for thread_count in [1, 2, 4, 8]:
                resolutions = []
                times = []

                for resolution in ['480p', '720p', '1080p', '1440p']:
                    if (resolution in self.results and
                        filter_type in self.results[resolution] and
                        str(thread_count) in self.results[resolution][filter_type]):

                        resolutions.append(resolution)
                        times.append(self.results[resolution][filter_type][str(thread_count)]['mean_ms'])

                if resolutions:
                    ax.plot(resolutions, times, marker='o', linewidth=2,
                           markersize=8, label=f'{thread_count}T')

            ax.set_xlabel('Resolution')
            ax.set_ylabel('Processing Time (ms)')
            ax.set_title(f'{filter_type.title()} Filter')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplot if odd number of filters
        if len(filters) < 6:
            fig.delaxes(axes[1, 2])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'resolution_impact.svg',
                   format='svg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: resolution_impact.svg")

    def plot_filter_comparison(self):
        """Compare filter performance"""
        print("\n📊 Comparing filter performance...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Filter Performance Comparison (8 Threads)',
                     fontsize=16, fontweight='bold')

        resolutions = ['480p', '720p', '1080p', '1440p']

        for idx, resolution in enumerate(resolutions):
            if resolution not in self.results:
                continue

            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            # Extract data for 8 threads
            filters = []
            times = []
            speedups = []

            for filter_type in self.results[resolution].keys():
                if '8' in self.results[resolution][filter_type]:
                    data = self.results[resolution][filter_type]['8']
                    filters.append(filter_type)
                    times.append(data['mean_ms'])
                    speedups.append(data.get('speedup', 0))

            # Create bar plot
            x = np.arange(len(filters))
            width = 0.35

            ax2 = ax.twinx()

            bars1 = ax.bar(x - width/2, times, width, label='Time (ms)', alpha=0.7)
            bars2 = ax2.bar(x + width/2, speedups, width, label='Speedup',
                          alpha=0.7, color='orange')

            ax.set_xlabel('Filter Type')
            ax.set_ylabel('Processing Time (ms)', color='tab:blue')
            ax2.set_ylabel('Speedup', color='tab:orange')
            ax.set_title(f'{resolution.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(filters, rotation=45, ha='right')
            ax.tick_params(axis='y', labelcolor='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax.grid(True, alpha=0.3, axis='y')

            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'filter_comparison.svg',
                   format='svg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: filter_comparison.svg")

    def plot_thread_scaling(self):
        """Overall thread scaling analysis"""
        print("\n🧵 Analyzing thread scaling...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Overall Thread Scaling Performance',
                     fontsize=16, fontweight='bold')

        # Average speedup across all filters and resolutions
        thread_counts = [2, 4, 8]
        avg_speedups = {t: [] for t in thread_counts}
        avg_efficiencies = {t: [] for t in thread_counts}

        for resolution in self.results:
            for filter_type in self.results[resolution]:
                for thread_count in thread_counts:
                    if str(thread_count) in self.results[resolution][filter_type]:
                        data = self.results[resolution][filter_type][str(thread_count)]
                        if 'speedup' in data:
                            avg_speedups[thread_count].append(data['speedup'])
                        if 'efficiency' in data:
                            avg_efficiencies[thread_count].append(data['efficiency'])

        # Plot average speedup
        threads = []
        speedups_mean = []
        speedups_std = []

        for t in thread_counts:
            if avg_speedups[t]:
                threads.append(t)
                speedups_mean.append(statistics.mean(avg_speedups[t]))
                speedups_std.append(statistics.stdev(avg_speedups[t]) if len(avg_speedups[t]) > 1 else 0)

        ax1.errorbar(threads, speedups_mean, yerr=speedups_std,
                    marker='o', linewidth=3, markersize=10, capsize=5,
                    label='Measured Speedup')
        ax1.plot(threads, threads, 'k--', linewidth=2, alpha=0.5,
                label='Ideal Speedup')
        ax1.set_xlabel('Thread Count', fontsize=12)
        ax1.set_ylabel('Average Speedup', fontsize=12)
        ax1.set_title('Average Speedup vs Thread Count')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(thread_counts)

        # Plot average efficiency
        effs_mean = []
        effs_std = []

        for t in thread_counts:
            if avg_efficiencies[t]:
                effs_mean.append(statistics.mean(avg_efficiencies[t]))
                effs_std.append(statistics.stdev(avg_efficiencies[t]) if len(avg_efficiencies[t]) > 1 else 0)

        ax2.errorbar(threads, effs_mean, yerr=effs_std,
                    marker='s', linewidth=3, markersize=10, capsize=5,
                    label='Measured Efficiency')
        ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=2,
                   alpha=0.5, label='Perfect Efficiency')
        ax2.set_xlabel('Thread Count', fontsize=12)
        ax2.set_ylabel('Average Parallel Efficiency', fontsize=12)
        ax2.set_title('Average Efficiency vs Thread Count')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(thread_counts)
        ax2.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'thread_scaling.svg',
                   format='svg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: thread_scaling.svg")

    def generate_statistical_report(self):
        """Generate comprehensive statistical report"""
        print("\n📄 Generating statistical report...")

        report_file = self.output_dir / 'statistical_analysis.md'

        with open(report_file, 'w') as f:
            f.write("# Comprehensive Benchmark Statistical Analysis\n\n")
            f.write(f"**Analysis Date:** {self.metadata.get('timestamp', 'Unknown')}\n\n")

            f.write("## Overall Performance Summary\n\n")

            # Calculate overall statistics
            all_speedups_8t = []
            all_efficiencies_8t = []

            for resolution in self.results:
                for filter_type in self.results[resolution]:
                    if '8' in self.results[resolution][filter_type]:
                        data = self.results[resolution][filter_type]['8']
                        if 'speedup' in data:
                            all_speedups_8t.append(data['speedup'])
                        if 'efficiency' in data:
                            all_efficiencies_8t.append(data['efficiency'])

            if all_speedups_8t:
                f.write(f"- **Average 8-thread speedup:** {statistics.mean(all_speedups_8t):.3f}x\n")
                f.write(f"- **Best 8-thread speedup:** {max(all_speedups_8t):.3f}x\n")
                f.write(f"- **Worst 8-thread speedup:** {min(all_speedups_8t):.3f}x\n")
                f.write(f"- **Speedup std deviation:** {statistics.stdev(all_speedups_8t):.3f}\n\n")

            if all_efficiencies_8t:
                f.write(f"- **Average 8-thread efficiency:** {statistics.mean(all_efficiencies_8t):.3f}\n")
                f.write(f"- **Best 8-thread efficiency:** {max(all_efficiencies_8t):.3f}\n")
                f.write(f"- **Worst 8-thread efficiency:** {min(all_efficiencies_8t):.3f}\n")
                f.write(f"- **Efficiency std deviation:** {statistics.stdev(all_efficiencies_8t):.3f}\n\n")

            # Per-resolution summary
            f.write("## Per-Resolution Analysis\n\n")

            for resolution in ['480p', '720p', '1080p', '1440p']:
                if resolution in self.results:
                    f.write(f"\n### {resolution.upper()}\n\n")

                    res_speedups = []
                    res_efficiencies = []

                    for filter_type in self.results[resolution]:
                        if '8' in self.results[resolution][filter_type]:
                            data = self.results[resolution][filter_type]['8']
                            if 'speedup' in data:
                                res_speedups.append(data['speedup'])
                            if 'efficiency' in data:
                                res_efficiencies.append(data['efficiency'])

                    if res_speedups:
                        f.write(f"- Average speedup (8T): {statistics.mean(res_speedups):.3f}x\n")
                        f.write(f"- Average efficiency (8T): {statistics.mean(res_efficiencies):.3f}\n\n")

        print(f"  Saved: statistical_analysis.md")

    def export_latex_tables(self):
        """Export results as LaTeX tables for academic papers"""
        print("\n📐 Generating LaTeX tables...")

        latex_file = self.output_dir / 'latex_tables.tex'

        with open(latex_file, 'w') as f:
            f.write("% LaTeX Tables for Academic Paper\n\n")

            # Table for each resolution
            for resolution in ['480p', '720p', '1080p', '1440p']:
                if resolution not in self.results:
                    continue

                f.write(f"% Table: {resolution.upper()} Results\n")
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{Performance Results for {resolution.upper()} Resolution}}\n")
                f.write("\\begin{tabular}{lrrrrrr}\n")
                f.write("\\hline\n")
                f.write("Filter & 1T (ms) & 2T (ms) & 4T (ms) & 8T (ms) & Speedup & Efficiency \\\\\n")
                f.write("\\hline\n")

                for filter_type in self.results[resolution]:
                    filter_data = self.results[resolution][filter_type]

                    t1 = filter_data.get('1', {}).get('mean_ms', 0)
                    t2 = filter_data.get('2', {}).get('mean_ms', 0)
                    t4 = filter_data.get('4', {}).get('mean_ms', 0)
                    t8 = filter_data.get('8', {}).get('mean_ms', 0)
                    speedup = filter_data.get('8', {}).get('speedup', 0)
                    efficiency = filter_data.get('8', {}).get('efficiency', 0)

                    f.write(f"{filter_type.title()} & {t1:.1f} & {t2:.1f} & {t4:.1f} & {t8:.1f} & {speedup:.2f} & {efficiency:.2f} \\\\\n")

                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write(f"\\label{{tab:{resolution}_results}}\n")
                f.write("\\end{table}\n\n")

        print(f"  Saved: latex_tables.tex")


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze comprehensive benchmark results"
    )
    parser.add_argument(
        '--results',
        default='results_comprehensive/results_complete.json',
        help='Path to results JSON file'
    )

    args = parser.parse_args()

    try:
        analyzer = BenchmarkAnalyzer(args.results)
        analyzer.generate_all_visualizations()

        print("\n✅ Analysis complete!")
        print(f"\nGenerated files in: {analyzer.output_dir}/")
        print("  - heatmap_performance.svg")
        print("  - speedup_comparison.svg")
        print("  - efficiency_analysis.svg")
        print("  - resolution_impact.svg")
        print("  - filter_comparison.svg")
        print("  - thread_scaling.svg")
        print("  - statistical_analysis.md")
        print("  - latex_tables.tex")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease run the comprehensive benchmark first:")
        print("  python3 run_comprehensive_benchmark.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
