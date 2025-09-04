#!/usr/bin/env python3
"""
Academic Research Benchmark for OPTIMIZED OpenMP Performance Analysis
Compares original vs optimized OpenMP implementations
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
import json
from pathlib import Path
import statistics
import psutil
import platform
from scipy import stats

class OptimizedOpenMPBenchmark:
    def __init__(self, iterations=100):
        self.thread_counts = [1, 2, 4, 8, 16]
        self.iterations = iterations
        self.filters = ['blur', 'sharpen', 'denoise', 'clahe', 'edge']
        self.results = []
        
        # Create results directory
        self.results_dir = Path('research_results_optimized')
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup controlled environment
        self.setup_controlled_environment()
        
    def setup_controlled_environment(self):
        """Setup controlled environment for reproducible benchmarks."""
        print("🔧 Setting up controlled benchmark environment...")
        
        # Disable CPU frequency scaling (warning only, requires sudo)
        print("⚠️  For most accurate results, consider disabling CPU frequency scaling:")
        if platform.system() == "Linux":
            print("   sudo cpupower frequency-set --governor performance")
        elif platform.system() == "Darwin":  # macOS
            print("   System Preferences > Energy Saver > Prevent computer from sleeping")
        
        # Set CPU affinity if possible
        try:
            process = psutil.Process()
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
            if cpu_count >= 4:
                # Use first 4 physical cores for consistent results
                process.cpu_affinity(list(range(min(4, cpu_count))))
                print(f"✅ CPU affinity set to cores: {list(range(min(4, cpu_count)))}")
            else:
                print(f"ℹ️  Using all {cpu_count} available cores")
        except (AttributeError, OSError):
            print("ℹ️  CPU affinity control not available on this platform")
        
        # Set environment variables for reproducible OpenMP behavior
        os.environ['OMP_DYNAMIC'] = 'FALSE'
        os.environ['OMP_NESTED'] = 'FALSE'
        os.environ['OMP_PROC_BIND'] = 'TRUE'
        os.environ['OMP_PLACES'] = 'cores'
        
        print(f"✅ OpenMP environment configured for reproducible results")
        
        # Warm up the system
        print("🏃 Warming up system...")
        self._warmup_system()
        
    def _warmup_system(self):
        """Warm up the system to stabilize performance."""
        warmup_image = "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg"
        if os.path.exists(warmup_image):
            for _ in range(3):
                try:
                    cmd = ["./bin/preprocess_optimized", warmup_image, "/tmp/warmup_output.jpg", "blur"]
                    subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                    if os.path.exists("/tmp/warmup_output.jpg"):
                        os.remove("/tmp/warmup_output.jpg")
                except (subprocess.SubprocessError, FileNotFoundError):
                    break
        print("✅ System warmup completed")
    
    def run_sequential_benchmark(self, image_path, filter_type):
        """Run sequential (baseline) benchmark"""
        times = []
        
        for i in range(self.iterations):
            start = time.time()
            result = subprocess.run([
                './bin/sequential_baseline', image_path, 'temp/seq_output.jpg', filter_type
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                continue
                
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        return {
            'type': 'sequential',
            'implementation': 'baseline',
            'threads': 1,
            'filter': filter_type,
            'times': times,
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times),
            'iterations': len(times)
        }
    
    def run_original_parallel_benchmark(self, image_path, filter_type, threads):
        """Run original (problematic) OpenMP benchmark"""
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        
        times = []
        
        for i in range(self.iterations):
            start = time.time()
            result = subprocess.run([
                './bin/preprocess', image_path, 'temp/orig_par_output.jpg', filter_type
            ], env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                continue
                
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        return {
            'type': 'parallel_original',
            'implementation': 'original_openmp',
            'threads': threads,
            'filter': filter_type,
            'times': times,
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times),
            'iterations': len(times)
        }
    
    def run_optimized_parallel_benchmark(self, image_path, filter_type, threads):
        """Run optimized OpenMP benchmark"""
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        
        times = []
        
        for i in range(self.iterations):
            start = time.time()
            result = subprocess.run([
                './bin/preprocess_optimized', image_path, 'temp/opt_par_output.jpg', filter_type
            ], env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                continue
                
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        return {
            'type': 'parallel_optimized',
            'implementation': 'optimized_openmp',
            'threads': threads,
            'filter': filter_type,
            'times': times,
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times),
            'iterations': len(times)
        }
    
    def run_opencv_baseline(self, image_path, filter_type):
        """Run OpenCV native implementation for comparison"""
        times = []
        
        for i in range(self.iterations):
            start = time.time()
            result = subprocess.run([
                './bin/opencv_baseline', image_path, 'temp/opencv_output.jpg', filter_type
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                continue
                
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        return {
            'type': 'opencv_native',
            'implementation': 'opencv_baseline',
            'threads': 1,
            'filter': filter_type,
            'times': times,
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times),
            'iterations': len(times)
        }
    
    def calculate_comprehensive_metrics(self, sequential_result, original_results, optimized_results, opencv_result):
        """Calculate comprehensive research metrics comparing all implementations"""
        baseline_time = sequential_result['mean_time_ms']
        opencv_time = opencv_result['mean_time_ms']
        
        analysis = {
            'filter': sequential_result['filter'],
            'baseline_time_ms': baseline_time,
            'opencv_time_ms': opencv_time,
            'opencv_speedup_vs_sequential': baseline_time / opencv_time,
            'thread_analysis': []
        }
        
        # Combine results by thread count
        for threads in self.thread_counts:
            if threads == 1:
                continue
                
            # Find results for this thread count
            orig_result = next((r for r in original_results if r['threads'] == threads), None)
            opt_result = next((r for r in optimized_results if r['threads'] == threads), None)
            
            if not orig_result or not opt_result:
                continue
                
            orig_time = orig_result['mean_time_ms']
            opt_time = opt_result['mean_time_ms']
            
            # Calculate speedups
            orig_speedup = baseline_time / orig_time
            opt_speedup = baseline_time / opt_time
            opt_vs_orig_improvement = orig_time / opt_time
            
            # Calculate efficiencies
            orig_efficiency = orig_speedup / threads
            opt_efficiency = opt_speedup / threads
            
            # Statistical significance tests
            orig_vs_seq_tstat, orig_vs_seq_pval = stats.ttest_ind(sequential_result['times'], orig_result['times'])
            opt_vs_seq_tstat, opt_vs_seq_pval = stats.ttest_ind(sequential_result['times'], opt_result['times'])
            opt_vs_orig_tstat, opt_vs_orig_pval = stats.ttest_ind(orig_result['times'], opt_result['times'])
            
            analysis['thread_analysis'].append({
                'threads': threads,
                'original_openmp': {
                    'mean_time_ms': orig_time,
                    'std_time_ms': orig_result['std_time_ms'],
                    'speedup': orig_speedup,
                    'efficiency': orig_efficiency,
                    'efficiency_percentage': orig_efficiency * 100
                },
                'optimized_openmp': {
                    'mean_time_ms': opt_time,
                    'std_time_ms': opt_result['std_time_ms'],
                    'speedup': opt_speedup,
                    'efficiency': opt_efficiency,
                    'efficiency_percentage': opt_efficiency * 100
                },
                'improvement_analysis': {
                    'optimized_vs_original_speedup': opt_vs_orig_improvement,
                    'optimized_vs_original_time_reduction': ((orig_time - opt_time) / orig_time * 100),
                    'statistical_significance_opt_vs_orig': opt_vs_orig_pval < 0.05,
                    'p_value_opt_vs_orig': opt_vs_orig_pval
                },
                'vs_opencv_comparison': {
                    'original_vs_opencv': orig_time / opencv_time,
                    'optimized_vs_opencv': opt_time / opencv_time
                }
            })
        
        return analysis
    
    def generate_comparison_plots(self, analysis):
        """Generate comprehensive comparison plots"""
        filter_name = analysis['filter']
        thread_data = analysis['thread_analysis']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        threads = [d['threads'] for d in thread_data]
        orig_times = [d['original_openmp']['mean_time_ms'] for d in thread_data]
        opt_times = [d['optimized_openmp']['mean_time_ms'] for d in thread_data]
        orig_speedups = [d['original_openmp']['speedup'] for d in thread_data]
        opt_speedups = [d['optimized_openmp']['speedup'] for d in thread_data]
        
        # 1. Execution Time Comparison
        ax1.plot(threads, orig_times, 'o-', linewidth=2, label='Original OpenMP', color='red')
        ax1.plot(threads, opt_times, 's-', linewidth=2, label='Optimized OpenMP', color='green')
        ax1.axhline(y=analysis['baseline_time_ms'], color='blue', linestyle='--', label='Sequential Baseline')
        ax1.axhline(y=analysis['opencv_time_ms'], color='orange', linestyle='--', label='OpenCV Native')
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title(f'Execution Time Comparison - {filter_name.upper()}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')
        
        # 2. Speedup Comparison
        ax2.plot(threads, orig_speedups, 'o-', linewidth=2, label='Original OpenMP', color='red')
        ax2.plot(threads, opt_speedups, 's-', linewidth=2, label='Optimized OpenMP', color='green')
        ax2.plot(threads, threads, '--', alpha=0.7, label='Ideal Speedup', color='black')
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title(f'Speedup Comparison - {filter_name.upper()}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency Comparison
        orig_efficiencies = [d['original_openmp']['efficiency_percentage'] for d in thread_data]
        opt_efficiencies = [d['optimized_openmp']['efficiency_percentage'] for d in thread_data]
        
        ax3.plot(threads, orig_efficiencies, 'o-', linewidth=2, label='Original OpenMP', color='red')
        ax3.plot(threads, opt_efficiencies, 's-', linewidth=2, label='Optimized OpenMP', color='green')
        ax3.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Ideal Efficiency')
        ax3.set_xlabel('Number of Threads')
        ax3.set_ylabel('Parallel Efficiency (%)')
        ax3.set_title(f'Efficiency Comparison - {filter_name.upper()}')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Improvement Factor
        improvements = [d['improvement_analysis']['optimized_vs_original_speedup'] for d in thread_data]
        colors = ['green' if imp > 1 else 'red' for imp in improvements]
        
        bars = ax4.bar(threads, improvements, alpha=0.7, color=colors)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='No Improvement')
        ax4.set_xlabel('Number of Threads')
        ax4.set_ylabel('Optimized vs Original Speedup')
        ax4.set_title(f'OpenMP Optimization Impact - {filter_name.upper()}')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend()
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax4.annotate(f'{imp:.2f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{filter_name}_optimization_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_research_report(self, all_analyses):
        """Generate comprehensive research report"""
        report = {
            'study_metadata': {
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'iterations_per_test': self.iterations,
                'thread_counts_tested': self.thread_counts,
                'filters_analyzed': self.filters,
                'implementations_compared': ['sequential', 'original_openmp', 'optimized_openmp', 'opencv_native'],
                'total_measurements': len(self.filters) * len(self.thread_counts) * self.iterations * 3
            },
            'optimization_summary': {},
            'detailed_results': all_analyses
        }
        
        # Generate optimization summary
        for analysis in all_analyses:
            filter_name = analysis['filter']
            thread_data = analysis['thread_analysis']
            
            if not thread_data:
                continue
                
            # Find best performance for each implementation
            best_orig = min(thread_data, key=lambda x: x['original_openmp']['mean_time_ms'])
            best_opt = min(thread_data, key=lambda x: x['optimized_openmp']['mean_time_ms'])
            
            max_improvement = max([d['improvement_analysis']['optimized_vs_original_speedup'] for d in thread_data])
            avg_improvement = np.mean([d['improvement_analysis']['optimized_vs_original_speedup'] for d in thread_data])
            
            report['optimization_summary'][filter_name] = {
                'best_original_performance': {
                    'threads': best_orig['threads'],
                    'speedup': best_orig['original_openmp']['speedup'],
                    'efficiency': best_orig['original_openmp']['efficiency_percentage']
                },
                'best_optimized_performance': {
                    'threads': best_opt['threads'],
                    'speedup': best_opt['optimized_openmp']['speedup'],
                    'efficiency': best_opt['optimized_openmp']['efficiency_percentage']
                },
                'optimization_impact': {
                    'max_improvement_factor': max_improvement,
                    'average_improvement_factor': avg_improvement,
                    'optimization_successful': max_improvement > 1.1  # 10% improvement threshold
                },
                'opencv_comparison': {
                    'original_vs_opencv': analysis['baseline_time_ms'] / analysis['opencv_time_ms'],
                    'optimized_vs_opencv_best': best_opt['vs_opencv_comparison']['optimized_vs_opencv']
                }
            }
        
        # Save report
        with open(self.results_dir / 'optimization_research_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_complete_optimization_study(self, image_path):
        """Run the complete optimization comparison study"""
        print("🔬 Starting OpenMP Optimization Research Study")
        print(f"📊 Iterations per test: {self.iterations}")
        print(f"🧵 Thread counts: {self.thread_counts}")
        print(f"🎯 Filters: {self.filters}")
        print("🔄 Comparing: Sequential, Original OpenMP, Optimized OpenMP, OpenCV Native")
        print()
        
        all_analyses = []
        
        for filter_type in self.filters:
            print(f"Analyzing {filter_type.upper()} filter...")
            
            # 1. Sequential baseline
            print(f"  Running sequential baseline ({self.iterations} iterations)...")
            sequential_result = self.run_sequential_benchmark(image_path, filter_type)
            
            # 2. OpenCV baseline
            print(f"  Running OpenCV baseline ({self.iterations} iterations)...")
            opencv_result = self.run_opencv_baseline(image_path, filter_type)
            
            # 3. Original OpenMP implementations
            original_results = []
            for threads in self.thread_counts:
                if threads == 1:
                    continue
                print(f"  Running original OpenMP - {threads} threads...")
                orig_result = self.run_original_parallel_benchmark(image_path, filter_type, threads)
                original_results.append(orig_result)
            
            # 4. Optimized OpenMP implementations
            optimized_results = []
            for threads in self.thread_counts:
                if threads == 1:
                    continue
                print(f"  Running optimized OpenMP - {threads} threads...")
                opt_result = self.run_optimized_parallel_benchmark(image_path, filter_type, threads)
                optimized_results.append(opt_result)
            
            # 5. Comprehensive analysis
            analysis = self.calculate_comprehensive_metrics(
                sequential_result, original_results, optimized_results, opencv_result
            )
            all_analyses.append(analysis)
            
            # 6. Generate comparison plots
            self.generate_comparison_plots(analysis)
            
            # 7. Save raw data
            all_results = [sequential_result, opencv_result] + original_results + optimized_results
            df = pd.DataFrame(all_results)
            df.to_csv(self.results_dir / f'{filter_type}_optimization_raw_data.csv', index=False)
            
            print(f"  ✅ {filter_type.upper()} optimization analysis complete")
            
            # Show improvement summary
            if analysis['thread_analysis']:
                best_improvement = max([d['improvement_analysis']['optimized_vs_original_speedup'] 
                                      for d in analysis['thread_analysis']])
                best_opt_speedup = max([d['optimized_openmp']['speedup'] 
                                      for d in analysis['thread_analysis']])
                print(f"     Best optimization improvement: {best_improvement:.2f}x")
                print(f"     Best optimized speedup vs sequential: {best_opt_speedup:.2f}x")
            print()
        
        # Generate final report
        report = self.generate_research_report(all_analyses)
        
        print("🎉 OpenMP Optimization Study Complete!")
        print(f"📁 Results saved to: {self.results_dir}")
        print()
        print("📈 Optimization Summary:")
        for filter_name, summary in report['optimization_summary'].items():
            print(f"  {filter_name.upper()}:")
            print(f"    Original Best Speedup: {summary['best_original_performance']['speedup']:.2f}x")
            print(f"    Optimized Best Speedup: {summary['best_optimized_performance']['speedup']:.2f}x")
            print(f"    Max Improvement: {summary['optimization_impact']['max_improvement_factor']:.2f}x")
            success = "✅ SUCCESS" if summary['optimization_impact']['optimization_successful'] else "❌ NEEDS WORK"
            print(f"    Optimization Status: {success}")
        
        return report

if __name__ == "__main__":
    benchmark = OptimizedOpenMPBenchmark(iterations=50)  # Reduced for faster testing
    benchmark.run_complete_optimization_study("images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg")