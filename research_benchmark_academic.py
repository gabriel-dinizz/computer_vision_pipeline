#!/usr/bin/env python3
"""
Academic Research Benchmark for OpenMP Performance Analysis
Implements rigorous statistical methodology for TCC research
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

class OpenMPResearchBenchmark:
    def __init__(self, iterations=100):
        self.thread_counts = [1, 2, 4, 8, 16]
        self.iterations = iterations
        self.filters = ['blur', 'sharpen', 'denoise', 'clahe', 'edge']
        self.results = []
        
        # Create results directory
        self.results_dir = Path('research_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup controlled environment for accurate benchmarking
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
                    cmd = ["./bin/preprocess", warmup_image, "/tmp/warmup_output.jpg", "blur"]
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
            # Use sequential baseline implementation
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
    
    def run_parallel_benchmark(self, image_path, filter_type, threads):
        """Run parallel OpenMP benchmark with specific thread count"""
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        
        times = []
        
        for i in range(self.iterations):
            start = time.time()
            result = subprocess.run([
                './bin/preprocess', image_path, 'temp/par_output.jpg', filter_type
            ], env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                continue
                
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        return {
            'type': 'parallel',
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
    
    def calculate_research_metrics(self, sequential_result, parallel_results):
        """Calculate academic research metrics"""
        baseline_time = sequential_result['mean_time_ms']
        
        analysis = {
            'filter': sequential_result['filter'],
            'baseline_time_ms': baseline_time,
            'baseline_std_ms': sequential_result['std_time_ms'],
            'thread_analysis': []
        }
        
        for par_result in parallel_results:
            threads = par_result['threads']
            par_time = par_result['mean_time_ms']
            
            speedup = baseline_time / par_time
            efficiency = speedup / threads
            
            # Statistical significance test (t-test)
            seq_times = sequential_result['times']
            par_times = par_result['times']
            
            t_stat, p_value = stats.ttest_ind(seq_times, par_times)
            
            analysis['thread_analysis'].append({
                'threads': threads,
                'mean_time_ms': par_time,
                'std_time_ms': par_result['std_time_ms'],
                'speedup': speedup,
                'efficiency': efficiency,
                'theoretical_max_speedup': threads,
                'efficiency_percentage': (efficiency * 100),
                'time_reduction_percentage': ((baseline_time - par_time) / baseline_time * 100),
                'statistical_significance': p_value < 0.05,
                'p_value': p_value,
                't_statistic': t_stat
            })
        
        return analysis
    
    def generate_academic_plots(self, analysis):
        """Generate publication-quality plots"""
        filter_name = analysis['filter']
        thread_data = analysis['thread_analysis']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        threads = [d['threads'] for d in thread_data]
        times = [d['mean_time_ms'] for d in thread_data]
        speedups = [d['speedup'] for d in thread_data]
        efficiencies = [d['efficiency'] for d in thread_data]
        
        # 1. Execution Time vs Threads
        ax1.errorbar(threads, times, 
                    yerr=[d['std_time_ms'] for d in thread_data],
                    marker='o', linewidth=2, capsize=5)
        ax1.axhline(y=analysis['baseline_time_ms'], color='red', 
                   linestyle='--', label='Sequential Baseline')
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title(f'Execution Time - {filter_name.upper()}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')
        
        # 2. Speedup vs Threads
        ax2.plot(threads, speedups, 'o-', linewidth=2, label='Actual Speedup')
        ax2.plot(threads, threads, '--', alpha=0.7, label='Ideal Speedup')
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title(f'Speedup Analysis - {filter_name.upper()}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Parallel Efficiency
        ax3.plot(threads, [e*100 for e in efficiencies], 'o-', linewidth=2, color='green')
        ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal Efficiency')
        ax3.set_xlabel('Number of Threads')
        ax3.set_ylabel('Parallel Efficiency (%)')
        ax3.set_title(f'Parallel Efficiency - {filter_name.upper()}')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Performance Improvement
        improvements = [d['time_reduction_percentage'] for d in thread_data]
        ax4.bar(threads, improvements, alpha=0.7, color='orange')
        ax4.set_xlabel('Number of Threads')
        ax4.set_ylabel('Performance Improvement (%)')
        ax4.set_title(f'Performance Improvement - {filter_name.upper()}')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{filter_name}_academic_analysis.png', 
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
                'total_measurements': len(self.filters) * len(self.thread_counts) * self.iterations
            },
            'summary_findings': {},
            'detailed_results': all_analyses
        }
        
        # Generate summary findings
        for analysis in all_analyses:
            filter_name = analysis['filter']
            thread_data = analysis['thread_analysis']
            
            max_speedup = max([d['speedup'] for d in thread_data])
            max_speedup_threads = [d['threads'] for d in thread_data if d['speedup'] == max_speedup][0]
            efficiency_at_8_threads = [d['efficiency'] for d in thread_data if d['threads'] == 8]
            efficiency_at_8 = efficiency_at_8_threads[0] if efficiency_at_8_threads else 0
            
            report['summary_findings'][filter_name] = {
                'max_speedup': max_speedup,
                'optimal_thread_count': max_speedup_threads,
                'efficiency_at_8_threads': efficiency_at_8,
                'recommended_for_production': efficiency_at_8 > 0.5
            }
        
        # Save report
        with open(self.results_dir / 'research_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_complete_research_study(self, image_path):
        """Run the complete academic research study"""
        print("🔬 Starting Academic Research Study")
        print(f"📊 Iterations per test: {self.iterations}")
        print(f"🧵 Thread counts: {self.thread_counts}")
        print(f"🎯 Filters: {self.filters}")
        print()
        
        all_analyses = []
        
        for filter_type in self.filters:
            print(f"Analyzing {filter_type.upper()} filter...")
            
            # 1. Sequential baseline
            print(f"  Running sequential baseline ({self.iterations} iterations)...")
            sequential_result = self.run_sequential_benchmark(image_path, filter_type)
            
            # 2. Parallel implementations
            parallel_results = []
            for threads in self.thread_counts:
                if threads == 1:
                    continue  # Skip, we have sequential baseline
                    
                print(f"  Running parallel benchmark - {threads} threads...")
                par_result = self.run_parallel_benchmark(image_path, filter_type, threads)
                parallel_results.append(par_result)
            
            # 3. Analysis
            analysis = self.calculate_research_metrics(sequential_result, parallel_results)
            all_analyses.append(analysis)
            
            # 4. Generate plots
            self.generate_academic_plots(analysis)
            
            # 5. Save raw data
            df = pd.DataFrame([sequential_result] + parallel_results)
            df.to_csv(self.results_dir / f'{filter_type}_raw_data.csv', index=False)
            
            print(f"  ✅ {filter_type.upper()} analysis complete")
            print(f"     Max speedup: {max([d['speedup'] for d in analysis['thread_analysis']]):.2f}x")
            print()
        
        # Generate final report
        report = self.generate_research_report(all_analyses)
        
        print("🎉 Research Study Complete!")
        print(f"📁 Results saved to: {self.results_dir}")
        print()
        print("📈 Summary Findings:")
        for filter_name, findings in report['summary_findings'].items():
            print(f"  {filter_name.upper()}:")
            print(f"    Max Speedup: {findings['max_speedup']:.2f}x")
            print(f"    Optimal Threads: {findings['optimal_thread_count']}")
            print(f"    8-Thread Efficiency: {findings['efficiency_at_8_threads']:.1%}")
            print(f"    Production Ready: {'✅' if findings['recommended_for_production'] else '❌'}")
        
        return report

if __name__ == "__main__":
    benchmark = OpenMPResearchBenchmark(iterations=100)
    benchmark.run_complete_research_study("images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg")
