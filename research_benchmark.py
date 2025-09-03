#!/usr/bin/env python3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os

class OpenMPResearchBenchmark:
    def __init__(self):
        self.thread_counts = [1, 2, 4, 8]
        self.iterations = 100
        self.results = []
    
    def run_cpp_benchmark(self, image_path, filter_type, threads):
        """Run C++ preprocessing with specific thread count"""
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        
        times = []
        for _ in range(self.iterations):
            start = time.time()
            subprocess.run([
                './bin/preprocess', image_path, 'temp/output.jpg', filter_type
            ], env=env, capture_output=True)
            times.append((time.time() - start) * 1000)  # ms
        
        return {
            'threads': threads,
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times)
        }
    
    def calculate_metrics(self, results_df):
        """Calculate speedup and efficiency"""
        baseline = results_df[results_df['threads'] == 1]['mean_time_ms'].iloc[0]
        
        results_df['speedup'] = baseline / results_df['mean_time_ms']
        results_df['efficiency'] = results_df['speedup'] / results_df['threads']
        results_df['fps'] = 1000 / results_df['mean_time_ms']
        
        return results_df
    
    def generate_plots(self, df, filter_name):
        """Generate academic-quality plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Execution Time
        ax1.plot(df['threads'], df['mean_time_ms'], 'o-', linewidth=2)
        ax1.set_xlabel('Thread Count')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title(f'Execution Time - {filter_name}')
        ax1.grid(True, alpha=0.3)
        
        # Speedup
        ax2.plot(df['threads'], df['speedup'], 'o-', linewidth=2, label='Actual')
        ax2.plot(df['threads'], df['threads'], '--', alpha=0.7, label='Ideal')
        ax2.set_xlabel('Thread Count')
        ax2.set_ylabel('Speedup')
        ax2.set_title(f'Speedup - {filter_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Efficiency
        ax3.plot(df['threads'], df['efficiency'], 'o-', linewidth=2)
        ax3.set_xlabel('Thread Count')
        ax3.set_ylabel('Parallel Efficiency')
        ax3.set_title(f'Parallel Efficiency - {filter_name}')
        ax3.grid(True, alpha=0.3)
        
        # FPS
        ax4.plot(df['threads'], df['fps'], 'o-', linewidth=2)
        ax4.set_xlabel('Thread Count')
        ax4.set_ylabel('Frames Per Second')
        ax4.set_title(f'Throughput - {filter_name}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/openmp_analysis_{filter_name}.png', dpi=300, bbox_inches='tight')
        
    def run_full_benchmark(self, image_path):
        """Run complete academic benchmark"""
        filters = ['blur', 'sharpen', 'denoise', 'clahe', 'edge']
        
        os.makedirs('results', exist_ok=True)
        
        for filter_type in filters:
            print(f"Benchmarking {filter_type}...")
            
            filter_results = []
            for threads in self.thread_counts:
                result = self.run_cpp_benchmark(image_path, filter_type, threads)
                result['filter'] = filter_type
                filter_results.append(result)
            
            df = pd.DataFrame(filter_results)
            df = self.calculate_metrics(df)
            
            # Save data
            df.to_csv(f'results/openmp_data_{filter_type}.csv', index=False)
            
            # Generate plots
            self.generate_plots(df, filter_type)
            
            # Print summary
            print(f"{filter_type.upper()} Results:")
            print(f"  Max Speedup: {df['speedup'].max():.2f}x")
            print(f"  Efficiency at 8 threads: {df[df['threads']==8]['efficiency'].iloc[0]:.2f}")
            print()

if __name__ == "__main__":
    benchmark = OpenMPResearchBenchmark()
    benchmark.run_full_benchmark("images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg")
