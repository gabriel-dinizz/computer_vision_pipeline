#!/usr/bin/env python3
"""
Quick performance test to validate OpenMP optimizations
"""

import subprocess
import time
import statistics

def time_command(cmd, iterations=10):
    """Time a command over multiple iterations"""
    times = []
    for _ in range(iterations):
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
    
    if times:
        return {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'count': len(times)
        }
    return None

def main():
    image_path = "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg"
    iterations = 10
    
    print("🚀 Quick OpenMP Performance Validation")
    print(f"📊 Testing with {iterations} iterations each")
    print("=" * 50)
    
    # Test different filters
    filters = ['blur', 'sharpen', 'denoise']
    
    for filter_type in filters:
        print(f"\n🎯 Testing {filter_type.upper()} filter:")
        
        # Test original implementation
        original_cmd = ['./bin/preprocess', image_path, f'temp/test_orig_{filter_type}.jpg', filter_type]
        original_stats = time_command(original_cmd, iterations)
        
        # Test optimized implementation  
        optimized_cmd = ['./bin/preprocess_optimized', image_path, f'temp/test_opt_{filter_type}.jpg', filter_type]
        optimized_stats = time_command(optimized_cmd, iterations)
        
        if original_stats and optimized_stats:
            speedup = original_stats['mean'] / optimized_stats['mean']
            improvement = ((original_stats['mean'] - optimized_stats['mean']) / original_stats['mean']) * 100
            
            print(f"   Original:  {original_stats['mean']:.1f}ms ±{original_stats['std']:.1f}ms")
            print(f"   Optimized: {optimized_stats['mean']:.1f}ms ±{optimized_stats['std']:.1f}ms")
            print(f"   Speedup:   {speedup:.2f}x")
            print(f"   Improvement: {improvement:+.1f}%")
            
            if speedup > 1.1:
                print("   Status: ✅ IMPROVEMENT ACHIEVED")
            elif speedup > 0.9:
                print("   Status: ≈ COMPARABLE PERFORMANCE")  
            else:
                print("   Status: ❌ NEEDS MORE OPTIMIZATION")
        else:
            print("   Status: ❌ FAILED TO RUN TESTS")
    
    print("\n" + "=" * 50)
    print("📈 Performance validation complete!")

if __name__ == "__main__":
    main()