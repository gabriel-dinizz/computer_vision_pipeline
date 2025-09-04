#!/usr/bin/env python3
"""
Simple test of the research benchmark framework
"""

import subprocess
import time
import os

def test_baselines():
    """Test all three baselines with a simple run"""
    test_image = "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg"
    
    if not os.path.exists(test_image):
        print("Test image not found!")
        return False
    
    os.makedirs("temp", exist_ok=True)
    
    tests = [
        ("Parallel OpenMP", ["./bin/preprocess", test_image, "temp/test_parallel.jpg", "blur"]),
        ("Sequential", ["./bin/sequential_baseline", test_image, "temp/test_sequential.jpg", "blur"]),
        ("OpenCV Native", ["./bin/opencv_baseline", test_image, "temp/test_opencv.jpg", "blur"])
    ]
    
    results = {}
    
    for name, cmd in tests:
        print(f"\n🧪 Testing {name}...")
        try:
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            end = time.time()
            
            if result.returncode == 0:
                runtime = (end - start) * 1000  # ms
                results[name] = runtime
                print(f"✅ {name}: {runtime:.2f} ms")
            else:
                print(f"❌ {name} failed: {result.stderr}")
                results[name] = None
        except subprocess.TimeoutExpired:
            print(f"⏱️ {name} timed out")
            results[name] = None
        except Exception as e:
            print(f"💥 {name} error: {e}")
            results[name] = None
    
    print("\n📊 Results Summary:")
    for name, runtime in results.items():
        if runtime is not None:
            print(f"  {name}: {runtime:.2f} ms")
        else:
            print(f"  {name}: FAILED")
    
    # Calculate speedups if all succeeded
    if all(v is not None for v in results.values()):
        seq_time = results["Sequential"]
        par_time = results["Parallel OpenMP"]
        opencv_time = results["OpenCV Native"]
        
        print(f"\n🚀 Performance Analysis:")
        print(f"  OpenMP Speedup: {seq_time/par_time:.2f}x")
        print(f"  vs OpenCV: {opencv_time/par_time:.2f}x")
        
    return True

if __name__ == "__main__":
    print("🔬 Simple Research Benchmark Test")
    test_baselines()
