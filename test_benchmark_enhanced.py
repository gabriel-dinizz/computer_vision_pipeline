#!/usr/bin/env python3
"""
Enhanced benchmark to demonstrate OpenMP benefits
"""

import subprocess
import time
import os
from PIL import Image

def create_large_test_image():
    """Create a larger test image for better parallel performance"""
    original = "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg"
    if not os.path.exists(original):
        return None
    
    # Create 4x larger image (2x scale in each dimension)
    large_image = "temp/large_test_image.jpg"
    os.makedirs("temp", exist_ok=True)
    
    try:
        with Image.open(original) as img:
            # Scale up to ~2M pixels (from ~500k)
            new_size = (img.width * 2, img.height * 2)
            large_img = img.resize(new_size, Image.Resampling.LANCZOS)
            large_img.save(large_image, quality=95)
            print(f"Created large test image: {new_size[0]}x{new_size[1]} pixels")
            return large_image
    except Exception as e:
        print(f"Failed to create large image: {e}")
        return None

def test_demanding_filters():
    """Test with more computationally intensive filters"""
    
    # Test with both original and large images
    test_images = ["images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg"]
    
    large_image = create_large_test_image()
    if large_image:
        test_images.append(large_image)
    
    # Test more demanding filters
    filters = ["denoise", "clahe", "sharpen"]  # More compute-intensive than blur
    
    results = {}
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            continue
            
        image_name = "Original" if "Toyota" in image_path else "Large (2x)"
        results[image_name] = {}
        
        print(f"\n📸 Testing with {image_name} image...")
        
        for filter_type in filters:
            print(f"\n🎯 Filter: {filter_type.upper()}")
            
            tests = [
                ("Sequential", ["./bin/sequential_baseline", image_path, f"temp/seq_{filter_type}.jpg", filter_type]),
                ("OpenMP Parallel", ["./bin/preprocess", image_path, f"temp/par_{filter_type}.jpg", filter_type]),
                ("OpenCV Native", ["./bin/opencv_baseline", image_path, f"temp/cv_{filter_type}.jpg", filter_type])
            ]
            
            filter_results = {}
            
            for name, cmd in tests:
                try:
                    # Set OpenMP threads for parallel version
                    env = os.environ.copy()
                    if "preprocess" in cmd[0]:
                        env['OMP_NUM_THREADS'] = '4'
                    
                    # Run multiple times for better accuracy
                    times = []
                    for _ in range(3):
                        start = time.time()
                        result = subprocess.run(cmd, capture_output=True, text=True, 
                                             timeout=30, env=env)
                        end = time.time()
                        
                        if result.returncode == 0:
                            times.append((end - start) * 1000)  # ms
                        else:
                            print(f"   ❌ {name} failed: {result.stderr}")
                            break
                    
                    if times:
                        avg_time = sum(times) / len(times)
                        filter_results[name] = avg_time
                        print(f"   ✅ {name}: {avg_time:.2f} ms")
                    else:
                        filter_results[name] = None
                        
                except Exception as e:
                    print(f"   💥 {name} error: {e}")
                    filter_results[name] = None
            
            # Calculate speedups
            if all(v is not None for v in filter_results.values()):
                seq_time = filter_results["Sequential"]
                par_time = filter_results["OpenMP Parallel"]
                cv_time = filter_results["OpenCV Native"]
                
                speedup = seq_time / par_time
                vs_opencv = cv_time / par_time
                
                print(f"   🚀 OpenMP Speedup: {speedup:.2f}x")
                print(f"   📊 vs OpenCV: {vs_opencv:.2f}x")
                
                if speedup > 1.0:
                    print(f"   ✅ Parallel is {speedup:.2f}x FASTER!")
                else:
                    print(f"   ⚠️  Parallel is {1/speedup:.2f}x slower (overhead dominates)")
            
            results[image_name][filter_type] = filter_results
    
    # Summary
    print(f"\n" + "="*60)
    print("🎯 PARALLEL PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    for image_name, image_results in results.items():
        print(f"\n📸 {image_name}:")
        for filter_type, filter_results in image_results.items():
            if all(v is not None for v in filter_results.values()):
                speedup = filter_results["Sequential"] / filter_results["OpenMP Parallel"]
                status = "🚀 FASTER" if speedup > 1.0 else "⚠️ SLOWER"
                print(f"   {filter_type.upper()}: {speedup:.2f}x {status}")

if __name__ == "__main__":
    print("🔬 Enhanced Parallel Performance Analysis")
    print("Testing with larger images and demanding filters...\n")
    test_demanding_filters()
