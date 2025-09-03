#!/usr/bin/env python3
import cv2
import numpy as np
import time
import subprocess

# Load test image
img = cv2.imread('images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg')
if img is None:
    print("Test image not found")
    exit(1)

print(f"Image size: {img.shape}")

# Test Python/OpenCV
start = time.time()
for _ in range(100):
    result = cv2.GaussianBlur(img, (5, 5), 1.0)
python_time = time.time() - start

print(f"Python/OpenCV (100 iterations): {python_time:.3f}s")
print(f"Per iteration: {python_time/100*1000:.1f}ms")

# Test your C++ implementation
start = time.time()
for _ in range(100):
    subprocess.run(['./bin/preprocess', 'images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg', 'temp/cpp_test.jpg', 'blur'], 
                   capture_output=True)
cpp_time = time.time() - start

print(f"Your C++ implementation (100 iterations): {cpp_time:.3f}s")
print(f"Per iteration: {cpp_time/100*1000:.1f}ms")
print(f"C++ is {cpp_time/python_time:.1f}x slower due to process overhead")
