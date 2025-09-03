#!/usr/bin/env python3
import cv2
import numpy as np
import time

# Create test image
img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
print(f"Test image size: {img.shape}")

# Test OpenCV's optimized Gaussian blur
iterations = 1000
start = time.time()
for _ in range(iterations):
    result = cv2.GaussianBlur(img, (5, 5), 1.0)
opencv_time = time.time() - start

# Test manual convolution (similar to your C++ approach)
kernel = cv2.getGaussianKernel(5, 1.0)
kernel_2d = kernel @ kernel.T

start = time.time()
for _ in range(iterations):
    result = cv2.filter2D(img, -1, kernel_2d)
manual_time = time.time() - start

print(f"\nResults ({iterations} iterations):")
print(f"OpenCV GaussianBlur: {opencv_time:.3f}s ({opencv_time/iterations*1000:.2f}ms per image)")
print(f"Manual convolution:  {manual_time:.3f}s ({manual_time/iterations*1000:.2f}ms per image)")
print(f"Manual is {manual_time/opencv_time:.1f}x slower")

# Test different image sizes
sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
print(f"\nPerformance scaling:")
for h, w in sizes:
    test_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    start = time.time()
    cv2.GaussianBlur(test_img, (5, 5), 1.0)
    opencv_single = time.time() - start
    
    print(f"{w}x{h}: {opencv_single*1000:.2f}ms")
