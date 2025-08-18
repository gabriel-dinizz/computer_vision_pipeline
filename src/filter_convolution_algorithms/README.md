# Filter Convolution Algorithms

This directory contains implementations of various convolution-based image filtering algorithms for computer vision applications.

## Available Algorithms

### 1. Gaussian Blur Filter (`gaussian_blur.cpp`)

A comprehensive implementation of Gaussian blur filtering using convolution. This algorithm is fundamental in computer vision for:
- Noise reduction
- Image smoothing
- Preprocessing for edge detection
- Creating image pyramids

#### Features:
- **Standard Convolution**: Full 2D convolution implementation
- **Separable Convolution**: Optimized implementation using separable kernels (more efficient)
- **Configurable Parameters**: Adjustable kernel size and sigma (standard deviation)
- **Automatic Kernel Generation**: Mathematically correct Gaussian kernel generation
- **Multi-format Support**: Handles both grayscale and color images

#### Usage:
```cpp
#include "gaussian_blur.h"

// Create filter with kernel size 5 and sigma 1.0
GaussianBlurFilter filter(5, 1.0);

// Apply blur using standard convolution
cv::Mat blurred = filter.applyConvolution(inputImage);

// Apply blur using separable convolution (faster)
cv::Mat blurred_fast = filter.applySeparableConvolution(inputImage);
```

#### Mathematical Background:
The Gaussian kernel is generated using the 2D Gaussian function:
```
G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
```

Where:
- `σ` (sigma) controls the amount of blur
- Larger σ values create more blur
- The kernel is normalized so all values sum to 1

## Building and Running

### Prerequisites:
- OpenCV library (version 3.0 or higher)
- C++ compiler with C++11 support
- pkg-config for OpenCV

### Build Instructions:

1. **Build all algorithms:**
   ```bash
   make all
   ```

2. **Build specific algorithm:**
   ```bash
   make gaussian_blur
   ```

3. **Run tests:**
   ```bash
   make test
   ```

4. **Clean build files:**
   ```bash
   make clean
   ```

### Installation of OpenCV (if needed):

**macOS (using Homebrew):**
```bash
brew install opencv
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libopencv-dev
```

**Build from source:**
Follow the official OpenCV installation guide at: https://opencv.org/get-started/

## Algorithm Performance

### Gaussian Blur:
- **Standard Convolution**: O(n²m²k²) where n,m are image dimensions and k is kernel size
- **Separable Convolution**: O(n²m²k) - significantly faster for larger kernels
- **Memory Usage**: Minimal additional memory requirements

### Recommended Parameters:
- **Small blur**: kernel size 3-5, sigma 0.5-1.0
- **Medium blur**: kernel size 7-9, sigma 1.0-2.0
- **Heavy blur**: kernel size 11+, sigma 2.0+

## Future Algorithms

Planned implementations:
- [ ] Sobel Edge Detection
- [ ] Laplacian Filter
- [ ] Box Blur
- [ ] Motion Blur
- [ ] Bilateral Filter
- [ ] Custom Convolution Framework

## Testing

Each algorithm includes test cases that:
1. Generate synthetic test images
2. Apply the filter
3. Save results for visual inspection
4. Validate numerical accuracy

## Contributing

When adding new convolution algorithms:
1. Follow the existing code structure
2. Include both header (.h) and implementation (.cpp) files
3. Add appropriate test cases
4. Update this README
5. Add build rules to the Makefile

## References

- Gonzalez, R. C., & Woods, R. E. (2017). Digital Image Processing (4th ed.)
- OpenCV Documentation: https://docs.opencv.org/
- Computer Vision: Algorithms and Applications by Richard Szeliski
