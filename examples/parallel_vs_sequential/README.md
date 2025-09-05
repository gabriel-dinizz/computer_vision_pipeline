# Parallel vs Sequential Filter Examples

This folder contains side-by-side comparisons of parallel (OpenMP) and sequential implementations of image filters to demonstrate performance differences.

## Files

- `gaussian_blur_comparison.cpp` - Gaussian blur: parallel vs sequential
- `sharpen_comparison.cpp` - Unsharp masking: parallel vs sequential
- `bilateral_comparison.cpp` - Bilateral filter: parallel vs sequential
- `benchmark.cpp` - Performance benchmarking tool
- `Makefile` - Build system

## Usage

```bash
# Build all examples
make all

# Run individual comparisons
./gaussian_blur_test image.jpg
./sharpen_test image.jpg
./bilateral_test image.jpg

# Run full benchmark
./benchmark image.jpg
```

## Expected Results

Typical performance improvements with OpenMP (8 threads):
- Gaussian Blur: 3-5x speedup
- Unsharp Masking: 4-6x speedup
- Bilateral Filter: 2-4x speedup
