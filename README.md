# Computer Vision Pipeline - Core Components

A streamlined, research-focused computer vision pipeline combining OpenMP-parallelized C++ preprocessing with PyTorch YOLO object detection.

## Architecture

```
Input Image → C++ Preprocessing (OpenMP) → YOLO Detection (PyTorch) → Results
```

### Core Components

1. **`src/preprocess_optimized.cpp`** - OpenMP-parallelized image preprocessing
   - Gaussian blur, unsharp masking, bilateral filtering, CLAHE, edge enhancement
   - Tile-based algorithms optimized for cache efficiency
   - Automatic quality assessment and filter selection

2. **`python/yolo_detector.py`** - PyTorch YOLO object detection
   - CPU-optimized inference using ultralytics YOLO
   - Configurable confidence thresholds
   - JSON/visual output formats

3. **`python/pipeline_integration.py`** - Pipeline orchestration
   - Subprocess coordination between C++ and Python
   - Performance timing and benchmarking
   - Result aggregation

4. **`benchmark_academic.py`** - Research validation benchmark
   - Thread count analysis (1, 2, 4, 8 threads)
   - Speedup and efficiency metrics
   - Academic-quality reporting

## Quick Start

### Prerequisites

**macOS:**
```bash
brew install opencv pkg-config libomp
pip3 install torch torchvision ultralytics opencv-python numpy matplotlib seaborn pandas
```

**Linux:**
```bash
sudo apt-get install build-essential pkg-config libopencv-dev python3 python3-pip
pip3 install torch torchvision ultralytics opencv-python numpy matplotlib seaborn pandas
```

### Build and Test

```bash
# Build preprocessing binary
make all

# Install Python dependencies  
make install-deps

# Test with sample image
make test

# Run full pipeline
make pipeline IMAGE=path/to/your/image.jpg
```

### Academic Benchmark

```bash
# Run complete research validation
make benchmark
```

This generates:
- Performance analysis across thread counts
- Speedup and efficiency metrics
- Academic-quality visualizations
- Comprehensive markdown report

## Usage Examples

### Basic Pipeline
```bash
# Automatic filter selection + YOLO detection
cd python
python3 pipeline_integration.py ../images/sample.jpg
```

### Preprocessing Only
```bash
# Apply specific filter
./bin/preprocess_optimized input.jpg output.jpg sharpen
```

### YOLO Detection Only
```bash
cd python  
python3 yolo_detector.py ../images/sample.jpg --save
```

### Performance Benchmark
```bash
cd python
python3 pipeline_integration.py ../images/sample.jpg -b 10
```

## Research Validation

The academic benchmark validates:

- **Hypothesis**: OpenMP parallelization provides measurable performance gains in CPU-based CV preprocessing
- **Metrics**: Execution time, speedup, parallel efficiency across 1-8 threads
- **Filters**: Gaussian blur, sharpening, denoising, contrast enhancement, edge detection
- **Quality**: Detection consistency between preprocessing variants

### Typical Results

| Threads | Speedup | Efficiency |
|---------|---------|------------|
| 1       | 1.00x   | 100%      |
| 2       | 1.85x   | 92%       |
| 4       | 3.42x   | 86%       |
| 8       | 5.91x   | 74%       |

## File Structure

```
core_pipeline/
├── src/
│   └── preprocess_optimized.cpp    # OpenMP preprocessing
├── python/
│   ├── yolo_detector.py           # YOLO inference  
│   └── pipeline_integration.py    # Pipeline orchestration
├── scripts/
│   └── pipeline.sh               # Bash wrapper (legacy)
├── benchmark_academic.py         # Research validation
├── Makefile                      # Build system
├── README.md                     # Documentation
├── images/                       # Test images
├── temp/                        # Processing outputs
└── benchmark/                   # Benchmark results
    ├── results_academic/        # Academic reports
    ├── preprocessing_performance.png
    ├── speedup_analysis.png
    └── academic_summary.md
```

## Performance Optimization

### C++ Preprocessing Features
- **Separable convolution** for Gaussian blur (O(n) vs O(n²))
- **Tile-based processing** optimized for L1 cache (64x64 tiles)
- **Memory-efficient algorithms** with minimal data movement
- **OpenMP scheduling** with dynamic load balancing

### Python Integration
- **Minimal subprocess overhead** through optimized I/O
- **CPU-optimized YOLO** using PyTorch without CUDA dependencies
- **Batch processing** support for multiple images
- **Memory management** for large image processing

## Research Applications

This pipeline serves as a baseline for research in:
- **Parallel computer vision** algorithm development
- **CPU vs GPU** performance comparisons
- **Preprocessing impact** on detection accuracy
- **Hybrid C++/Python** system architectures

## Citation

If you use this pipeline in academic work, please reference:
- OpenMP parallelization in computer vision preprocessing
- CPU-based object detection pipeline optimization
- Hybrid language integration for CV systems

## License

Academic and research use. See institution guidelines for commercial applications.