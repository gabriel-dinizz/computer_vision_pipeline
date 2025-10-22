# ============================================================================
# Computer Vision Pipeline Makefile
# ============================================================================
# Purpose: Build and run OpenMP-optimized image preprocessing + YOLO detection
# Author:  TCC Project - Computer Vision Pipeline with Parallel Processing
# ============================================================================

# ============================================================================
# PLATFORM DETECTION AND COMPILER CONFIGURATION
# ============================================================================
# Detect operating system to configure platform-specific settings
UNAME_S := $(shell uname -s)

# macOS Configuration (Darwin)
# - Uses clang++ compiler with Homebrew-installed libraries
# - Requires special flags for OpenMP support on macOS
ifeq ($(UNAME_S),Darwin)
    CXX := clang++
    BREW_PREFIX := $(shell brew --prefix)
    # macOS requires -Xpreprocessor flag for OpenMP
    OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
    # Set runtime library path to find libomp.dylib at runtime
    OMP_LIBS   := -L$(BREW_PREFIX)/opt/libomp/lib -lomp -Wl,-rpath,$(BREW_PREFIX)/opt/libomp/lib
else
    # Linux Configuration
    # - Uses g++ compiler with native OpenMP support
    CXX := g++
    OMP_CFLAGS := -fopenmp
    OMP_LIBS   := -fopenmp
endif

# ============================================================================
# OPENCV CONFIGURATION
# ============================================================================
# Auto-detect OpenCV installation via pkg-config
# Tries opencv4 first, falls back to opencv for compatibility
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# ============================================================================
# COMPILER FLAGS
# ============================================================================
# -std=c++17    : Use C++17 standard features
# -O3           : Maximum optimization for performance
# -Wall -Wextra : Enable comprehensive warnings for code quality
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra $(OPENCV_CFLAGS) $(OMP_CFLAGS)
LDFLAGS  := $(OPENCV_LIBS) $(OMP_LIBS)

# ============================================================================
# PROJECT DIRECTORY STRUCTURE
# ============================================================================
SRC_DIR := src# C++ source files
BIN_DIR := bin# Compiled binaries
TEMP_DIR := temp# Temporary processing outputs
PYTHON_DIR := python# Python YOLO detection scripts
BENCHMARK_DIR := benchmark# Academic benchmark results

# ============================================================================
# BUILD TARGETS
# ============================================================================
PREPROCESS_SRC := $(SRC_DIR)/preprocess_optimized.cpp
PREPROCESS_BIN := $(BIN_DIR)/preprocess_optimized

# Declare phony targets (not actual files)
.PHONY: all clean test benchmark install-deps pipeline help

# ============================================================================
# TARGET: all (default)
# ============================================================================
# Purpose: Build the C++ preprocessing binary with OpenMP optimizations
# Usage:   make
#          make all
# Output:  bin/preprocess_optimized executable
all: $(PREPROCESS_BIN)

# Main build rule for preprocessing binary
# Dependencies: src/preprocess_optimized.cpp
# Creates optimized binary with full compiler optimizations (-O3)
$(PREPROCESS_BIN): $(PREPROCESS_SRC)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built optimized preprocessing pipeline: $@"

# ============================================================================
# TARGET: install-deps
# ============================================================================
# Purpose: Install all required Python dependencies for YOLO detection
# Usage:   make install-deps
# Installs:
#   - PyTorch (torch, torchvision, torchaudio): Deep learning framework
#   - Ultralytics: YOLOv5/YOLOv8 implementation
#   - OpenCV: Python bindings for image processing
#   - NumPy: Numerical computing
#   - Matplotlib, Seaborn: Visualization for benchmarks
#   - Pandas: Data analysis for benchmark results
install-deps:
	@echo "Installing Python dependencies..."
	pip3 install torch torchvision torchaudio ultralytics opencv-python numpy matplotlib seaborn pandas
	@echo "✓ Dependencies installed"

# ============================================================================
# TARGET: test
# ============================================================================
# Purpose: Quick test of preprocessing pipeline with sample image
# Usage:   make test
# Input:   Expects images/sample.jpg to exist
# Output:  temp/test_output.jpg
# Note:    Uses 'auto' mode to automatically select best filter
test: $(PREPROCESS_BIN)
	@echo "Testing preprocessing pipeline..."
	@mkdir -p $(TEMP_DIR)
	@if [ -f "images/sample.jpg" ]; then \
		echo "Testing with sample image..."; \
		./$(PREPROCESS_BIN) images/sample.jpg $(TEMP_DIR)/test_output.jpg auto; \
		echo "✓ Test completed - output: $(TEMP_DIR)/test_output.jpg"; \
	else \
		echo "Please add a sample.jpg to the images/ directory"; \
	fi

# ============================================================================
# TARGET: pipeline
# ============================================================================
# Purpose: Run complete end-to-end computer vision pipeline
# Usage:   make pipeline IMAGE=path/to/image.jpg
#          OMP_NUM_THREADS=4 make pipeline IMAGE=images/times.jpg
# Steps:
#   1. C++ preprocessing with OpenMP parallelization (auto filter selection)
#   2. YOLO object detection using PyTorch
# Output:
#   - temp/preprocessed.jpg: Preprocessed image
#   - temp/detection_results/: Detection results and annotated images
# Options:
#   - Set OMP_NUM_THREADS to control parallelization (default: max cores)
#   - IMAGE parameter is required
pipeline: $(PREPROCESS_BIN)
	@echo "Running full computer vision pipeline..."
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make pipeline IMAGE=path/to/image.jpg"; \
		exit 1; \
	fi
	@echo "1. Preprocessing image..."
	@mkdir -p $(TEMP_DIR)
	./$(PREPROCESS_BIN) $(IMAGE) $(TEMP_DIR)/preprocessed.jpg auto
	@echo "2. Running YOLO detection..."
	cd $(PYTHON_DIR) && python3 yolo_detector.py ../$(TEMP_DIR)/preprocessed.jpg --save -o ../$(TEMP_DIR)/detection_results
	@echo "✓ Pipeline completed - results in $(TEMP_DIR)/detection_results/"

# ============================================================================
# TARGET: benchmark
# ============================================================================
# Purpose: Run comprehensive academic benchmark for TCC/research validation
# Usage:   make benchmark
#          make benchmark IMAGE=../images/times.jpg
# Tests:
#   - All preprocessing filters (blur, sharpen, denoise, clahe, edge)
#   - Multiple thread counts (1, 2, 4, 8)
#   - Statistical analysis (mean, median, std dev)
#   - Speedup and parallel efficiency calculations
# Output:
#   - benchmark/results_academic/complete_results.json: Raw data
#   - benchmark/results_academic/*.png: Performance graphs
#   - benchmark/results_academic/academic_summary.md: Research report
# Note: Each configuration runs 10 iterations with 3 warmup runs
benchmark: $(PREPROCESS_BIN)
	@echo "Running academic benchmark..."
	@mkdir -p $(BENCHMARK_DIR)
	cd $(BENCHMARK_DIR) && python3 benchmark_academic.py $(if $(IMAGE),--image $(IMAGE),)
	@echo "✓ Benchmark completed - results in $(BENCHMARK_DIR)/"

# ============================================================================
# TARGET: clean
# ============================================================================
# Purpose: Remove all build artifacts and temporary files
# Usage:   make clean
# Removes:
#   - bin/: Compiled binaries
#   - temp/: Temporary processing outputs
#   - benchmark/results_*: Benchmark results
# Note: Source code and images are preserved
clean:
	rm -rf $(BIN_DIR) $(TEMP_DIR) $(BENCHMARK_DIR)/results_*
	@echo "✓ Cleaned build files"

# ============================================================================
# TARGET: help
# ============================================================================
# Purpose: Display comprehensive help and usage information
# Usage:   make help
help:
	@echo "============================================================================"
	@echo "Computer Vision Pipeline - Makefile Help"
	@echo "============================================================================"
	@echo "TCC Project: OpenMP-Optimized Image Preprocessing + YOLO Detection"
	@echo ""
	@echo "QUICK START:"
	@echo "  make                              # Build preprocessing binary"
	@echo "  make pipeline IMAGE=image.jpg     # Run full pipeline"
	@echo ""
	@echo "AVAILABLE TARGETS:"
	@echo "  all           Build C++ preprocessing binary (default)"
	@echo "  install-deps  Install Python dependencies (PyTorch, YOLO, etc.)"
	@echo "  test          Test preprocessing with images/sample.jpg"
	@echo "  pipeline      Run full pipeline (preprocessing + YOLO detection)"
	@echo "  benchmark     Run academic performance benchmarks"
	@echo "  clean         Remove build artifacts and temporary files"
	@echo "  help          Show this help message"
	@echo ""
	@echo "DETAILED USAGE:"
	@echo ""
	@echo "  1. Build the preprocessing binary:"
	@echo "     $$ make all"
	@echo ""
	@echo "  2. Run full pipeline on an image:"
	@echo "     $$ make pipeline IMAGE=path/to/image.jpg"
	@echo ""
	@echo "  3. Control thread count (for performance testing):"
	@echo "     $$ OMP_NUM_THREADS=1 make pipeline IMAGE=image.jpg  # Single-threaded"
	@echo "     $$ OMP_NUM_THREADS=4 make pipeline IMAGE=image.jpg  # 4 threads"
	@echo "     $$ OMP_NUM_THREADS=8 make pipeline IMAGE=image.jpg  # 8 threads"
	@echo ""
	@echo "  4. Run academic benchmarks (for TCC/thesis):"
	@echo "     $$ make benchmark                           # Uses images/sample.jpg"
	@echo "     $$ make benchmark IMAGE=../images/times.jpg # Custom image"
	@echo ""
	@echo "PIPELINE FLOW:"
	@echo "  Input Image → C++ Preprocessing (OpenMP) → YOLO Detection → Results"
	@echo ""
	@echo "AVAILABLE FILTERS:"
	@echo "  - blur      : Gaussian blur (noise reduction)"
	@echo "  - sharpen   : Unsharp mask (enhance details)"
	@echo "  - denoise   : Bilateral filter (preserve edges)"
	@echo "  - clahe     : Contrast enhancement"
	@echo "  - edge      : Edge enhancement"
	@echo "  - auto      : Automatic selection (default)"
	@echo ""
	@echo "REQUIREMENTS:"
	@echo "  C++ Dependencies:"
	@echo "    - OpenCV 4.x    : brew install opencv (macOS) / apt install libopencv-dev (Linux)"
	@echo "    - OpenMP        : brew install libomp (macOS) / included with g++ (Linux)"
	@echo "    - C++17 compiler: clang++ (macOS) / g++ (Linux)"
	@echo ""
	@echo "  Python Dependencies (install with 'make install-deps'):"
	@echo "    - Python 3.8+"
	@echo "    - PyTorch, Ultralytics (YOLO), OpenCV, NumPy, Matplotlib, Pandas"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  # Basic usage"
	@echo "  make pipeline IMAGE=images/times.jpg"
	@echo ""
	@echo "  # Performance comparison"
	@echo "  OMP_NUM_THREADS=1 make pipeline IMAGE=test.jpg"
	@echo "  OMP_NUM_THREADS=8 make pipeline IMAGE=test.jpg"
	@echo ""
	@echo "  # Academic benchmark for research"
	@echo "  make benchmark IMAGE=../images/times.jpg"
	@echo ""
	@echo "For more information, see README.md"
	@echo "============================================================================"