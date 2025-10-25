# ============================================================================
# Computer Vision Pipeline - Unified Makefile
# ============================================================================
# Purpose: Build OpenMP-optimized image preprocessing + YOLO detection pipeline
# Author:  TCC Project - Computer Vision Pipeline with Parallel Processing
# ============================================================================

# ============================================================================
# PLATFORM DETECTION AND COMPILER CONFIGURATION
# ============================================================================
UNAME_S := $(shell uname -s)

# macOS Configuration
ifeq ($(UNAME_S),Darwin)
    CXX := clang++
    BREW_PREFIX := $(shell brew --prefix)
    # macOS requires -Xpreprocessor flag for OpenMP
    OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
    OMP_LIBS   := -L$(BREW_PREFIX)/opt/libomp/lib -lomp -Wl,-rpath,$(BREW_PREFIX)/opt/libomp/lib
else
    # Linux Configuration
    CXX := g++
    OMP_CFLAGS := -fopenmp
    OMP_LIBS   := -fopenmp
endif

# ============================================================================
# OPENCV CONFIGURATION
# ============================================================================
# Auto-detect OpenCV installation via pkg-config
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
CORE_DIR := core_pipeline
SRC_DIR := $(CORE_DIR)/src
BIN_DIR := $(CORE_DIR)/bin
TEMP_DIR := $(CORE_DIR)/temp
PYTHON_DIR := $(CORE_DIR)/python
BENCHMARK_DIR := $(CORE_DIR)/benchmark
IMAGES_DIR := $(CORE_DIR)/images

# ============================================================================
# BUILD TARGETS
# ============================================================================
PREPROCESS_OPTIMIZED_SRC := $(SRC_DIR)/preprocess_optimized.cpp
PREPROCESS_OPTIMIZED_BIN := $(BIN_DIR)/preprocess_optimized

# ============================================================================
# PHONY TARGETS
# ============================================================================
.PHONY: all clean test benchmark install-deps pipeline help run-custom assess detect-only

# ============================================================================
# TARGET: all (default)
# ============================================================================
# Purpose: Build the optimized preprocessing binary
# Usage:   make
#          make all
all: $(PREPROCESS_OPTIMIZED_BIN)
	@echo "✓ Build complete! Use 'make help' for usage information."

# Main build rule for optimized preprocessing binary
$(PREPROCESS_OPTIMIZED_BIN): $(PREPROCESS_OPTIMIZED_SRC)
	@echo "Building optimized preprocessing pipeline..."
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built: $@"

# ============================================================================
# TARGET: install-deps
# ============================================================================
# Purpose: Install all required Python dependencies
# Installs: PyTorch, Ultralytics (YOLO), OpenCV, NumPy, Matplotlib, Pandas
install-deps:
	@echo "Installing Python dependencies..."
	pip3 install torch torchvision torchaudio ultralytics opencv-python numpy matplotlib seaborn pandas
	@echo "✓ Dependencies installed successfully"

# ============================================================================
# TARGET: test
# ============================================================================
# Purpose: Quick test of preprocessing pipeline with sample image
# Input:   Expects core_pipeline/images/sample.jpg to exist
# Output:  core_pipeline/temp/test_output.jpg
test: $(PREPROCESS_OPTIMIZED_BIN)
	@echo "Testing preprocessing pipeline..."
	@mkdir -p $(TEMP_DIR)
	@if [ -f "$(IMAGES_DIR)/sample.jpg" ]; then \
		echo "Testing with sample image..."; \
		./$(PREPROCESS_OPTIMIZED_BIN) $(IMAGES_DIR)/sample.jpg $(TEMP_DIR)/test_output.jpg auto; \
		echo "✓ Test completed - output: $(TEMP_DIR)/test_output.jpg"; \
	else \
		echo "Sample image not found. Looking for any .jpg in $(IMAGES_DIR)/..."; \
		IMAGE=$$(find $(IMAGES_DIR) -name "*.jpg" -o -name "*.jpeg" | head -1); \
		if [ -n "$$IMAGE" ]; then \
			echo "Testing with: $$IMAGE"; \
			./$(PREPROCESS_OPTIMIZED_BIN) $$IMAGE $(TEMP_DIR)/test_output.jpg auto; \
			echo "✓ Test completed - output: $(TEMP_DIR)/test_output.jpg"; \
		else \
			echo "Error: No images found in $(IMAGES_DIR)/"; \
			echo "Please add a .jpg image to test with."; \
		fi; \
	fi

# ============================================================================
# TARGET: pipeline
# ============================================================================
# Purpose: Run complete end-to-end computer vision pipeline
# Usage:   make pipeline IMAGE=path/to/image.jpg
#          OMP_NUM_THREADS=4 make pipeline IMAGE=core_pipeline/images/sample.jpg
# Steps:   1. C++ preprocessing (OpenMP parallelized)
#          2. YOLO object detection
pipeline: $(PREPROCESS_OPTIMIZED_BIN)
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make pipeline IMAGE=path/to/image.jpg"; \
		echo "Example: make pipeline IMAGE=core_pipeline/images/sample.jpg"; \
		exit 1; \
	fi
	@echo "============================================================"
	@echo "Running Computer Vision Pipeline"
	@echo "============================================================"
	@echo "Step 1: Preprocessing image with OpenMP optimization..."
	@mkdir -p $(TEMP_DIR)
	./$(PREPROCESS_OPTIMIZED_BIN) $(IMAGE) $(TEMP_DIR)/preprocessed.jpg auto
	@echo ""
	@echo "Step 2: Running YOLO object detection..."
	cd $(PYTHON_DIR) && python3 yolo_detector.py ../temp/preprocessed.jpg --save -o ../temp/detection_results
	@echo ""
	@echo "✓ Pipeline completed successfully!"
	@echo "Results saved in: $(TEMP_DIR)/detection_results/"
	@echo "============================================================"

# ============================================================================
# TARGET: run-custom
# ============================================================================
# Purpose: Run pipeline with custom image (alternative syntax)
run-custom: pipeline

# ============================================================================
# TARGET: assess
# ============================================================================
# Purpose: Assess image quality only (no detection)
assess: $(PREPROCESS_OPTIMIZED_BIN)
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make assess IMAGE=path/to/image.jpg"; \
		exit 1; \
	fi
	@echo "Assessing image quality..."
	@mkdir -p $(TEMP_DIR)
	./$(PREPROCESS_OPTIMIZED_BIN) $(IMAGE) $(TEMP_DIR)/assessed.jpg auto
	@echo "✓ Assessment complete - output: $(TEMP_DIR)/assessed.jpg"

# ============================================================================
# TARGET: detect-only
# ============================================================================
# Purpose: Run YOLO detection only (skip preprocessing)
# Usage:   make detect-only IMAGE=path/to/image.jpg
#          make detect-only IMAGE=core_pipeline/images/car.jpg
#          make detect-only IMAGE=/absolute/path/to/image.jpg
detect-only:
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make detect-only IMAGE=path/to/image.jpg"; \
		echo "Example: make detect-only IMAGE=core_pipeline/images/car.jpg"; \
		exit 1; \
	fi
	@echo "============================================================"
	@echo "Running YOLO Object Detection (No Preprocessing)"
	@echo "============================================================"
	@echo "Input: $(IMAGE)"
	@mkdir -p $(TEMP_DIR)
	@# Convert to absolute path to handle both relative and absolute inputs
	@IMAGE_ABS=$$(cd $$(dirname "$(IMAGE)") 2>/dev/null && pwd)/$$(basename "$(IMAGE)") || IMAGE_ABS="$(IMAGE)"; \
	cd $(PYTHON_DIR) && python3 yolo_detector.py "$$IMAGE_ABS" --save -o ../temp/detection_results
	@echo ""
	@echo "✓ Detection completed successfully!"
	@echo "Results saved in: $(TEMP_DIR)/detection_results/"
	@echo "============================================================"

# ============================================================================
# TARGET: benchmark
# ============================================================================
# Purpose: Run comprehensive academic performance benchmark
# Usage:   make benchmark
#          make benchmark IMAGE=core_pipeline/images/sample.jpg
# Output:  Results in core_pipeline/benchmark/results_academic/
benchmark: $(PREPROCESS_OPTIMIZED_BIN)
	@echo "Running academic benchmark..."
	@mkdir -p $(BENCHMARK_DIR)
	@if [ -d "$(BENCHMARK_DIR)" ] && [ -f "$(BENCHMARK_DIR)/benchmark_academic.py" ]; then \
		cd $(BENCHMARK_DIR) && python3 benchmark_academic.py $(if $(IMAGE),--image $(IMAGE),); \
		echo "✓ Benchmark completed - results in $(BENCHMARK_DIR)/results_academic/"; \
	else \
		echo "Running basic performance benchmark..."; \
		mkdir -p $(TEMP_DIR); \
		IMAGE_FILE=$(if $(IMAGE),$(IMAGE),$$(find $(IMAGES_DIR) -name "*.jpg" -o -name "*.jpeg" | head -1)); \
		if [ -n "$$IMAGE_FILE" ]; then \
			echo "Testing filters with: $$IMAGE_FILE"; \
			for filter in blur sharpen denoise clahe edge auto; do \
				echo ""; \
				echo "Testing $$filter filter..."; \
				time ./$(PREPROCESS_OPTIMIZED_BIN) $$IMAGE_FILE $(TEMP_DIR)/bench_$$filter.jpg $$filter; \
			done; \
			echo "✓ Basic benchmark completed - outputs in $(TEMP_DIR)/"; \
		else \
			echo "Error: No image specified and none found in $(IMAGES_DIR)/"; \
		fi; \
	fi

# ============================================================================
# TARGET: clean
# ============================================================================
# Purpose: Remove all build artifacts and temporary files
clean:
	@echo "Cleaning build files..."
	rm -rf $(BIN_DIR) $(TEMP_DIR) $(BENCHMARK_DIR)/results_*
	@echo "✓ Cleaned successfully"

# ============================================================================
# TARGET: help
# ============================================================================
# Purpose: Display comprehensive usage information
help:
	@echo "============================================================================"
	@echo "Computer Vision Pipeline - Makefile Help"
	@echo "============================================================================"
	@echo "TCC Project: OpenMP-Optimized Image Preprocessing + YOLO Detection"
	@echo ""
	@echo "QUICK START:"
	@echo "  make                                    # Build preprocessing binary"
	@echo "  make pipeline IMAGE=path/to/image.jpg   # Run full pipeline"
	@echo ""
	@echo "AVAILABLE TARGETS:"
	@echo "  all           Build C++ preprocessing binary (default)"
	@echo "  install-deps  Install Python dependencies (PyTorch, YOLO, etc.)"
	@echo "  test          Test preprocessing with sample image"
	@echo "  pipeline      Run full pipeline (preprocessing + YOLO detection)"
	@echo "  detect-only   Run YOLO detection only (skip preprocessing)"
	@echo "  assess        Assess image quality only (no detection)"
	@echo "  benchmark     Run performance benchmarks"
	@echo "  clean         Remove build artifacts and temporary files"
	@echo "  help          Show this help message"
	@echo ""
	@echo "DETAILED USAGE:"
	@echo ""
	@echo "  1. First-time setup:"
	@echo "     $$ make install-deps    # Install Python dependencies"
	@echo "     $$ make all             # Build C++ binary"
	@echo ""
	@echo "  2. Run pipeline on an image:"
	@echo "     $$ make pipeline IMAGE=core_pipeline/images/sample.jpg"
	@echo ""
	@echo "  3. Run YOLO detection only (skip preprocessing):"
	@echo "     $$ make detect-only IMAGE=core_pipeline/images/car.jpg"
	@echo ""
	@echo "  4. Control thread count (for performance testing):"
	@echo "     $$ OMP_NUM_THREADS=1 make pipeline IMAGE=image.jpg  # Single-threaded"
	@echo "     $$ OMP_NUM_THREADS=4 make pipeline IMAGE=image.jpg  # 4 threads"
	@echo "     $$ OMP_NUM_THREADS=8 make pipeline IMAGE=image.jpg  # 8 threads"
	@echo ""
	@echo "  5. Run benchmarks (for academic analysis):"
	@echo "     $$ make benchmark"
	@echo "     $$ make benchmark IMAGE=core_pipeline/images/sample.jpg"
	@echo ""
	@echo "  6. Quick quality assessment:"
	@echo "     $$ make assess IMAGE=core_pipeline/images/blurry_photo.jpg"
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
	@echo "  - auto      : Automatic selection (default, recommended)"
	@echo ""
	@echo "REQUIREMENTS:"
	@echo "  C++ Dependencies:"
	@echo "    - OpenCV 4.x    : brew install opencv (macOS)"
	@echo "    - OpenMP        : brew install libomp (macOS)"
	@echo "    - C++17 compiler: clang++ (macOS) / g++ (Linux)"
	@echo ""
	@echo "  Python Dependencies (install with 'make install-deps'):"
	@echo "    - Python 3.8+"
	@echo "    - PyTorch, Ultralytics (YOLO), OpenCV, NumPy, Matplotlib, Pandas"
	@echo ""
	@echo "PROJECT STRUCTURE:"
	@echo "  core_pipeline/"
	@echo "    ├── src/              C++ source code"
	@echo "    ├── bin/              Compiled binaries"
	@echo "    ├── python/           YOLO detection scripts"
	@echo "    ├── images/           Input images"
	@echo "    ├── temp/             Temporary outputs"
	@echo "    └── benchmark/        Performance benchmarks"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  # Full pipeline (preprocessing + detection)"
	@echo "  make pipeline IMAGE=core_pipeline/images/car.jpg"
	@echo ""
	@echo "  # Detection only (no preprocessing)"
	@echo "  make detect-only IMAGE=core_pipeline/images/car.jpg"
	@echo ""
	@echo "  # Performance comparison"
	@echo "  OMP_NUM_THREADS=1 make pipeline IMAGE=test.jpg"
	@echo "  OMP_NUM_THREADS=8 make pipeline IMAGE=test.jpg"
	@echo ""
	@echo "  # Academic benchmark"
	@echo "  make benchmark IMAGE=core_pipeline/images/sample.jpg"
	@echo ""
	@echo "For more information, see core_pipeline/README.md"
	@echo "============================================================================"
