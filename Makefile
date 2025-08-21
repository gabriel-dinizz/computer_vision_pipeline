# Supports macOS (clang + libomp) and Linux (g++)

# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS with Homebrew
    CXX := clang++
    BREW_PREFIX := $(shell brew --prefix)
    OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
    OMP_LIBS   := -L$(BREW_PREFIX)/opt/libomp/lib -lomp -Wl,-rpath,$(BREW_PREFIX)/opt/libomp/lib
else
    # Linux
    CXX := g++
    OMP_CFLAGS := -fopenmp
    OMP_LIBS   := -fopenmp
endif

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

# If opencv4 is not found, try opencv
ifeq ($(OPENCV_CFLAGS),)
    OPENCV_CFLAGS := $(shell pkg-config --cflags opencv)
    OPENCV_LIBS   := $(shell pkg-config --libs opencv)
endif

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra $(OPENCV_CFLAGS) $(OMP_CFLAGS)
LDFLAGS  := $(OPENCV_LIBS) $(OMP_LIBS)

# Source files
SRC_DIR := src
FILTER_SRC_DIR := $(SRC_DIR)/filter_convolution_algorithms
BIN_DIR := bin
TEMP_DIR := temp

# Main preprocessing binary
PREPROCESS_SRC := $(SRC_DIR)/preprocess.cpp
PREPROCESS_BIN := $(BIN_DIR)/preprocess

# Filter algorithm binaries
GAUSSIAN_SRC := $(FILTER_SRC_DIR)/gaussian_blur.cpp
GAUSSIAN_BIN := $(BIN_DIR)/gaussian_blur

# Python files
PYTHON_DIR := python
PIPELINE_SCRIPT := $(PYTHON_DIR)/pipeline.py

.PHONY: all clean test install-deps help run-pipeline build-filters

all: $(PREPROCESS_BIN) build-filters

# Main preprocessing binary
$(PREPROCESS_BIN): $(PREPROCESS_SRC)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built enhanced preprocessing pipeline: $@"

# Build filter algorithms
build-filters:
	@echo "Building filter convolution algorithms..."
	@cd $(FILTER_SRC_DIR) && $(MAKE) all
	@echo "Filter algorithms built successfully"

# Test the preprocessing pipeline
test: $(PREPROCESS_BIN)
	@echo "Testing preprocessing pipeline..."
	@mkdir -p $(TEMP_DIR)
	@if [ -f "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg" ]; then \
		echo "Testing with sample image..."; \
		./$(PREPROCESS_BIN) images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg $(TEMP_DIR)/test_output.jpg auto; \
	else \
		echo "Sample image not found. Please add an image to the images/ directory."; \
	fi

# Run the full Python pipeline
run-pipeline: $(PREPROCESS_BIN)
	@echo "Running full computer vision pipeline..."
	@if [ -f "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg" ]; then \
		cd python && python3 pipeline.py ../images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg -v; \
	else \
		echo "Sample image not found. Usage: make run-pipeline IMAGE=path/to/image.jpg"; \
	fi

# Run pipeline with custom image
run-custom:
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make run-custom IMAGE=path/to/image.jpg"; \
		exit 1; \
	fi
	cd python && python3 pipeline.py $(IMAGE) -v

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	pip3 install opencv-python numpy torch torchvision
	@echo "Downloading YOLOv5 weights..."
	@cd python && python3 -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)"

# Assessment only
assess:
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make assess IMAGE=path/to/image.jpg"; \
		exit 1; \
	fi
	cd python && python3 pipeline.py $(IMAGE) --assess-only

# Performance benchmark
benchmark: $(PREPROCESS_BIN)
	@echo "Running performance benchmark..."
	@mkdir -p $(TEMP_DIR)
	@if [ -f "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg" ]; then \
		echo "Testing different filters..."; \
		for filter in blur sharpen denoise clahe edge; do \
			echo "Testing $$filter filter..."; \
			time ./$(PREPROCESS_BIN) images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg $(TEMP_DIR)/test_$$filter.jpg $$filter; \
		done; \
	else \
		echo "Sample image not found for benchmark."; \
	fi

clean:
	rm -rf $(BIN_DIR) $(TEMP_DIR)
	@cd $(FILTER_SRC_DIR) && $(MAKE) clean
	@echo "Cleaned all build files"

help:
	@echo "Computer Vision Pipeline - Available targets:"
	@echo "  all          - Build preprocessing binary and filter algorithms"
	@echo "  test         - Test preprocessing with sample image"
	@echo "  run-pipeline - Run full Python pipeline with YOLO detection"
	@echo "  run-custom   - Run pipeline with custom image (make run-custom IMAGE=path)"
	@echo "  assess       - Assess image quality only (make assess IMAGE=path)"
	@echo "  benchmark    - Run performance benchmark"
	@echo "  install-deps - Install Python dependencies and download YOLO weights"
	@echo "  build-filters- Build filter convolution algorithms separately"
	@echo "  clean        - Remove all build files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make run-custom IMAGE=images/my_photo.jpg"
	@echo "  make assess IMAGE=images/blurry_photo.jpg"
	@echo ""
	@echo "Requirements:"
	@echo "  - OpenCV library (brew install opencv on macOS)"
	@echo "  - OpenMP library (brew install libomp on macOS)"
	@echo "  - Python 3 with pip"
