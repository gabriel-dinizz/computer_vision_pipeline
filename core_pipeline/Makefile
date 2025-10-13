# Streamlined Computer Vision Pipeline Makefile
# Core components: C++ preprocessing + Python YOLO detection

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    CXX := clang++
    BREW_PREFIX := $(shell brew --prefix)
    OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
    OMP_LIBS   := -L$(BREW_PREFIX)/opt/libomp/lib -lomp -Wl,-rpath,$(BREW_PREFIX)/opt/libomp/lib
else
    CXX := g++
    OMP_CFLAGS := -fopenmp
    OMP_LIBS   := -fopenmp
endif

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra $(OPENCV_CFLAGS) $(OMP_CFLAGS)
LDFLAGS  := $(OPENCV_LIBS) $(OMP_LIBS)

SRC_DIR := src
BIN_DIR := bin
TEMP_DIR := temp
PYTHON_DIR := python
BENCHMARK_DIR := benchmark

PREPROCESS_SRC := $(SRC_DIR)/preprocess_optimized.cpp
PREPROCESS_BIN := $(BIN_DIR)/preprocess_optimized

.PHONY: all clean test benchmark install-deps pipeline help

# Build the core preprocessing binary
all: $(PREPROCESS_BIN)

$(PREPROCESS_BIN): $(PREPROCESS_SRC)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built optimized preprocessing pipeline: $@"

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	pip3 install torch torchvision torchaudio ultralytics opencv-python numpy matplotlib seaborn pandas
	@echo "✓ Dependencies installed"

# Test preprocessing with sample image
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

# Run full pipeline (preprocessing + YOLO detection)
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

# Academic benchmark for research validation
benchmark: $(PREPROCESS_BIN)
	@echo "Running academic benchmark..."
	@mkdir -p $(BENCHMARK_DIR)
	cd $(BENCHMARK_DIR) && python3 ../benchmark_academic.py
	@echo "✓ Benchmark completed - results in $(BENCHMARK_DIR)/"

clean:
	rm -rf $(BIN_DIR) $(TEMP_DIR) $(BENCHMARK_DIR)/results_*
	@echo "✓ Cleaned build files"

help:
	@echo "Streamlined Computer Vision Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build preprocessing binary"
	@echo "  install-deps  - Install Python dependencies"
	@echo "  test          - Test preprocessing with sample image"
	@echo "  pipeline      - Run full pipeline (make pipeline IMAGE=path/to/image.jpg)"
	@echo "  benchmark     - Run academic benchmark for research validation"
	@echo "  clean         - Remove build files"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Pipeline Flow:"
	@echo "  Image → C++ Preprocessing (OpenMP) → YOLO Detection (PyTorch) → Results"
	@echo ""
	@echo "Requirements:"
	@echo "  - OpenCV 4.x (brew install opencv on macOS)"
	@echo "  - OpenMP (brew install libomp on macOS)"
	@echo "  - Python 3.8+ with pip"