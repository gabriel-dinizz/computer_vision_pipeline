CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

TARGETS = gaussian_blur_test sharpen_test benchmark

all: $(TARGETS)

gaussian_blur_test: gaussian_blur_comparison.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(OPENCV_FLAGS)

sharpen_test: sharpen_comparison.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(OPENCV_FLAGS)

benchmark: benchmark.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(OPENCV_FLAGS)

clean:
	rm -f $(TARGETS) *.jpg

test: all
	@echo "Testing with sample image..."
	@if [ -f "../../images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg" ]; then \
		echo "Running Gaussian blur comparison:"; \
		./gaussian_blur_test ../../images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg; \
		echo ""; \
		echo "Running sharpen comparison:"; \
		./sharpen_test ../../images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg; \
		echo ""; \
		echo "Running full benchmark:"; \
		./benchmark ../../images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg; \
	else \
		echo "Sample image not found. Please provide an image path:"; \
		echo "./gaussian_blur_test <image_path>"; \
	fi

.PHONY: all clean test
