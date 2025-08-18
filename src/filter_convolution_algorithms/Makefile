# Makefile for Filter Convolution Algorithms
# Requires OpenCV to be installed

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

# If opencv4 is not found, try opencv
ifeq ($(OPENCV_FLAGS),)
    OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv)
endif

SRCDIR = .
OBJDIR = obj
BINDIR = bin

# Create directories if they don't exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Source files
SOURCES = gaussian_blur.cpp
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)
EXECUTABLES = $(SOURCES:%.cpp=$(BINDIR)/%)

.PHONY: all clean

all: $(EXECUTABLES)

# Rule to build executables
$(BINDIR)/%: $(OBJDIR)/%.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_FLAGS)

# Rule to build object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(shell pkg-config --cflags opencv4 || pkg-config --cflags opencv)

# Individual targets
gaussian_blur: $(BINDIR)/gaussian_blur

clean:
	rm -rf $(OBJDIR) $(BINDIR)
	rm -f *.jpg *.png  # Clean up generated images

test: gaussian_blur
	./$(BINDIR)/gaussian_blur

help:
	@echo "Available targets:"
	@echo "  all          - Build all convolution algorithms"
	@echo "  gaussian_blur - Build Gaussian blur filter"
	@echo "  test         - Run Gaussian blur test"
	@echo "  clean        - Remove build files and generated images"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - OpenCV library installed"
	@echo "  - pkg-config for OpenCV"
