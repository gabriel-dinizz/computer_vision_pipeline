# Makefile for Filter Convolution Algorithms
# Requires OpenCV to be installed

# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS with Homebrew
    CXX = clang++
    BREW_PREFIX := $(shell brew --prefix)
    OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
    OMP_LIBS   := -L$(BREW_PREFIX)/opt/libomp/lib -lomp -Wl,-rpath,$(BREW_PREFIX)/opt/libomp/lib
else
    # Linux
    CXX = g++
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

CXXFLAGS = -std=c++11 -Wall -Wextra -O2 $(OPENCV_CFLAGS) $(OMP_CFLAGS)
LDFLAGS = $(OPENCV_LIBS) $(OMP_LIBS)

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
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Rule to build object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Individual targets
gaussian_blur: $(BINDIR)/gaussian_blur

clean:
	rm -rf $(OBJDIR) $(BINDIR)
	rm -f *.jpg *.png  # Clean up generated images

test: $(BINDIR)/gaussian_blur
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
