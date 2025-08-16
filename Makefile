# Makefile (macOS, clang + libomp)
CXX := clang++
BREW_PREFIX := $(shell brew --prefix)

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

# Tell clang where to find libomp headers and libs
OMP_CFLAGS := -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
OMP_LIBS   := -L$(BREW_PREFIX)/opt/libomp/lib -lomp -Wl,-rpath,$(BREW_PREFIX)/opt/libomp/lib

CXXFLAGS := -std=c++17 -O3 $(OPENCV_CFLAGS) $(OMP_CFLAGS)
LDFLAGS  := $(OPENCV_LIBS) $(OMP_LIBS)

SRC := src/preprocess.cpp
BIN := bin/preprocess

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf bin
