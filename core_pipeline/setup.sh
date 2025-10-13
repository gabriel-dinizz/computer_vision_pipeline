#!/bin/bash
set -e

echo "==> Installing dependencies with Homebrew..."
brew update
brew install gcc libomp opencv pkg-config cmake make wget curl python@3.11

echo "==> Creating Python virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio ultralytics opencv-python numpy pandas matplotlib seaborn

echo "==> Setting up directories..."
mkdir -p data/input data/output benchmarks scripts

echo "==> Creating CMakeLists.txt..."
cat <<'CMAKE' > CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(preprocessing_pipeline)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(preprocess src/preprocess.cpp)
target_link_libraries(preprocess OpenMP::OpenMP_CXX \${OpenCV_LIBS})
CMAKE

echo "==> Creating placeholder C++ source..."
mkdir -p src
cat <<'CPP' > src/preprocess.cpp
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>

int main() {
    std::cout << "Preprocessing module placeholder" << std::endl;
    return 0;
}
CPP

echo "==> Creating Makefile..."
cat <<'MAKE' > Makefile
CXX=g++
CXXFLAGS=-std=c++17 -fopenmp \$(shell pkg-config --cflags opencv4)
LDFLAGS=\$(shell pkg-config --libs opencv4) -fopenmp
SRC=src/preprocess.cpp
OUT=bin/preprocess

all: \$(OUT)

\$(OUT): \$(SRC)
	mkdir -p bin
	\$(CXX) \$(CXXFLAGS) -o \$(OUT) \$(SRC) \$(LDFLAGS)

clean:
	rm -rf bin

test: all
	./bin/preprocess
MAKE

echo "==> Creating helper scripts..."
cat <<'BASH' > scripts/run_pipeline.sh
#!/bin/bash
set -e
echo "Running pipeline..."
# placeholder
BASH

chmod +x setup.sh scripts/run_pipeline.sh

echo "==> Bootstrap complete!"
