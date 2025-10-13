#!/bin/bash

# Computer Vision Pipeline Runner
# Convenient wrapper for the preprocessing + YOLO detection pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Computer Vision Pipeline - Preprocessing + YOLO Detection"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  process <image>      - Run full pipeline on image"
    echo "  full-pipeline <image> - Run enhanced full pipeline"
    echo "  compare <image>      - Compare with/without preprocessing"
    echo "  yolo-only <image>    - Run YOLO detection only"
    echo "  benchmark <image>    - Run comprehensive benchmark"
    echo "  assess <image>       - Assess image quality only"
    echo "  preprocess <image>   - Preprocess image only"
    echo "  detect <image>       - Run YOLO detection only"
    echo "  setup               - Install dependencies and build"
    echo "  test                - Run tests with sample image"
    echo ""
    echo "Options:"
    echo "  -f, --filter FILTER  - Preprocessing filter (auto, blur, sharpen, denoise, clahe, edge)"
    echo "  -c, --confidence NUM - YOLO confidence threshold (0.0-1.0, default: 0.25)"
    echo "  -d, --device DEVICE  - Processing device (cpu, cuda, mps, default: cpu)"
    echo "  -o, --output DIR     - Output directory"
    echo "  -v, --verbose        - Verbose output"
    echo "  -h, --help           - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 process images/photo.jpg                    # Full pipeline with auto filter"
    echo "  $0 compare images/photo.jpg -f sharpen         # Compare with/without preprocessing"
    echo "  $0 yolo-only images/photo.jpg -c 0.5           # YOLO detection only"
    echo "  $0 benchmark images/photo.jpg                  # Full performance benchmark"
    echo "  $0 assess images/blurry.jpg                    # Check image quality"
    echo "  $0 preprocess images/photo.jpg -f denoise      # Denoise only"
    echo ""
}

check_dependencies() {
    local missing_deps=()

    # Check for required tools
    if ! command -v make &> /dev/null; then
        missing_deps+=("make")
    fi

    if ! command -v pkg-config &> /dev/null; then
        missing_deps+=("pkg-config")
    fi

    if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
        missing_deps+=("opencv")
    fi

    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}Missing dependencies: ${missing_deps[*]}${NC}"
        echo ""
        echo "On macOS, install with:"
        echo "  brew install opencv pkg-config"
        echo "  # Python 3 is usually pre-installed"
        echo ""
        echo "On Ubuntu/Debian, install with:"
        echo "  sudo apt-get install build-essential pkg-config libopencv-dev python3 python3-pip"
        echo ""
        return 1
    fi

    return 0
}

build_project() {
    echo -e "${BLUE}Building project...${NC}"
    cd "$PROJECT_ROOT"

    if ! make all; then
        echo -e "${RED}Build failed!${NC}"
        return 1
    fi

    echo -e "${GREEN}Build successful!${NC}"
    return 0
}

setup_pipeline() {
    echo -e "${BLUE}Setting up Computer Vision Pipeline...${NC}"

    # Check dependencies
    if ! check_dependencies; then
        return 1
    fi

    # Build project
    if ! build_project; then
        return 1
    fi

    # Install Python dependencies
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    if ! make install-deps; then
        echo -e "${YELLOW}Warning: Python dependencies installation failed${NC}"
        echo "You may need to install manually:"
        echo "  pip3 install opencv-python numpy torch torchvision"
    fi

    echo -e "${GREEN}Setup complete!${NC}"
    echo ""
    echo "You can now run:"
    echo "  $0 test                    # Run tests"
    echo "  $0 process images/photo.jpg # Process an image"

    return 0
}

run_test() {
    echo -e "${BLUE}Running pipeline tests...${NC}"
    cd "$PROJECT_ROOT"

    if [ ! -f bin/preprocess ]; then
        echo -e "${YELLOW}Binary not found, building...${NC}"
        if ! build_project; then
            return 1
        fi
    fi

    make test
}

run_benchmark() {
    echo -e "${BLUE}Running performance benchmark...${NC}"
    cd "$PROJECT_ROOT"

    if [ ! -f bin/preprocess ]; then
        echo -e "${YELLOW}Binary not found, building...${NC}"
        if ! build_project; then
            return 1
        fi
    fi

    make benchmark
}

run_command() {
    local cmd="$1"
    local image="$2"
    shift 2

    # Parse options
    local filter="auto"
    local confidence="0.25"
    local device="cpu"
    local output=""
    local verbose=""
    local threads=""
    local controlled="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--filter)
                filter="$2"
                shift 2
                ;;
            -c|--confidence)
                confidence="$2"
                shift 2
                ;;
            -d|--device)
                device="$2"
                shift 2
                ;;
            -o|--output)
                output="$2"
                shift 2
                ;;
            -v|--verbose)
                verbose="-v"
                shift
                ;;
            -t|--threads)
                threads="$2"
                shift 2
                ;;
            --controlled)
                controlled="true"
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                return 1
                ;;
        esac
    done

    # Validate image path (skip for research commands)
    if [[ "$cmd" != "research-validate" && "$cmd" != "research-benchmark" ]]; then
        if [ ! -f "$image" ]; then
            echo -e "${RED}Image file not found: $image${NC}"
            return 1
        fi
    fi

    cd "$PROJECT_ROOT"

    # Ensure binary exists
    if [[ "$cmd" == "process" || "$cmd" == "preprocess" ]] && [ ! -f bin/preprocess ]; then
        echo -e "${YELLOW}Binary not found, building...${NC}"
        if ! build_project; then
            return 1
        fi
    fi

    # Set controlled environment if requested
    if [[ "$controlled" == "true" ]]; then
        echo -e "${BLUE}Setting controlled research environment...${NC}"

        # Set CPU affinity if available (Linux)
        if command -v taskset &> /dev/null && [ -n "$threads" ]; then
            local cpu_list=""
            for ((i=0; i<threads; i++)); do
                cpu_list="$cpu_list$i"
                if [ $i -lt $((threads-1)) ]; then
                    cpu_list="$cpu_list,"
                fi
            done
            export TASKSET_CMD="taskset -c $cpu_list"
            echo "CPU affinity set to cores: $cpu_list"
        fi

        # Set thread count
        if [ -n "$threads" ]; then
            export OMP_NUM_THREADS="$threads"
            echo "OpenMP threads set to: $threads"
        fi

        # Disable CPU frequency scaling if possible
        echo "Note: For research accuracy, consider disabling CPU frequency scaling"
    fi

    case "$cmd" in
        "process")
            echo -e "${BLUE}Running full pipeline on $image${NC}"

            # Run with controlled environment if set
            local run_cmd="python3 python/pipeline.py \"$image\" -f \"$filter\" -c \"$confidence\" -d \"$device\" $verbose $([ -n \"$output\" ] && echo \"-o $output\")"
            if [ -n "$TASKSET_CMD" ]; then
                run_cmd="$TASKSET_CMD $run_cmd"
            fi

            eval $run_cmd
            ;;
        "full-pipeline")
            echo -e "${BLUE}Running enhanced full pipeline on $image${NC}"
            
            local run_cmd="python3 python/full_pipeline.py \"$image\" -f \"$filter\" -c \"$confidence\" -d \"$device\" $verbose $([ -n \"$output\" ] && echo \"-o $output\")"
            if [ -n "$TASKSET_CMD" ]; then
                run_cmd="$TASKSET_CMD $run_cmd"
            fi

            eval $run_cmd
            ;;
        "compare")
            echo -e "${BLUE}Comparing pipeline performance on $image${NC}"
            
            local run_cmd="python3 python/full_pipeline.py \"$image\" -m compare -f \"$filter\" -c \"$confidence\" -d \"$device\" $verbose $([ -n \"$output\" ] && echo \"-o $output\")"
            if [ -n "$TASKSET_CMD" ]; then
                run_cmd="$TASKSET_CMD $run_cmd"
            fi

            eval $run_cmd
            ;;
        "yolo-only")
            echo -e "${BLUE}Running YOLO detection only on $image${NC}"
            
            local run_cmd="python3 python/yolo_detector.py \"$image\" -c \"$confidence\" -d \"$device\" $([ -n \"$output\" ] && echo \"-o $output\") --save"
            if [ -n "$TASKSET_CMD" ]; then
                run_cmd="$TASKSET_CMD $run_cmd"
            fi

            eval $run_cmd
            ;;
        "benchmark")
            echo -e "${BLUE}Running comprehensive benchmark on $image${NC}"
            
            local bench_args=""
            if [ -n "$threads" ]; then
                bench_args="$bench_args -r $threads"
            else
                bench_args="$bench_args -r 5"
            fi
            
            local run_cmd="python3 benchmark_pipeline.py \"$image\" $bench_args -f \"$filter\" -c \"$confidence\" -d \"$device\" $([ -n \"$output\" ] && echo \"-o $output\")"
            if [ -n "$TASKSET_CMD" ]; then
                run_cmd="$TASKSET_CMD $run_cmd"
            fi

            eval $run_cmd
            ;;
        "assess")
            echo -e "${BLUE}Assessing image quality: $image${NC}"
            python3 python/pipeline.py "$image" --assess-only $verbose
            ;;
        "preprocess")
            echo -e "${BLUE}Preprocessing image: $image${NC}"
            local output_path="${output:-temp/preprocessed_$(basename "$image")}"
            mkdir -p "$(dirname "$output_path")"

            # Run with controlled environment if set
            local run_cmd="./bin/preprocess \"$image\" \"$output_path\" \"$filter\""
            if [ -n "$TASKSET_CMD" ]; then
                run_cmd="$TASKSET_CMD $run_cmd"
            fi

            eval $run_cmd
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Preprocessed image saved to: $output_path${NC}"
            else
                echo -e "${RED}Preprocessing failed!${NC}"
                return 1
            fi
            ;;
        "detect")
            echo -e "${BLUE}Running YOLO detection on: $image${NC}"
            python3 python/pipeline.py "$image" -c "$confidence" -d "$device" $verbose $([ -n "$output" ] && echo "-o $output")
            ;;
        "research-benchmark")
            echo -e "${BLUE}Starting academic research benchmark...${NC}"
            if [ ! -f python/research_benchmark_academic.py ]; then
                echo -e "${RED}Research benchmark script not found!${NC}"
                return 1
            fi

            # Run with controlled environment if set
            cd python
            local run_cmd="python research_benchmark_academic.py"
            if [ -n "$TASKSET_CMD" ]; then
                run_cmd="$TASKSET_CMD $run_cmd"
            fi

            echo "Running: $run_cmd"
            eval $run_cmd
            cd ..
            ;;
        "research-validate")
            echo -e "${BLUE}Validating academic research setup...${NC}"

            # Check all required files
            local required_files=(
                "src/preprocess.cpp"
                "src/sequential_baseline.cpp"
                "src/opencv_baseline.cpp"
                "research_benchmark_academic.py"
                "bin/preprocess"
                "bin/sequential_baseline"
                "bin/opencv_baseline"
            )

            local missing_files=()
            for file in "${required_files[@]}"; do
                if [ ! -f "$file" ]; then
                    missing_files+=("$file")
                fi
            done

            if [ ${#missing_files[@]} -gt 0 ]; then
                echo -e "${RED}Missing required files:${NC}"
                for file in "${missing_files[@]}"; do
                    echo "  - $file"
                done
                return 1
            fi

            echo -e "${GREEN}All required files present!${NC}"
            echo -e "${BLUE}Running quick validation...${NC}"

            # Test preprocessing with a quick run
            if [ -f "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg" ]; then
                echo "Testing parallel preprocessing..."
                ./bin/preprocess "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg" "temp/test_parallel.jpg" "blur"

                echo "Testing sequential baseline..."
                ./bin/sequential_baseline "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg" "temp/test_sequential.jpg" "blur"

                echo "Testing OpenCV baseline..."
                ./bin/opencv_baseline "images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg" "temp/test_opencv.jpg" "blur"

                echo -e "${GREEN}All baselines working correctly!${NC}"
            else
                echo -e "${YELLOW}No test image found, skipping processing validation${NC}"
            fi
            ;;
        "research-benchmark")
            echo -e "${BLUE}Starting academic research benchmark...${NC}"
            if [ ! -f research_benchmark_academic.py ]; then
                echo -e "${RED}Research benchmark script not found!${NC}"
                return 1
            fi

            # Activate virtual environment if it exists
            if [ -d "research_env" ]; then
                echo "Activating research environment..."
                source research_env/bin/activate
            fi

            # Run with controlled environment if set
            local run_cmd="python research_benchmark_academic.py"
            if [ -n "$TASKSET_CMD" ]; then
                run_cmd="$TASKSET_CMD $run_cmd"
            fi

            echo "Running: $run_cmd"
            eval $run_cmd
            ;;
        *)
            echo -e "${RED}Unknown command: $cmd${NC}"
            return 1
            ;;
    esac
}

# Main script logic
if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

case "$1" in
    "setup")
        setup_pipeline
        ;;
    "test")
        run_test
        ;;
    "benchmark")
        run_benchmark
        ;;
    "research-benchmark")
        # Research benchmark with controlled environment
        shift
        run_command research-benchmark "" "$@"
        ;;
    "research-validate")
        # Validate academic research setup
        shift
        run_command research-validate "" "$@"
        ;;
    "process"|"full-pipeline"|"compare"|"yolo-only"|"benchmark"|"assess"|"preprocess"|"detect")
        if [ $# -lt 2 ]; then
            echo -e "${RED}Error: Image path required for $1 command${NC}"
            echo ""
            print_usage
            exit 1
        fi
        run_command "$@"
        ;;
    "-h"|"--help"|"help")
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
