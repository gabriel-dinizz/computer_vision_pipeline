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
    echo "  assess <image>       - Assess image quality only"
    echo "  preprocess <image>   - Preprocess image only"
    echo "  detect <image>       - Run YOLO detection only"
    echo "  setup               - Install dependencies and build"
    echo "  test                - Run tests with sample image"
    echo "  benchmark           - Run performance benchmark"
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
    echo "  $0 process images/photo.jpg -f sharpen         # Use sharpening filter"
    echo "  $0 assess images/blurry.jpg                    # Check image quality"
    echo "  $0 preprocess images/photo.jpg -f denoise      # Denoise only"
    echo "  $0 detect images/processed.jpg -c 0.5          # YOLO detection with 50% confidence"
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
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                return 1
                ;;
        esac
    done
    
    # Validate image path
    if [ ! -f "$image" ]; then
        echo -e "${RED}Image file not found: $image${NC}"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Ensure binary exists
    if [[ "$cmd" == "process" || "$cmd" == "preprocess" ]] && [ ! -f bin/preprocess ]; then
        echo -e "${YELLOW}Binary not found, building...${NC}"
        if ! build_project; then
            return 1
        fi
    fi
    
    case "$cmd" in
        "process")
            echo -e "${BLUE}Running full pipeline on $image${NC}"
            python3 python/pipeline.py "$image" -f "$filter" -c "$confidence" -d "$device" $verbose $([ -n "$output" ] && echo "-o $output")
            ;;
        "assess")
            echo -e "${BLUE}Assessing image quality: $image${NC}"
            python3 python/pipeline.py "$image" --assess-only $verbose
            ;;
        "preprocess")
            echo -e "${BLUE}Preprocessing image: $image${NC}"
            local output_path="${output:-temp/preprocessed_$(basename "$image")}"
            mkdir -p "$(dirname "$output_path")"
            ./bin/preprocess "$image" "$output_path" "$filter"
            echo -e "${GREEN}Preprocessed image saved to: $output_path${NC}"
            ;;
        "detect")
            echo -e "${BLUE}Running YOLO detection on: $image${NC}"
            python3 python/pipeline.py "$image" -c "$confidence" -d "$device" $verbose $([ -n "$output" ] && echo "-o $output")
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
    "process"|"assess"|"preprocess"|"detect")
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
