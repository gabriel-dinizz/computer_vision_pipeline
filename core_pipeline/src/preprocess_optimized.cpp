#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

enum class FilterType {
  GAUSSIAN_BLUR,
  UNSHARP_MASK,
  LAPLACIAN_SHARPEN,
  BILATERAL_DENOISE,
  CLAHE_ENHANCE,
  EDGE_ENHANCE
};

/**
 * Optimized ImagePreprocessor with proper OpenMP implementations
 * Fixes fundamental algorithmic issues with custom parallel convolution
 */
class OptimizedImagePreprocessor {
 private:
  bool verbose;

  // Performance counters
  struct PerformanceCounters {
    double kernelGenTime = 0.0;
    double convolutionTime = 0.0;
    double memoryAllocTime = 0.0;
    double totalTime = 0.0;
    int threadsUsed = 1;
    size_t cacheHits = 0;
    size_t cacheMisses = 0;
  };

  mutable PerformanceCounters perfCounters;

  // Optimal tile sizes for cache efficiency
  static constexpr int CACHE_LINE_SIZE = 64;
  static constexpr int OPTIMAL_TILE_SIZE = 64;  // 64x64 fits L1 cache
  static constexpr int PIXELS_PER_CACHE_LINE =
      CACHE_LINE_SIZE / sizeof(cv::Vec3b);

 public:
  OptimizedImagePreprocessor(bool verbose = true) : verbose(verbose) {}

  /**
   * Generate 1D Gaussian kernel for separable convolution
   */
  std::vector<float> generateGaussianKernel(int size, double sigma) const {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> kernel(size);
    int center = size / 2;
    float sum = 0.0f;

    // Generate kernel values
    for (int i = 0; i < size; i++) {
      float x = i - center;
      kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
      sum += kernel[i];
    }

    // Normalize
    for (float& val : kernel) {
      val /= sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    perfCounters.kernelGenTime +=
        std::chrono::duration<double, std::milli>(end - start).count();

    return kernel;
  }

  /**
   * Optimized separable Gaussian blur with proper OpenMP parallelization
   */
  cv::Mat applyOptimizedGaussianBlur(const cv::Mat& img, double sigma = 1.0) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    // Calculate optimal kernel size
    int kernelSize = static_cast<int>(6 * sigma + 1);
    if (kernelSize % 2 == 0) kernelSize++;  // Ensure odd size

    // Generate 1D Gaussian kernel
    std::vector<float> kernel = generateGaussianKernel(kernelSize, sigma);
    int radius = kernelSize / 2;

    // Create intermediate and result matrices
    cv::Mat intermediate(img.size(), CV_32FC3);
    cv::Mat result(img.size(), CV_32FC3);

    // Convert input to float for precision
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32FC3, 1.0 / 255.0);

    auto convStart = std::chrono::high_resolution_clock::now();

// PHASE 1: Horizontal convolution (parallelized by rows)
// Use static scheduling with larger chunks for better performance
// Dynamic scheduling with chunk=4 causes too much overhead
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < img.rows; y++) {
      const cv::Vec3f* srcRow = imgFloat.ptr<cv::Vec3f>(y);
      cv::Vec3f* dstRow = intermediate.ptr<cv::Vec3f>(y);

      // Process row with proper boundary handling
      for (int x = 0; x < img.cols; x++) {
        cv::Vec3f sum(0, 0, 0);

        for (int k = 0; k < kernelSize; k++) {
          int srcX = x - radius + k;
          // Handle boundaries by clamping
          srcX = std::max(0, std::min(srcX, img.cols - 1));

          cv::Vec3f pixel = srcRow[srcX];
          sum += pixel * kernel[k];
        }

        dstRow[x] = sum;
      }
    }

    // PHASE 2: Vertical convolution (parallelized by rows for better cache locality)
    // Process by rows instead of columns to maintain cache-friendly access patterns
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < img.rows; y++) {
      cv::Vec3f* dstRow = result.ptr<cv::Vec3f>(y);

      for (int x = 0; x < img.cols; x++) {
        cv::Vec3f sum(0, 0, 0);

        for (int k = 0; k < kernelSize; k++) {
          int srcY = y - radius + k;
          // Handle boundaries by clamping
          srcY = std::max(0, std::min(srcY, img.rows - 1));

          cv::Vec3f pixel = intermediate.ptr<cv::Vec3f>(srcY)[x];
          sum += pixel * kernel[k];
        }

        dstRow[x] = sum;
      }
    }

    auto convEnd = std::chrono::high_resolution_clock::now();
    perfCounters.convolutionTime +=
        std::chrono::duration<double, std::milli>(convEnd - convStart).count();

    // Convert back to 8-bit
    cv::Mat finalResult;
    result.convertTo(finalResult, CV_8UC3, 255.0);

    auto totalEnd = std::chrono::high_resolution_clock::now();
    perfCounters.totalTime +=
        std::chrono::duration<double, std::milli>(totalEnd - totalStart)
            .count();
    perfCounters.threadsUsed = omp_get_max_threads();

    if (verbose)
      std::cout << "Applied Optimized Separable Gaussian Blur (OpenMP)\n";
    return finalResult;
  }

  /**
   * Tile-based unsharp masking with optimal memory access patterns
   */
  cv::Mat applyOptimizedUnsharpMask(const cv::Mat& img, double sigma = 1.0,
                                    double strength = 1.5) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    // Step 1: Create blurred version using optimized Gaussian blur
    cv::Mat blurred = applyOptimizedGaussianBlur(img, sigma);

    // Convert to float for precision
    cv::Mat imgFloat, blurredFloat;
    img.convertTo(imgFloat, CV_32FC3, 1.0 / 255.0);
    blurred.convertTo(blurredFloat, CV_32FC3, 1.0 / 255.0);

    cv::Mat result(img.size(), CV_32FC3);

    // Calculate optimal tile dimensions
    int tilesX = (img.cols + OPTIMAL_TILE_SIZE - 1) / OPTIMAL_TILE_SIZE;
    int tilesY = (img.rows + OPTIMAL_TILE_SIZE - 1) / OPTIMAL_TILE_SIZE;
    int totalTiles = tilesX * tilesY;

    auto convStart = std::chrono::high_resolution_clock::now();

// Tile-based parallel processing with optimal work distribution
// Use static scheduling for better performance (tiles are uniform)
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int tileIdx = 0; tileIdx < totalTiles; tileIdx++) {
      int tileY = tileIdx / tilesX;
      int tileX = tileIdx % tilesX;

      int startY = tileY * OPTIMAL_TILE_SIZE;
      int startX = tileX * OPTIMAL_TILE_SIZE;
      int endY = std::min(startY + OPTIMAL_TILE_SIZE, img.rows);
      int endX = std::min(startX + OPTIMAL_TILE_SIZE, img.cols);

      // Process tile with cache-friendly access pattern
      for (int y = startY; y < endY; y++) {
        const cv::Vec3f* originalRow = imgFloat.ptr<cv::Vec3f>(y);
        const cv::Vec3f* blurredRow = blurredFloat.ptr<cv::Vec3f>(y);
        cv::Vec3f* resultRow = result.ptr<cv::Vec3f>(y);

        for (int x = startX; x < endX; x++) {
          cv::Vec3f original = originalRow[x];
          cv::Vec3f blur = blurredRow[x];

          // Unsharp mask: original + strength * (original - blurred)
          cv::Vec3f mask = original - blur;
          cv::Vec3f enhanced = original + strength * mask;

          // Clamp values to valid range
          for (int c = 0; c < 3; c++) {
            enhanced[c] = std::max(0.0f, std::min(1.0f, enhanced[c]));
          }

          resultRow[x] = enhanced;
        }
      }
    }

    auto convEnd = std::chrono::high_resolution_clock::now();
    perfCounters.convolutionTime +=
        std::chrono::duration<double, std::milli>(convEnd - convStart).count();

    // Convert back to 8-bit
    cv::Mat finalResult;
    result.convertTo(finalResult, CV_8UC3, 255.0);

    auto totalEnd = std::chrono::high_resolution_clock::now();
    perfCounters.totalTime +=
        std::chrono::duration<double, std::milli>(totalEnd - totalStart)
            .count();

    if (verbose)
      std::cout << "Applied Optimized Tile-based Unsharp Mask (OpenMP)\n";
    return finalResult;
  }

  /**
   * Memory-efficient bilateral filtering with spatial-color separation
   */
  cv::Mat applyOptimizedBilateralFilter(const cv::Mat& img, int d = 9,
                                        double sigmaColor = 75,
                                        double sigmaSpace = 75) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    cv::Mat result(img.size(), CV_8UC3);
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32FC3, 1.0 / 255.0);

    int radius = d / 2;

    // Precompute spatial weights (Gaussian)
    std::vector<std::vector<float>> spatialWeights(d, std::vector<float>(d));
    float spatialSigmaInv = 1.0f / (2.0f * sigmaSpace * sigmaSpace);

    for (int dy = -radius; dy <= radius; dy++) {
      for (int dx = -radius; dx <= radius; dx++) {
        float spatialDist = dx * dx + dy * dy;
        spatialWeights[dy + radius][dx + radius] =
            std::exp(-spatialDist * spatialSigmaInv);
      }
    }

    // Color sigma for bilateral weight calculation
    float colorSigmaInv =
        1.0f / (2.0f * sigmaColor * sigmaColor / (255.0f * 255.0f));

    auto convStart = std::chrono::high_resolution_clock::now();

// Parallel processing with optimized memory access
// Static scheduling for uniform row processing
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < img.rows; y++) {
      cv::Vec3b* resultRow = result.ptr<cv::Vec3b>(y);
      const cv::Vec3f* centerRow = imgFloat.ptr<cv::Vec3f>(y);

      for (int x = 0; x < img.cols; x++) {
        cv::Vec3f centerPixel = centerRow[x];
        cv::Vec3f weightedSum(0, 0, 0);
        float totalWeight = 0.0f;

        // Bilateral filtering within neighborhood
        for (int dy = -radius; dy <= radius; dy++) {
          int ny = std::max(0, std::min(y + dy, img.rows - 1));
          const cv::Vec3f* neighborRow = imgFloat.ptr<cv::Vec3f>(ny);

          for (int dx = -radius; dx <= radius; dx++) {
            int nx = std::max(0, std::min(x + dx, img.cols - 1));
            cv::Vec3f neighborPixel = neighborRow[nx];

            // Calculate color distance
            cv::Vec3f colorDiff = centerPixel - neighborPixel;
            float colorDist = colorDiff.dot(colorDiff);

            // Combined spatial and color weight
            float spatialWeight = spatialWeights[dy + radius][dx + radius];
            float colorWeight = std::exp(-colorDist * colorSigmaInv);
            float totalWeightPixel = spatialWeight * colorWeight;

            weightedSum += neighborPixel * totalWeightPixel;
            totalWeight += totalWeightPixel;
          }
        }

        // Normalize and convert to 8-bit
        if (totalWeight > 0) {
          cv::Vec3f filtered = weightedSum / totalWeight;
          resultRow[x] =
              cv::Vec3b(cv::saturate_cast<uchar>(filtered[0] * 255.0f),
                        cv::saturate_cast<uchar>(filtered[1] * 255.0f),
                        cv::saturate_cast<uchar>(filtered[2] * 255.0f));
        } else {
          resultRow[x] = img.ptr<cv::Vec3b>(y)[x];
        }
      }
    }

    auto convEnd = std::chrono::high_resolution_clock::now();
    perfCounters.convolutionTime +=
        std::chrono::duration<double, std::milli>(convEnd - convStart).count();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    perfCounters.totalTime +=
        std::chrono::duration<double, std::milli>(totalEnd - totalStart)
            .count();

    if (verbose)
      std::cout
          << "Applied Optimized Memory-Efficient Bilateral Filter (OpenMP)\n";
    return result;
  }

  /**
   * Apply CLAHE with parallel channel processing
   */
  cv::Mat applyOptimizedCLAHE(const cv::Mat& img, double clipLimit = 2.0,
                              cv::Size tileGridSize = cv::Size(8, 8)) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    cv::Mat result;
    std::vector<cv::Mat> bgrChannels(3);
    std::vector<cv::Mat> processedChannels(3);

    cv::split(img, bgrChannels);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);

    auto convStart = std::chrono::high_resolution_clock::now();

// Parallel channel processing
#pragma omp parallel for schedule(static) num_threads(3)
    for (int i = 0; i < 3; ++i) {
      clahe->apply(bgrChannels[i], processedChannels[i]);
    }

    auto convEnd = std::chrono::high_resolution_clock::now();
    perfCounters.convolutionTime +=
        std::chrono::duration<double, std::milli>(convEnd - convStart).count();

    cv::merge(processedChannels, result);

    auto totalEnd = std::chrono::high_resolution_clock::now();
    perfCounters.totalTime +=
        std::chrono::duration<double, std::milli>(totalEnd - totalStart)
            .count();

    if (verbose) std::cout << "Applied Optimized CLAHE (OpenMP)\n";
    return result;
  }

  /**
   * Apply edge enhancement with optimized parallel processing
   */
  cv::Mat applyOptimizedEdgeEnhance(const cv::Mat& img, double strength = 1.0) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    cv::Mat gray, edges, result;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 100, 200);

    result = img.clone();

    auto convStart = std::chrono::high_resolution_clock::now();

// Parallel edge enhancement
// Static scheduling for uniform row processing
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < img.rows; y++) {
      const cv::Vec3b* originalRow = img.ptr<cv::Vec3b>(y);
      const uchar* edgeRow = edges.ptr<uchar>(y);
      cv::Vec3b* resultRow = result.ptr<cv::Vec3b>(y);

      for (int x = 0; x < img.cols; x++) {
        cv::Vec3b original = originalRow[x];
        float edgeStrength = edgeRow[x] / 255.0f;

        for (int c = 0; c < 3; c++) {
          float enhanced = original[c] + strength * edgeStrength * 50.0f;
          resultRow[x][c] = cv::saturate_cast<uchar>(enhanced);
        }
      }
    }

    auto convEnd = std::chrono::high_resolution_clock::now();
    perfCounters.convolutionTime +=
        std::chrono::duration<double, std::milli>(convEnd - convStart).count();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    perfCounters.totalTime +=
        std::chrono::duration<double, std::milli>(totalEnd - totalStart)
            .count();

    if (verbose) std::cout << "Applied Optimized Edge Enhancement (OpenMP)\n";
    return result;
  }

  /**
   * Assess image quality and suggest appropriate filter
   */
  FilterType assessImageQuality(const cv::Mat& img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Calculate Laplacian variance (measure of blurriness)
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double variance = stddev.val[0] * stddev.val[0];

    // Calculate brightness
    cv::Scalar meanBrightness = cv::mean(gray);
    double brightness = meanBrightness.val[0];

    // Calculate noise level
    cv::Mat noise, diff;
    cv::Mat kernel = cv::getGaussianKernel(5, 1.0);
    cv::filter2D(gray, noise, CV_64F, kernel);

    cv::Mat grayDouble;
    gray.convertTo(grayDouble, CV_64F);
    diff = grayDouble - noise;

    cv::Scalar noiseMean, noiseStd;
    cv::meanStdDev(diff, noiseMean, noiseStd);
    double noiseLevel = noiseStd.val[0];

    if (verbose) {
      std::cout << "Image Quality Assessment:\n";
      std::cout << "  Blur variance: " << std::fixed << std::setprecision(1)
                << variance << " (>100 = sharp, <100 = blurry)\n";
      std::cout << "  Brightness: " << std::fixed << std::setprecision(1)
                << brightness << " (0-255)\n";
      std::cout << "  Noise level: " << std::fixed << std::setprecision(1)
                << noiseLevel << "\n";
    }

    // Decision logic
    if (variance < 100) {
      std::cout
          << "  Recommendation: Image appears blurry - applying sharpening\n";
      return FilterType::UNSHARP_MASK;
    } else if (noiseLevel > 15) {
      std::cout
          << "  Recommendation: Image appears noisy - applying denoising\n";
      return FilterType::BILATERAL_DENOISE;
    } else if (brightness < 50 || brightness > 200) {
      std::cout << "  Recommendation: Poor contrast - applying CLAHE\n";
      return FilterType::CLAHE_ENHANCE;
    } else {
      std::cout
          << "  Recommendation: Good quality - applying edge enhancement\n";
      return FilterType::EDGE_ENHANCE;
    }
  }

  /**
   * Apply the optimized filter based on type
   */
  cv::Mat processImage(const cv::Mat& img,
                       FilterType filter = FilterType::GAUSSIAN_BLUR) {
    // Reset performance counters
    perfCounters = PerformanceCounters();

    cv::Mat result;
    switch (filter) {
      case FilterType::GAUSSIAN_BLUR:
        result = applyOptimizedGaussianBlur(img);
        break;
      case FilterType::UNSHARP_MASK:
        result = applyOptimizedUnsharpMask(img);
        break;
      case FilterType::BILATERAL_DENOISE:
        result = applyOptimizedBilateralFilter(img);
        break;
      case FilterType::CLAHE_ENHANCE:
        result = applyOptimizedCLAHE(img);
        break;
      case FilterType::EDGE_ENHANCE:
        result = applyOptimizedEdgeEnhance(img);
        break;
      default:
        result = applyOptimizedGaussianBlur(img);
    }

    return result;
  }

  /**
   * Get detailed performance analysis
   */
  void printPerformanceAnalysis() const {
    if (verbose) {
      std::cout << "\n=== Performance Analysis ===\n";
      std::cout << "Total processing time: " << std::fixed
                << std::setprecision(2) << perfCounters.totalTime << " ms\n";
      std::cout << "Convolution time: " << perfCounters.convolutionTime
                << " ms ("
                << (perfCounters.convolutionTime / perfCounters.totalTime * 100)
                << "%)\n";
      std::cout << "Kernel generation time: " << perfCounters.kernelGenTime
                << " ms\n";
      std::cout << "Memory allocation time: " << perfCounters.memoryAllocTime
                << " ms\n";
      std::cout << "Threads utilized: " << perfCounters.threadsUsed << "\n";
      std::cout << "Processing efficiency: "
                << (100.0 / perfCounters.threadsUsed) << "% per thread\n";
      std::cout << "============================\n";
    }
  }

  const PerformanceCounters& getPerformanceCounters() const {
    return perfCounters;
  }
};

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input_img> <output_img> [filter_type] [auto_assess]\n";
    std::cerr
        << "Filter types: blur, sharpen, laplacian, denoise, clahe, edge\n";
    std::cerr
        << "Auto assess: use 'auto' to automatically choose best filter\n";
    return 1;
  }

  std::string in = argv[1], out = argv[2];
  std::string filterStr = (argc > 3) ? argv[3] : "auto";
  bool autoAssess =
      (filterStr == "auto") || (argc > 4 && std::string(argv[4]) == "auto");

  // Load image
  cv::Mat img = cv::imread(in, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Failed to open " << in << "\n";
    return 1;
  }

  std::cout << "=== Optimized OpenMP Image Preprocessing Pipeline ===\n";
  std::cout << "Input: " << in << " (" << img.cols << "x" << img.rows << ")\n";

  auto start = std::chrono::high_resolution_clock::now();

  OptimizedImagePreprocessor processor(true);
  FilterType selectedFilter;

  if (autoAssess) {
    selectedFilter = processor.assessImageQuality(img);
  } else {
    // Manual filter selection
    if (filterStr == "blur")
      selectedFilter = FilterType::GAUSSIAN_BLUR;
    else if (filterStr == "sharpen")
      selectedFilter = FilterType::UNSHARP_MASK;
    else if (filterStr == "laplacian")
      selectedFilter = FilterType::LAPLACIAN_SHARPEN;
    else if (filterStr == "denoise")
      selectedFilter = FilterType::BILATERAL_DENOISE;
    else if (filterStr == "clahe")
      selectedFilter = FilterType::CLAHE_ENHANCE;
    else if (filterStr == "edge")
      selectedFilter = FilterType::EDGE_ENHANCE;
    else
      selectedFilter = FilterType::GAUSSIAN_BLUR;
  }

  // Apply selected filter
  cv::Mat processed = processor.processImage(img, selectedFilter);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Save result
  if (!cv::imwrite(out, processed)) {
    std::cerr << "Failed to save " << out << "\n";
    return 1;
  }

  int numThreads = 1;
#ifdef _OPENMP
  numThreads = omp_get_max_threads();
#endif

  std::cout << "=== Processing Complete ===\n";
  std::cout << "Output: " << out << "\n";
  std::cout << "Total processing time: " << duration.count() << " ms\n";
  std::cout << "Max threads available: " << numThreads << "\n";

  // Show detailed performance analysis
  processor.printPerformanceAnalysis();

  std::cout << "Ready for YOLO detection!\n";

  return 0;
}
