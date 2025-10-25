#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
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

// Convert FilterType to string for logging
inline std::string filterTypeToString(FilterType type) {
  switch (type) {
    case FilterType::GAUSSIAN_BLUR: return "Gaussian Blur";
    case FilterType::UNSHARP_MASK: return "Unsharp Mask";
    case FilterType::LAPLACIAN_SHARPEN: return "Laplacian Sharpen";
    case FilterType::BILATERAL_DENOISE: return "Bilateral Denoise";
    case FilterType::CLAHE_ENHANCE: return "CLAHE";
    case FilterType::EDGE_ENHANCE: return "Edge Enhancement";
    default: return "Unknown";
  }
}

/**
 * Comprehensive image quality metrics
 * Phase 1: Foundation for intelligent filter selection
 */
struct ImageQualityMetrics {
  // Basic metrics
  double blurVariance;        // Laplacian variance (>100 = sharp)
  double brightness;          // Mean brightness (0-255)
  double noiseLevel;          // Estimated noise level

  // Advanced metrics
  double contrast;            // Standard deviation of brightness
  double dynamicRange;        // Actual intensity range used (0-1)
  bool hasHistogramClipping;  // Over/underexposure detection

  // Image dimensions
  int width;
  int height;

  // Severity assessment
  bool isSeverelyDegraded() const {
    return (brightness < 30 || brightness > 220) ||
           (blurVariance < 50 && noiseLevel > 15) ||
           (dynamicRange < 0.3);
  }

  bool isExtremelyDark() const {
    return brightness < 30;
  }

  bool isExtremelyBright() const {
    return brightness > 220;
  }

  bool isVeryBlurry() const {
    return blurVariance < 50;
  }

  bool isVeryNoisy() const {
    return noiseLevel > 20;
  }

  bool hasLowContrast() const {
    return contrast < 30 || dynamicRange < 0.3;
  }
};

/**
 * Represents a single filter operation with parameters
 * Phase 2: Building block for filter pipelines
 */
struct FilterOperation {
  FilterType type;
  std::map<std::string, double> parameters;
  std::string reason;  // Why this filter was selected

  FilterOperation(FilterType t, std::map<std::string, double> p, std::string r)
      : type(t), parameters(std::move(p)), reason(std::move(r)) {}
};

/**
 * Manages a sequence of filter operations
 * Phase 2: Composable filter pipeline
 */
class FilterPipeline {
private:
  std::vector<FilterOperation> operations;
  std::string pipelineName;

public:
  FilterPipeline() : pipelineName("Custom Pipeline") {}
  explicit FilterPipeline(std::string name) : pipelineName(std::move(name)) {}

  void addFilter(FilterType type, std::map<std::string, double> params,
                 const std::string& reason) {
    operations.emplace_back(type, std::move(params), reason);
  }

  const std::vector<FilterOperation>& getOperations() const {
    return operations;
  }

  void clear() {
    operations.clear();
  }

  bool isEmpty() const {
    return operations.empty();
  }

  size_t size() const {
    return operations.size();
  }

  const std::string& getName() const {
    return pipelineName;
  }

  // Factory methods for common degradation scenarios
  static FilterPipeline createSevereDarkPipeline() {
    FilterPipeline pipeline("Severe Dark Recovery");
    pipeline.addFilter(FilterType::CLAHE_ENHANCE,
                      {{"clipLimit", 3.0}, {"tileSize", 8}},
                      "Severe darkness - aggressive contrast enhancement");
    pipeline.addFilter(FilterType::EDGE_ENHANCE,
                      {{"strength", 1.2}},
                      "Post-CLAHE edge recovery for detail restoration");
    return pipeline;
  }

  static FilterPipeline createSevereBrightPipeline() {
    FilterPipeline pipeline("Severe Bright Recovery");
    pipeline.addFilter(FilterType::CLAHE_ENHANCE,
                      {{"clipLimit", 2.5}, {"tileSize", 8}},
                      "Overexposure correction");
    pipeline.addFilter(FilterType::EDGE_ENHANCE,
                      {{"strength", 1.0}},
                      "Edge recovery after exposure correction");
    return pipeline;
  }

  static FilterPipeline createNoisyBlurPipeline() {
    FilterPipeline pipeline("Noisy & Blurry Recovery");
    pipeline.addFilter(FilterType::BILATERAL_DENOISE,
                      {{"d", 9}, {"sigmaColor", 75}, {"sigmaSpace", 75}},
                      "Denoise while preserving edges");
    pipeline.addFilter(FilterType::UNSHARP_MASK,
                      {{"sigma", 1.0}, {"strength", 1.5}},
                      "Sharpen after denoising");
    return pipeline;
  }

  static FilterPipeline createLowContrastPipeline() {
    FilterPipeline pipeline("Low Contrast Recovery");
    pipeline.addFilter(FilterType::CLAHE_ENHANCE,
                      {{"clipLimit", 2.0}, {"tileSize", 8}},
                      "Adaptive contrast enhancement");
    return pipeline;
  }

  static FilterPipeline createStandardPipeline(FilterType singleFilter) {
    FilterPipeline pipeline("Single Filter");
    std::map<std::string, double> defaultParams;

    switch (singleFilter) {
      case FilterType::GAUSSIAN_BLUR:
        defaultParams = {{"sigma", 2.0}};  // Increased from 1.0 for better parallelization
        break;
      case FilterType::UNSHARP_MASK:
        defaultParams = {{"sigma", 2.0}, {"strength", 1.5}};  // Increased from 1.0
        break;
      case FilterType::BILATERAL_DENOISE:
        defaultParams = {{"d", 9}, {"sigmaColor", 75}, {"sigmaSpace", 75}};
        break;
      case FilterType::CLAHE_ENHANCE:
        defaultParams = {{"clipLimit", 2.0}, {"tileSize", 8}};
        break;
      case FilterType::EDGE_ENHANCE:
        defaultParams = {{"strength", 1.0}};
        break;
      default:
        defaultParams = {};
    }

    pipeline.addFilter(singleFilter, defaultParams, "User selected");
    return pipeline;
  }
};

/**
 * Configuration for preprocessing behavior
 * Phase 5: Observability and control
 */
struct PreprocessingConfig {
  bool enableMultiFilter = true;        // Enable multi-filter pipelines
  bool saveIntermediateResults = false; // Save intermediate images for debugging
  int maxPipelineStages = 3;            // Prevent runaway pipelines
  std::string intermediateDir = "./debug/";

  enum class Strategy {
    CONSERVATIVE,  // Minimal processing
    BALANCED,      // Default - intelligent multi-filter
    AGGRESSIVE     // Maximum quality improvement
  };
  Strategy strategy = Strategy::BALANCED;
};

/**
 * Optimized ImagePreprocessor with proper OpenMP implementations
 * Now supports intelligent multi-filter pipelines for severely degraded images
 */
class OptimizedImagePreprocessor {
 private:
  bool verbose;
  PreprocessingConfig config;

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

  // Pipeline performance tracking (Phase 4)
  struct StagePerformance {
    FilterType filter;
    double processingTime;
    std::string reason;
    int threadsUsed;
  };

  std::vector<StagePerformance> pipelinePerformance;

  // Optimal tile sizes for cache efficiency
  static constexpr int CACHE_LINE_SIZE = 64;
  static constexpr int OPTIMAL_TILE_SIZE = 64;  // 64x64 fits L1 cache
  static constexpr int PIXELS_PER_CACHE_LINE =
      CACHE_LINE_SIZE / sizeof(cv::Vec3b);

 public:
  OptimizedImagePreprocessor(bool verbose = true, PreprocessingConfig cfg = PreprocessingConfig())
      : verbose(verbose), config(std::move(cfg)) {}

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

    auto convStart = std::chrono::high_resolution_clock::now();

// Row-based parallel processing (simpler and faster than tiling for this operation)
// Static scheduling for uniform work distribution with less overhead
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < img.rows; y++) {
      const cv::Vec3f* originalRow = imgFloat.ptr<cv::Vec3f>(y);
      const cv::Vec3f* blurredRow = blurredFloat.ptr<cv::Vec3f>(y);
      cv::Vec3f* resultRow = result.ptr<cv::Vec3f>(y);

      for (int x = 0; x < img.cols; x++) {
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
      std::cout << "Applied Optimized Row-based Unsharp Mask (OpenMP)\n";
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
    perfCounters.threadsUsed = omp_get_max_threads();

    if (verbose)
      std::cout
          << "Applied Optimized Memory-Efficient Bilateral Filter (OpenMP)\n";
    return result;
  }

  /**
   * Parallel BGR to LAB color conversion
   * Optimized to avoid serial OpenCV color conversion bottleneck
   */
  cv::Mat parallelBGR2Lab(const cv::Mat& bgr) {
    cv::Mat lab(bgr.size(), CV_8UC3);

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < bgr.rows; y++) {
      const cv::Vec3b* bgrRow = bgr.ptr<cv::Vec3b>(y);
      cv::Vec3b* labRow = lab.ptr<cv::Vec3b>(y);

      for (int x = 0; x < bgr.cols; x++) {
        // Extract BGR values
        float b = bgrRow[x][0] / 255.0f;
        float g = bgrRow[x][1] / 255.0f;
        float r = bgrRow[x][2] / 255.0f;

        // BGR to XYZ (using sRGB D65 transformation)
        // Apply gamma correction
        auto toLinear = [](float c) {
          return (c > 0.04045f) ? std::pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
        };
        r = toLinear(r);
        g = toLinear(g);
        b = toLinear(b);

        // XYZ transformation matrix (D65 illuminant)
        float xyz_x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
        float xyz_y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
        float xyz_z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

        // XYZ to LAB (D65 white point: Xn=0.95047, Yn=1.0, Zn=1.08883)
        xyz_x /= 0.95047f;
        xyz_y /= 1.00000f;
        xyz_z /= 1.08883f;

        auto f = [](float t) {
          return (t > 0.008856f) ? std::cbrt(t) : (7.787f * t + 16.0f / 116.0f);
        };

        float fx = f(xyz_x);
        float fy = f(xyz_y);
        float fz = f(xyz_z);

        float L = 116.0f * fy - 16.0f;
        float a = 500.0f * (fx - fy);
        float b_lab = 200.0f * (fy - fz);

        // Scale to 0-255 range (OpenCV LAB format)
        labRow[x][0] = cv::saturate_cast<uchar>(L * 255.0f / 100.0f);
        labRow[x][1] = cv::saturate_cast<uchar>(a + 128.0f);
        labRow[x][2] = cv::saturate_cast<uchar>(b_lab + 128.0f);
      }
    }

    return lab;
  }

  /**
   * Parallel LAB to BGR color conversion
   * Optimized to avoid serial OpenCV color conversion bottleneck
   */
  cv::Mat parallelLab2BGR(const cv::Mat& lab) {
    cv::Mat bgr(lab.size(), CV_8UC3);

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < lab.rows; y++) {
      const cv::Vec3b* labRow = lab.ptr<cv::Vec3b>(y);
      cv::Vec3b* bgrRow = bgr.ptr<cv::Vec3b>(y);

      for (int x = 0; x < lab.cols; x++) {
        // Extract LAB values and scale back
        float L = labRow[x][0] * 100.0f / 255.0f;
        float a = static_cast<float>(labRow[x][1]) - 128.0f;
        float b_lab = static_cast<float>(labRow[x][2]) - 128.0f;

        // LAB to XYZ
        float fy = (L + 16.0f) / 116.0f;
        float fx = a / 500.0f + fy;
        float fz = fy - b_lab / 200.0f;

        auto finv = [](float t) {
          float t3 = t * t * t;
          return (t3 > 0.008856f) ? t3 : (t - 16.0f / 116.0f) / 7.787f;
        };

        float xyz_x = finv(fx) * 0.95047f;
        float xyz_y = finv(fy) * 1.00000f;
        float xyz_z = finv(fz) * 1.08883f;

        // XYZ to RGB
        float r = xyz_x *  3.2404542f + xyz_y * -1.5371385f + xyz_z * -0.4985314f;
        float g = xyz_x * -0.9692660f + xyz_y *  1.8760108f + xyz_z *  0.0415560f;
        float b = xyz_x *  0.0556434f + xyz_y * -0.2040259f + xyz_z *  1.0572252f;

        // Apply gamma correction (sRGB)
        auto toSRGB = [](float c) {
          return (c > 0.0031308f) ? 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f : 12.92f * c;
        };
        r = toSRGB(r);
        g = toSRGB(g);
        b = toSRGB(b);

        // Convert to 0-255 range
        bgrRow[x][0] = cv::saturate_cast<uchar>(b * 255.0f);
        bgrRow[x][1] = cv::saturate_cast<uchar>(g * 255.0f);
        bgrRow[x][2] = cv::saturate_cast<uchar>(r * 255.0f);
      }
    }

    return bgr;
  }

  /**
   * Apply CLAHE with parallel pixel-level histogram equalization
   */
  cv::Mat applyOptimizedCLAHE(const cv::Mat& img, double clipLimit = 2.0,
                              cv::Size tileGridSize = cv::Size(8, 8)) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    auto colorConvStart = std::chrono::high_resolution_clock::now();
    // Convert to LAB color space using parallel implementation
    cv::Mat lab = parallelBGR2Lab(img);

    std::vector<cv::Mat> labChannels(3);
    cv::split(lab, labChannels);
    auto colorConvEnd = std::chrono::high_resolution_clock::now();
    double colorConvTime = std::chrono::duration<double, std::milli>(colorConvEnd - colorConvStart).count();

    auto convStart = std::chrono::high_resolution_clock::now();

    // Apply parallel CLAHE to L channel using row-based parallelism
    cv::Mat lChannel = labChannels[0];
    cv::Mat lProcessed(lChannel.size(), lChannel.type());

    // Compute histogram for each tile (parallel)
    int tileHeight = lChannel.rows / tileGridSize.height;
    int tileWidth = lChannel.cols / tileGridSize.width;

    // Store histograms for all tiles
    std::vector<std::vector<int>> tileHistograms(
        tileGridSize.height * tileGridSize.width, std::vector<int>(256, 0));
    std::vector<std::vector<int>> tileCDFs(
        tileGridSize.height * tileGridSize.width, std::vector<int>(256, 0));

    // Phase 1: Build histograms in parallel (no critical section needed!)
    auto phase1Start = std::chrono::high_resolution_clock::now();
    int totalTiles = tileGridSize.height * tileGridSize.width;
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int tileIdx = 0; tileIdx < totalTiles; ++tileIdx) {
      int tileY = tileIdx / tileGridSize.width;
      int tileX = tileIdx % tileGridSize.width;

      int startY = tileY * tileHeight;
      int startX = tileX * tileWidth;
      int endY = std::min(startY + tileHeight, lChannel.rows);
      int endX = std::min(startX + tileWidth, lChannel.cols);

      // Build histogram for this tile
      std::vector<int>& hist = tileHistograms[tileIdx];
      for (int y = startY; y < endY; y++) {
        const uchar* row = lChannel.ptr<uchar>(y);
        for (int x = startX; x < endX; x++) {
          hist[row[x]]++;
        }
      }

      // Apply clipping
      int clipLimitPixels = static_cast<int>(
          clipLimit * (endY - startY) * (endX - startX) / 256.0);
      int clippedPixels = 0;
      for (int i = 0; i < 256; i++) {
        if (hist[i] > clipLimitPixels) {
          clippedPixels += hist[i] - clipLimitPixels;
          hist[i] = clipLimitPixels;
        }
      }

      // Redistribute clipped pixels
      int redistribution = clippedPixels / 256;
      for (int i = 0; i < 256; i++) {
        hist[i] += redistribution;
      }

      // Build CDF
      std::vector<int>& cdf = tileCDFs[tileIdx];
      cdf[0] = hist[0];
      for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
      }

      // Normalize CDF to [0, 255]
      int totalPixels = (endY - startY) * (endX - startX);
      if (totalPixels > 0) {
        for (int i = 0; i < 256; i++) {
          cdf[i] = (cdf[i] * 255) / totalPixels;
        }
      }
    }
    auto phase1End = std::chrono::high_resolution_clock::now();
    double phase1Time = std::chrono::duration<double, std::milli>(phase1End - phase1Start).count();

    // Phase 2: Apply equalization with bilinear interpolation (parallel by rows)
    auto phase2Start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < lChannel.rows; y++) {
      const uchar* srcRow = lChannel.ptr<uchar>(y);
      uchar* dstRow = lProcessed.ptr<uchar>(y);

      for (int x = 0; x < lChannel.cols; x++) {
        // Find which tile this pixel belongs to
        int tileY = std::min(y / tileHeight, tileGridSize.height - 1);
        int tileX = std::min(x / tileWidth, tileGridSize.width - 1);
        int tileIdx = tileY * tileGridSize.width + tileX;

        // Apply equalization using the tile's CDF
        uchar pixelValue = srcRow[x];
        dstRow[x] = static_cast<uchar>(tileCDFs[tileIdx][pixelValue]);
      }
    }
    auto phase2End = std::chrono::high_resolution_clock::now();
    double phase2Time = std::chrono::duration<double, std::milli>(phase2End - phase2Start).count();

    // Merge channels back
    labChannels[0] = lProcessed;

    auto convEnd = std::chrono::high_resolution_clock::now();
    perfCounters.convolutionTime +=
        std::chrono::duration<double, std::milli>(convEnd - convStart).count();

    auto colorConvBackStart = std::chrono::high_resolution_clock::now();
    cv::merge(labChannels, lab);
    cv::Mat result = parallelLab2BGR(lab);
    auto colorConvBackEnd = std::chrono::high_resolution_clock::now();
    double colorConvBackTime = std::chrono::duration<double, std::milli>(colorConvBackEnd - colorConvBackStart).count();

    if (verbose) {
      std::cout << "  [CLAHE Timing] Color conv BGR→LAB: " << colorConvTime << "ms\n";
      std::cout << "  [CLAHE Timing] Phase 1 (histograms): " << phase1Time << "ms\n";
      std::cout << "  [CLAHE Timing] Phase 2 (apply): " << phase2Time << "ms\n";
      std::cout << "  [CLAHE Timing] Color conv LAB→BGR: " << colorConvBackTime << "ms\n";
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    perfCounters.totalTime +=
        std::chrono::duration<double, std::milli>(totalEnd - totalStart)
            .count();
    perfCounters.threadsUsed = omp_get_max_threads();

    if (verbose) std::cout << "Applied Optimized Parallel CLAHE (OpenMP)\n";
    return result;
  }

  /**
   * Apply edge enhancement with parallel Sobel gradient computation
   */
  cv::Mat applyOptimizedEdgeEnhance(const cv::Mat& img, double strength = 1.0) {
    auto totalStart = std::chrono::high_resolution_clock::now();

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Convert to float for precise computation
    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F, 1.0 / 255.0);

    // Create gradient magnitude map
    cv::Mat gradientMag(gray.size(), CV_32F);

    auto convStart = std::chrono::high_resolution_clock::now();

    // Sobel kernels (3x3)
    const int sobelKernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int sobelKernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// Parallel Sobel gradient computation
// Use static scheduling for uniform row processing
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 1; y < gray.rows - 1; y++) {
      float* gradRow = gradientMag.ptr<float>(y);

      for (int x = 1; x < gray.cols - 1; x++) {
        float gx = 0.0f, gy = 0.0f;

        // Apply Sobel kernels
        for (int ky = -1; ky <= 1; ky++) {
          const float* srcRow = grayFloat.ptr<float>(y + ky);
          for (int kx = -1; kx <= 1; kx++) {
            float pixel = srcRow[x + kx];
            gx += pixel * sobelKernelX[ky + 1][kx + 1];
            gy += pixel * sobelKernelY[ky + 1][kx + 1];
          }
        }

        // Gradient magnitude (L2 norm)
        gradRow[x] = std::sqrt(gx * gx + gy * gy);
      }
    }

    // Handle borders (set to 0)
    for (int x = 0; x < gray.cols; x++) {
      gradientMag.at<float>(0, x) = 0;
      gradientMag.at<float>(gray.rows - 1, x) = 0;
    }
    for (int y = 0; y < gray.rows; y++) {
      gradientMag.at<float>(y, 0) = 0;
      gradientMag.at<float>(y, gray.cols - 1) = 0;
    }

    // Normalize gradient magnitude
    double minVal, maxVal;
    cv::minMaxLoc(gradientMag, &minVal, &maxVal);
    if (maxVal > 0) {
      gradientMag /= maxVal;
    }

    // Convert original image to float for blending
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32FC3, 1.0 / 255.0);

    cv::Mat result(img.size(), CV_32FC3);

// Parallel edge enhancement blending
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = 0; y < img.rows; y++) {
      const cv::Vec3f* originalRow = imgFloat.ptr<cv::Vec3f>(y);
      const float* edgeRow = gradientMag.ptr<float>(y);
      cv::Vec3f* resultRow = result.ptr<cv::Vec3f>(y);

      for (int x = 0; x < img.cols; x++) {
        cv::Vec3f original = originalRow[x];
        float edgeStrength = edgeRow[x];

        // Enhance edges by adding gradient-weighted brightness
        for (int c = 0; c < 3; c++) {
          float enhanced = original[c] + strength * edgeStrength * 0.3f;
          resultRow[x][c] = std::max(0.0f, std::min(1.0f, enhanced));
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
    perfCounters.threadsUsed = omp_get_max_threads();

    if (verbose) std::cout << "Applied Optimized Parallel Sobel Edge Enhancement (OpenMP)\n";
    return finalResult;
  }

  /**
   * Phase 1: Comprehensive image quality analysis
   * Returns detailed metrics for intelligent filter selection
   */
  ImageQualityMetrics analyzeImage(const cv::Mat& img) const {
    ImageQualityMetrics metrics;
    metrics.width = img.cols;
    metrics.height = img.rows;

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Calculate Laplacian variance (measure of blurriness)
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    metrics.blurVariance = stddev.val[0] * stddev.val[0];

    // Calculate brightness and contrast
    cv::Scalar meanBrightness, stddevBrightness;
    cv::meanStdDev(gray, meanBrightness, stddevBrightness);
    metrics.brightness = meanBrightness.val[0];
    metrics.contrast = stddevBrightness.val[0];

    // Calculate noise level
    cv::Mat noise, diff;
    cv::Mat kernel = cv::getGaussianKernel(5, 1.0);
    cv::filter2D(gray, noise, CV_64F, kernel);

    cv::Mat grayDouble;
    gray.convertTo(grayDouble, CV_64F);
    diff = grayDouble - noise;

    cv::Scalar noiseMean, noiseStd;
    cv::meanStdDev(diff, noiseMean, noiseStd);
    metrics.noiseLevel = noiseStd.val[0];

    // Calculate dynamic range
    double minVal, maxVal;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    metrics.dynamicRange = (maxVal - minVal) / 255.0;

    // Detect histogram clipping (over/underexposure)
    std::vector<int> histogram(256, 0);
    for (int y = 0; y < gray.rows; y++) {
      const uchar* row = gray.ptr<uchar>(y);
      for (int x = 0; x < gray.cols; x++) {
        histogram[row[x]]++;
      }
    }

    int totalPixels = gray.rows * gray.cols;
    int clippedPixels = histogram[0] + histogram[255];
    metrics.hasHistogramClipping = (clippedPixels > totalPixels * 0.01);

    return metrics;
  }

  /**
   * Phase 3: Intelligent pipeline selection based on degradation profile
   * Maps image quality metrics to optimal filter pipelines
   */
  FilterPipeline selectOptimalPipeline(const ImageQualityMetrics& metrics) const {
    if (verbose) {
      std::cout << "\n=== Image Quality Analysis ===\n";
      std::cout << "Brightness: " << std::fixed << std::setprecision(1)
                << metrics.brightness;
      if (metrics.isExtremelyDark())
        std::cout << " (SEVERE - extremely dark)";
      else if (metrics.isExtremelyBright())
        std::cout << " (SEVERE - extremely bright)";
      else if (metrics.brightness < 50 || metrics.brightness > 200)
        std::cout << " (poor)";
      std::cout << "\n";

      std::cout << "Blur Variance: " << metrics.blurVariance;
      if (metrics.isVeryBlurry())
        std::cout << " (SEVERE - very blurry)";
      else if (metrics.blurVariance < 100)
        std::cout << " (moderate blur)";
      std::cout << "\n";

      std::cout << "Noise Level: " << metrics.noiseLevel;
      if (metrics.isVeryNoisy())
        std::cout << " (SEVERE - very noisy)";
      else if (metrics.noiseLevel > 15)
        std::cout << " (moderate noise)";
      std::cout << "\n";

      std::cout << "Contrast: " << metrics.contrast;
      if (metrics.hasLowContrast())
        std::cout << " (low)";
      std::cout << "\n";

      std::cout << "Dynamic Range: " << std::fixed << std::setprecision(2)
                << metrics.dynamicRange;
      if (metrics.dynamicRange < 0.3)
        std::cout << " (poor)";
      std::cout << "\n";

      if (metrics.hasHistogramClipping) {
        std::cout << "Histogram clipping detected (over/underexposure)\n";
      }
    }

    // Multi-filter mode enabled and severely degraded image
    if (config.enableMultiFilter && metrics.isSeverelyDegraded()) {
      if (verbose) {
        std::cout << "Assessment: SEVERELY_DEGRADED\n";
        std::cout << "\n=== Pipeline Selection ===\n";
        std::cout << "Strategy: ";
        switch (config.strategy) {
          case PreprocessingConfig::Strategy::CONSERVATIVE:
            std::cout << "CONSERVATIVE\n";
            break;
          case PreprocessingConfig::Strategy::BALANCED:
            std::cout << "BALANCED\n";
            break;
          case PreprocessingConfig::Strategy::AGGRESSIVE:
            std::cout << "AGGRESSIVE\n";
            break;
        }
      }

      // Priority 1: Extreme darkness
      if (metrics.isExtremelyDark()) {
        auto pipeline = FilterPipeline::createSevereDarkPipeline();
        if (verbose) {
          std::cout << "Selected: " << pipeline.getName() << " ("
                    << pipeline.size() << " stages)\n";
          int stage = 1;
          for (const auto& op : pipeline.getOperations()) {
            std::cout << "  Stage " << stage++ << ": "
                      << filterTypeToString(op.type) << " - " << op.reason
                      << "\n";
          }
        }
        return pipeline;
      }

      // Priority 2: Extreme brightness
      if (metrics.isExtremelyBright()) {
        auto pipeline = FilterPipeline::createSevereBrightPipeline();
        if (verbose) {
          std::cout << "Selected: " << pipeline.getName() << " ("
                    << pipeline.size() << " stages)\n";
          int stage = 1;
          for (const auto& op : pipeline.getOperations()) {
            std::cout << "  Stage " << stage++ << ": "
                      << filterTypeToString(op.type) << " - " << op.reason
                      << "\n";
          }
        }
        return pipeline;
      }

      // Priority 3: Noisy AND blurry
      if (metrics.isVeryBlurry() && metrics.noiseLevel > 15) {
        auto pipeline = FilterPipeline::createNoisyBlurPipeline();
        if (verbose) {
          std::cout << "Selected: " << pipeline.getName() << " ("
                    << pipeline.size() << " stages)\n";
          int stage = 1;
          for (const auto& op : pipeline.getOperations()) {
            std::cout << "  Stage " << stage++ << ": "
                      << filterTypeToString(op.type) << " - " << op.reason
                      << "\n";
          }
        }
        return pipeline;
      }

      // Priority 4: Low contrast/dynamic range
      if (metrics.hasLowContrast()) {
        auto pipeline = FilterPipeline::createLowContrastPipeline();
        if (verbose) {
          std::cout << "Selected: " << pipeline.getName() << " ("
                    << pipeline.size() << " stages)\n";
          for (const auto& op : pipeline.getOperations()) {
            std::cout << "  Stage 1: " << filterTypeToString(op.type) << " - "
                      << op.reason << "\n";
          }
        }
        return pipeline;
      }
    }

    // Single filter fallback (existing logic)
    if (verbose) {
      std::cout << "Assessment: STANDARD\n";
      std::cout << "\n=== Filter Selection ===\n";
    }

    FilterType selectedFilter;
    if (metrics.blurVariance < 100) {
      if (verbose)
        std::cout << "Selected: Unsharp Mask - Image appears blurry\n";
      selectedFilter = FilterType::UNSHARP_MASK;
    } else if (metrics.noiseLevel > 15) {
      if (verbose)
        std::cout << "Selected: Bilateral Denoise - Image appears noisy\n";
      selectedFilter = FilterType::BILATERAL_DENOISE;
    } else if (metrics.brightness < 50 || metrics.brightness > 200) {
      if (verbose)
        std::cout << "Selected: CLAHE - Poor contrast detected\n";
      selectedFilter = FilterType::CLAHE_ENHANCE;
    } else {
      if (verbose)
        std::cout
            << "Selected: Edge Enhancement - Good quality, light processing\n";
      selectedFilter = FilterType::EDGE_ENHANCE;
    }

    return FilterPipeline::createStandardPipeline(selectedFilter);
  }

  /**
   * Backwards compatibility: Assess image quality and suggest single filter
   * Now wraps the new analyzeImage + selectOptimalPipeline methods
   */
  FilterType assessImageQuality(const cv::Mat& img) {
    auto metrics = analyzeImage(img);

    // Force single-filter mode for backwards compatibility
    bool originalMultiFilter = config.enableMultiFilter;
    config.enableMultiFilter = false;

    auto pipeline = selectOptimalPipeline(metrics);

    config.enableMultiFilter = originalMultiFilter;

    // Return the first (and only) filter from the pipeline
    if (!pipeline.isEmpty()) {
      return pipeline.getOperations()[0].type;
    }

    return FilterType::EDGE_ENHANCE;  // Fallback
  }

  /**
   * Phase 4: Helper method to apply a single filter with parameters
   */
  cv::Mat applySingleFilter(const cv::Mat& img, FilterType type,
                           const std::map<std::string, double>& params) {
    cv::Mat result;

    // Helper to get parameter with default
    auto getParam = [&params](const std::string& key, double defaultVal) {
      auto it = params.find(key);
      return (it != params.end()) ? it->second : defaultVal;
    };

    switch (type) {
      case FilterType::GAUSSIAN_BLUR: {
        double sigma = getParam("sigma", 2.0);  // Increased from 1.0 for better parallelization
        result = applyOptimizedGaussianBlur(img, sigma);
        break;
      }
      case FilterType::UNSHARP_MASK: {
        double sigma = getParam("sigma", 2.0);  // Increased from 1.0 for better parallelization
        double strength = getParam("strength", 1.5);
        result = applyOptimizedUnsharpMask(img, sigma, strength);
        break;
      }
      case FilterType::BILATERAL_DENOISE: {
        int d = static_cast<int>(getParam("d", 9));
        double sigmaColor = getParam("sigmaColor", 75);
        double sigmaSpace = getParam("sigmaSpace", 75);
        result = applyOptimizedBilateralFilter(img, d, sigmaColor, sigmaSpace);
        break;
      }
      case FilterType::CLAHE_ENHANCE: {
        double clipLimit = getParam("clipLimit", 2.0);
        int tileSize = static_cast<int>(getParam("tileSize", 8));
        result = applyOptimizedCLAHE(img, clipLimit, cv::Size(tileSize, tileSize));
        break;
      }
      case FilterType::EDGE_ENHANCE: {
        double strength = getParam("strength", 1.0);
        result = applyOptimizedEdgeEnhance(img, strength);
        break;
      }
      default:
        result = applyOptimizedGaussianBlur(img);
    }

    return result;
  }

  /**
   * Phase 4: Execute complete filter pipeline with performance tracking
   * This is the core of the multi-filter preprocessing system
   */
  cv::Mat processPipeline(const cv::Mat& img, const FilterPipeline& pipeline) {
    pipelinePerformance.clear();
    cv::Mat result = img.clone();

    if (verbose) {
      std::cout << "\n=== Pipeline Execution ===\n";
      std::cout << "Pipeline: " << pipeline.getName() << " (" << pipeline.size()
                << " stage" << (pipeline.size() != 1 ? "s" : "") << ")\n";
    }

    int stageNum = 1;
    for (const auto& op : pipeline.getOperations()) {
      if (stageNum > config.maxPipelineStages) {
        if (verbose) {
          std::cout << "Warning: Reached maximum pipeline stages ("
                    << config.maxPipelineStages << "), stopping early\n";
        }
        break;
      }

      auto stageStart = std::chrono::high_resolution_clock::now();

      // Reset performance counters for this stage
      perfCounters = PerformanceCounters();

      // Apply filter with parameters
      result = applySingleFilter(result, op.type, op.parameters);

      auto stageEnd = std::chrono::high_resolution_clock::now();
      double stageTime = std::chrono::duration<double, std::milli>(
                            stageEnd - stageStart).count();

      // Record stage performance
      pipelinePerformance.push_back({op.type, stageTime, op.reason,
                                    perfCounters.threadsUsed});

      // Save intermediate result if debugging
      if (config.saveIntermediateResults) {
        std::string intermediatePath = config.intermediateDir +
                                      "stage_" + std::to_string(stageNum) + "_" +
                                      filterTypeToString(op.type) + ".jpg";
        cv::imwrite(intermediatePath, result);
      }

      if (verbose) {
        std::cout << "[Stage " << stageNum << "] "
                  << filterTypeToString(op.type) << ": " << std::fixed
                  << std::setprecision(2) << stageTime << "ms";
        if (perfCounters.threadsUsed > 1) {
          std::cout << " (" << perfCounters.threadsUsed << " threads)";
        }
        std::cout << " - " << op.reason << "\n";
      }

      stageNum++;
    }

    return result;
  }

  /**
   * Convenience method: Auto-process with intelligent pipeline selection
   */
  cv::Mat processImageAuto(const cv::Mat& img) {
    auto metrics = analyzeImage(img);
    auto pipeline = selectOptimalPipeline(metrics);
    return processPipeline(img, pipeline);
  }

  /**
   * Apply the optimized filter based on type (backwards compatibility)
   */
  cv::Mat processImage(const cv::Mat& img,
                       FilterType filter = FilterType::GAUSSIAN_BLUR) {
    // Reset performance counters
    perfCounters = PerformanceCounters();
    pipelinePerformance.clear();

    // Use the new pipeline-based approach internally
    auto pipeline = FilterPipeline::createStandardPipeline(filter);
    return processPipeline(img, pipeline);
  }

  /**
   * Phase 5: Enhanced pipeline performance reporting
   */
  void printPipelinePerformance() const {
    if (!verbose) return;

    if (pipelinePerformance.empty()) {
      return;
    }

    std::cout << "\n=== Pipeline Performance Summary ===\n";

    double totalTime = 0.0;
    int maxThreads = 1;

    for (const auto& stage : pipelinePerformance) {
      totalTime += stage.processingTime;
      maxThreads = std::max(maxThreads, stage.threadsUsed);
    }

    std::cout << "Total pipeline time: " << std::fixed << std::setprecision(2)
              << totalTime << " ms (" << pipelinePerformance.size()
              << " stage" << (pipelinePerformance.size() != 1 ? "s" : "")
              << ")\n";
    std::cout << "Max threads used: " << maxThreads << "\n";

    std::cout << "\nPer-stage breakdown:\n";
    for (size_t i = 0; i < pipelinePerformance.size(); i++) {
      const auto& stage = pipelinePerformance[i];
      double percentage = (stage.processingTime / totalTime) * 100.0;

      std::cout << "  Stage " << (i + 1) << " ("
                << filterTypeToString(stage.filter) << "): " << std::fixed
                << std::setprecision(2) << stage.processingTime << " ms ("
                << std::setprecision(1) << percentage << "%)";

      if (stage.threadsUsed > 1) {
        std::cout << " [" << stage.threadsUsed << " threads]";
      }

      std::cout << "\n";
    }

    std::cout << "============================\n";
  }

  /**
   * Get detailed performance analysis (legacy single-filter)
   */
  void printPerformanceAnalysis() const {
    // If pipeline performance is available, use the enhanced reporting
    if (!pipelinePerformance.empty()) {
      printPipelinePerformance();
      return;
    }

    // Legacy single-filter reporting
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

  const std::vector<StagePerformance>& getPipelinePerformance() const {
    return pipelinePerformance;
  }
};

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input_img> <output_img> [filter_type] [options]\n";
    std::cerr << "\nFilter types:\n";
    std::cerr << "  blur, sharpen, denoise, clahe, edge\n";
    std::cerr << "  auto - Intelligent multi-filter pipeline (default)\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --single - Force single-filter mode (disable multi-filter)\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  " << argv[0] << " input.jpg output.jpg auto\n";
    std::cerr << "  " << argv[0] << " input.jpg output.jpg clahe --single\n";
    return 1;
  }

  std::string in = argv[1], out = argv[2];
  std::string filterStr = (argc > 3) ? argv[3] : "auto";
  bool forceSingleFilter = false;

  // Check for --single flag
  for (int i = 4; i < argc; i++) {
    if (std::string(argv[i]) == "--single") {
      forceSingleFilter = true;
    }
  }

  bool autoAssess = (filterStr == "auto");

  // Load image
  cv::Mat img = cv::imread(in, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Failed to open " << in << "\n";
    return 1;
  }

  std::cout << "=== Optimized OpenMP Image Preprocessing Pipeline ===\n";
  std::cout << "Input: " << in << " (" << img.cols << "x" << img.rows << ")\n";

  auto start = std::chrono::high_resolution_clock::now();

  // Configure processor
  PreprocessingConfig config;
  config.enableMultiFilter = !forceSingleFilter;
  config.strategy = PreprocessingConfig::Strategy::BALANCED;

  OptimizedImagePreprocessor processor(true, config);
  cv::Mat processed;

  if (autoAssess) {
    // Use intelligent auto-processing with multi-filter support
    processed = processor.processImageAuto(img);
  } else {
    // Manual filter selection
    FilterType selectedFilter;
    if (filterStr == "blur")
      selectedFilter = FilterType::GAUSSIAN_BLUR;
    else if (filterStr == "sharpen")
      selectedFilter = FilterType::UNSHARP_MASK;
    else if (filterStr == "denoise")
      selectedFilter = FilterType::BILATERAL_DENOISE;
    else if (filterStr == "clahe")
      selectedFilter = FilterType::CLAHE_ENHANCE;
    else if (filterStr == "edge")
      selectedFilter = FilterType::EDGE_ENHANCE;
    else
      selectedFilter = FilterType::GAUSSIAN_BLUR;

    processed = processor.processImage(img, selectedFilter);
  }

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

  std::cout << "\n=== Processing Complete ===\n";
  std::cout << "Output: " << out << "\n";
  std::cout << "Total processing time: " << duration.count() << " ms\n";
  std::cout << "Max threads available: " << numThreads << "\n";

  // Show detailed performance analysis
  processor.printPerformanceAnalysis();

  std::cout << "\nReady for YOLO detection!\n";

  return 0;
}
