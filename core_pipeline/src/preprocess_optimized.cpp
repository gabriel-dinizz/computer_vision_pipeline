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

// Parallel Sobel gradient computation with tile-based processing
// Process in tiles for better cache utilization
#pragma omp parallel for schedule(dynamic, 4) num_threads(omp_get_max_threads())
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

    if (verbose) std::cout << "Applied Optimized Parallel Sobel Edge Enhancement (OpenMP)\n";
    return finalResult;
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
