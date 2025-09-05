#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Sequential baseline implementations for academic comparison
 * These provide reference implementations without parallelization
 */

class SequentialBaseline {
 private:
  bool verbose;

 public:
  SequentialBaseline(bool verbose = true) : verbose(verbose) {}

  /**
   * Sequential Gaussian Blur (baseline)
   */
  cv::Mat sequentialGaussianBlur(const cv::Mat& img,
                                 cv::Size kernelSize = cv::Size(5, 5),
                                 double sigma = 1.0) {
    cv::Mat result;
    cv::GaussianBlur(img, result, kernelSize, sigma, sigma);

    if (verbose) std::cout << "Applied Sequential Gaussian Blur\n";
    return result;
  }

  /**
   * Sequential Unsharp Masking
   */
  cv::Mat sequentialUnsharpMask(const cv::Mat& img, double sigma = 1.0,
                                double strength = 1.5) {
    cv::Mat blurred, mask, result;

    // Create gaussian blurred version
    cv::GaussianBlur(img, blurred, cv::Size(0, 0), sigma);

    // Create unsharp mask
    cv::Mat imgFloat, blurredFloat;
    img.convertTo(imgFloat, CV_32F);
    blurred.convertTo(blurredFloat, CV_32F);
    mask = imgFloat - blurredFloat;

    // Apply sharpening sequentially
    result = img.clone();
    result.convertTo(result, CV_32F);

    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
        cv::Vec3f original = imgFloat.at<cv::Vec3f>(i, j);
        cv::Vec3f maskVal = mask.at<cv::Vec3f>(i, j);

        for (int c = 0; c < 3; ++c) {
          float enhanced = original[c] + strength * maskVal[c];
          result.at<cv::Vec3f>(i, j)[c] = enhanced;
        }
      }
    }

    // Convert back to 8-bit
    cv::Mat finalResult;
    result.convertTo(finalResult, CV_8U);

    if (verbose) std::cout << "Applied Sequential Unsharp Mask\n";
    return finalResult;
  }

  /**
   * Sequential Bilateral Filter
   */
  cv::Mat sequentialBilateral(const cv::Mat& img, int d = 9,
                              double sigmaColor = 75, double sigmaSpace = 75) {
    cv::Mat result;
    cv::bilateralFilter(img, result, d, sigmaColor, sigmaSpace);

    if (verbose) std::cout << "Applied Sequential Bilateral Filter\n";
    return result;
  }

  /**
   * Sequential CLAHE
   */
  cv::Mat sequentialCLAHE(const cv::Mat& img, double clipLimit = 2.0,
                          cv::Size tileGridSize = cv::Size(8, 8)) {
    cv::Mat result;
    std::vector<cv::Mat> bgrChannels(3);
    std::vector<cv::Mat> processedChannels(3);

    cv::split(img, bgrChannels);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);

    for (int i = 0; i < 3; ++i) {
      clahe->apply(bgrChannels[i], processedChannels[i]);
    }

    cv::merge(processedChannels, result);

    if (verbose) std::cout << "Applied Sequential CLAHE\n";
    return result;
  }
};

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <input_image> <output_image> <filter_type>" << std::endl;
    std::cerr << "Filter types: blur, sharpen, denoise, clahe, edge"
              << std::endl;
    return -1;
  }

  std::string inputPath = argv[1];
  std::string outputPath = argv[2];
  std::string filterType = argv[3];

  cv::Mat image = cv::imread(inputPath);
  if (image.empty()) {
    std::cerr << "Error: Could not load image " << inputPath << std::endl;
    return -1;
  }

  SequentialBaseline processor(true);
  cv::Mat result;

  auto start = std::chrono::high_resolution_clock::now();

  if (filterType == "blur") {
    result = processor.sequentialGaussianBlur(image);
  } else if (filterType == "sharpen") {
    result = processor.sequentialUnsharpMask(image);
  } else if (filterType == "denoise") {
    result = processor.sequentialBilateral(image);
  } else if (filterType == "clahe") {
    result = processor.sequentialCLAHE(image);
  } else if (filterType == "edge") {
    // Simple edge enhancement for sequential baseline
    cv::Mat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 100, 200);
    cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);
  } else {
    std::cerr << "Unknown filter type: " << filterType << std::endl;
    return -1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  if (!cv::imwrite(outputPath, result)) {
    std::cerr << "Error: Could not save result to " << outputPath << std::endl;
    return -1;
  }

  std::cout << "Sequential processing completed in " << duration.count()
            << " ms" << std::endl;
  std::cout << "Result saved to: " << outputPath << std::endl;

  return 0;
}
