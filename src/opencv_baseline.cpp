#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <string>

/**
 * OpenCV Native Baseline Implementations
 * These use OpenCV's optimized native functions for comparison
 */

class OpenCVBaseline {
private:
    bool verbose;
    
public:
    OpenCVBaseline(bool verbose = true) : verbose(verbose) {}
    
    /**
     * OpenCV Native Gaussian Blur
     */
    cv::Mat opencvGaussianBlur(const cv::Mat& img, cv::Size kernelSize = cv::Size(5,5), double sigma = 1.0) {
        cv::Mat result;
        cv::GaussianBlur(img, result, kernelSize, sigma, sigma);
        
        if (verbose) std::cout << "Applied OpenCV Native Gaussian Blur\n";
        return result;
    }
    
    /**
     * OpenCV Native Bilateral Filter
     */
    cv::Mat opencvBilateral(const cv::Mat& img, int d = 9, double sigmaColor = 75, double sigmaSpace = 75) {
        cv::Mat result;
        cv::bilateralFilter(img, result, d, sigmaColor, sigmaSpace);
        
        if (verbose) std::cout << "Applied OpenCV Native Bilateral Filter\n";
        return result;
    }
    
    /**
     * OpenCV Native CLAHE
     */
    cv::Mat opencvCLAHE(const cv::Mat& img, double clipLimit = 2.0, cv::Size tileGridSize = cv::Size(8,8)) {
        cv::Mat result;
        std::vector<cv::Mat> bgrChannels(3);
        std::vector<cv::Mat> processedChannels(3);
        
        cv::split(img, bgrChannels);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
        
        for (int i = 0; i < 3; ++i) {
            clahe->apply(bgrChannels[i], processedChannels[i]);
        }
        
        cv::merge(processedChannels, result);
        
        if (verbose) std::cout << "Applied OpenCV Native CLAHE\n";
        return result;
    }
    
    /**
     * OpenCV Native Sobel Edge Detection
     */
    cv::Mat opencvSobel(const cv::Mat& img) {
        cv::Mat gray, grad_x, grad_y, result;
        
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        
        cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
        
        cv::convertScaleAbs(grad_x, grad_x);
        cv::convertScaleAbs(grad_y, grad_y);
        
        cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, result);
        
        // Convert to 3-channel
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
        
        if (verbose) std::cout << "Applied OpenCV Native Sobel\n";
        return result;
    }
    
    /**
     * Process image with specified filter
     */
    cv::Mat processImage(const cv::Mat& img, const std::string& filterType) {
        if (filterType == "blur") {
            return opencvGaussianBlur(img);
        } else if (filterType == "denoise") {
            return opencvBilateral(img);
        } else if (filterType == "clahe") {
            return opencvCLAHE(img);
        } else if (filterType == "edge") {
            return opencvSobel(img);
        } else {
            return opencvGaussianBlur(img); // default
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_img> <output_img> <filter_type>\n";
        std::cerr << "Filter types: blur, denoise, clahe, edge\n";
        return 1;
    }
    
    std::string input = argv[1];
    std::string output = argv[2];
    std::string filterType = argv[3];
    
    // Load image
    cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to open " << input << "\n";
        return 1;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    OpenCVBaseline processor(true);
    cv::Mat processed = processor.processImage(img, filterType);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Save result
    if (!cv::imwrite(output, processed)) {
        std::cerr << "Failed to save " << output << "\n";
        return 1;
    }
    
    std::cout << "=== OpenCV Native Processing Complete ===\n";
    std::cout << "Filter: " << filterType << "\n";
    std::cout << "Processing time: " << duration.count() << " ms\n";
    
    return 0;
}
