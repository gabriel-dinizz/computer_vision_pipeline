#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>

enum class FilterType {
    GAUSSIAN_BLUR,
    UNSHARP_MASK,
    LAPLACIAN_SHARPEN,
    BILATERAL_DENOISE,
    CLAHE_ENHANCE,
    EDGE_ENHANCE
};

class ImagePreprocessor {
private:
    bool verbose;
    
public:
    ImagePreprocessor(bool verbose = true) : verbose(verbose) {}
    
    /**
     * Apply Gaussian Blur in parallel
     */
    cv::Mat applyGaussianBlur(const cv::Mat& img, cv::Size kernelSize = cv::Size(5,5), double sigma = 1.0) {
        cv::Mat result = img.clone();
        
        #pragma omp parallel for
        for (int r = 0; r < img.rows; ++r) {
            cv::GaussianBlur(img.row(r), result.row(r), kernelSize, sigma, sigma);
        }
        
        if (verbose) std::cout << "Applied Gaussian Blur (parallel)\n";
        return result;
    }
    
    /**
     * Apply Unsharp Masking for sharpening (parallelized)
     */
    cv::Mat applyUnsharpMask(const cv::Mat& img, double sigma = 1.0, double strength = 1.5) {
        cv::Mat blurred, mask, result;
        
        // Create gaussian blurred version
        cv::GaussianBlur(img, blurred, cv::Size(0, 0), sigma);
        
        // Create unsharp mask - ensure same data type
        cv::Mat imgFloat, blurredFloat;
        img.convertTo(imgFloat, CV_32F);
        blurred.convertTo(blurredFloat, CV_32F);
        mask = imgFloat - blurredFloat;
        
        // Apply sharpening in parallel
        result = img.clone();
        result.convertTo(result, CV_32F);
        
        #pragma omp parallel for collapse(2)
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
        
        if (verbose) std::cout << "Applied Unsharp Mask sharpening (parallel)\n";
        return finalResult;
    }
    
    /**
     * Apply Laplacian-based sharpening (parallelized)
     */
    cv::Mat applyLaplacianSharpen(const cv::Mat& img, double strength = 0.5) {
        cv::Mat gray, laplacian, result;
        
        // Convert to grayscale for edge detection
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        
        // Apply Laplacian filter
        cv::Laplacian(gray, laplacian, CV_16S, 3);
        cv::convertScaleAbs(laplacian, laplacian);
        
        // Convert back to 3-channel for blending
        cv::Mat laplacian3ch;
        cv::cvtColor(laplacian, laplacian3ch, cv::COLOR_GRAY2BGR);
        
        // Apply sharpening in parallel
        result = img.clone();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3b original = img.at<cv::Vec3b>(i, j);
                cv::Vec3b edge = laplacian3ch.at<cv::Vec3b>(i, j);
                
                for (int c = 0; c < 3; ++c) {
                    int sharpened = original[c] + strength * edge[c];
                    result.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(sharpened);
                }
            }
        }
        
        if (verbose) std::cout << "Applied Laplacian sharpening (parallel)\n";
        return result;
    }
    
    /**
     * Apply Bilateral Filter for noise reduction while preserving edges
     */
    cv::Mat applyBilateralDenoise(const cv::Mat& img, int d = 9, double sigmaColor = 75, double sigmaSpace = 75) {
        cv::Mat result;
        
        // Split into channels for parallel processing
        std::vector<cv::Mat> channels(3);
        std::vector<cv::Mat> filteredChannels(3);
        cv::split(img, channels);
        
        #pragma omp parallel for
        for (int i = 0; i < 3; ++i) {
            cv::bilateralFilter(channels[i], filteredChannels[i], d, sigmaColor, sigmaSpace);
        }
        
        cv::merge(filteredChannels, result);
        
        if (verbose) std::cout << "Applied Bilateral denoising (parallel)\n";
        return result;
    }
    
    /**
     * Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
     */
    cv::Mat applyCLAHE(const cv::Mat& img, double clipLimit = 2.0, cv::Size tileGridSize = cv::Size(8,8)) {
        cv::Mat result;
        std::vector<cv::Mat> bgrChannels(3);
        std::vector<cv::Mat> processedChannels(3);
        
        // Split into BGR channels
        cv::split(img, bgrChannels);
        
        // Apply CLAHE to each channel separately
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
        
        #pragma omp parallel for
        for (int i = 0; i < 3; ++i) {
            clahe->apply(bgrChannels[i], processedChannels[i]);
        }
        
        // Merge channels back
        cv::merge(processedChannels, result);
        
        if (verbose) std::cout << "Applied CLAHE enhancement\n";
        return result;
    }
    
    /**
     * Apply edge enhancement filter
     */
    cv::Mat applyEdgeEnhance(const cv::Mat& img, double strength = 1.0) {
        cv::Mat gray, edges, result;
        
        // Convert to grayscale
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        
        // Detect edges using Canny
        cv::Canny(gray, edges, 100, 200);
        
        // Convert edges to 3-channel
        cv::Mat edges3ch;
        cv::cvtColor(edges, edges3ch, cv::COLOR_GRAY2BGR);
        
        // Enhance edges in parallel
        result = img.clone();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                cv::Vec3b original = img.at<cv::Vec3b>(i, j);
                cv::Vec3b edge = edges3ch.at<cv::Vec3b>(i, j);
                
                for (int c = 0; c < 3; ++c) {
                    int enhanced = original[c] + strength * (edge[c] / 255.0) * 50;
                    result.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(enhanced);
                }
            }
        }
        
        if (verbose) std::cout << "Applied edge enhancement (parallel)\n";
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
        
        // Calculate noise level (using local standard deviation)
        cv::Mat noise, diff;
        cv::Mat kernel = cv::getGaussianKernel(5, 1.0);
        cv::filter2D(gray, noise, CV_64F, kernel);
        
        // Convert gray to same type as noise for subtraction
        cv::Mat grayDouble;
        gray.convertTo(grayDouble, CV_64F);
        diff = grayDouble - noise;
        
        cv::Scalar noiseMean, noiseStd;
        cv::meanStdDev(diff, noiseMean, noiseStd);
        double noiseLevel = noiseStd.val[0];
        
        if (verbose) {
            std::cout << "Image Quality Assessment:\n";
            std::cout << "  Blur variance: " << std::fixed << std::setprecision(1) << variance << " (>100 = sharp, <100 = blurry)\n";
            std::cout << "  Brightness: " << std::fixed << std::setprecision(1) << brightness << " (0-255)\n";
            std::cout << "  Noise level: " << std::fixed << std::setprecision(1) << noiseLevel << "\n";
        }
        
        // Decision logic
        if (variance < 100) {
            std::cout << "  Recommendation: Image appears blurry - applying sharpening\n";
            return FilterType::UNSHARP_MASK;
        } else if (noiseLevel > 15) {
            std::cout << "  Recommendation: Image appears noisy - applying denoising\n";
            return FilterType::BILATERAL_DENOISE;
        } else if (brightness < 50 || brightness > 200) {
            std::cout << "  Recommendation: Poor contrast - applying CLAHE\n";
            return FilterType::CLAHE_ENHANCE;
        } else {
            std::cout << "  Recommendation: Good quality - applying edge enhancement\n";
            return FilterType::EDGE_ENHANCE;
        }
    }
    
    /**
     * Apply the recommended filter based on image quality
     */
    cv::Mat processImage(const cv::Mat& img, FilterType filter = FilterType::GAUSSIAN_BLUR) {
        switch (filter) {
            case FilterType::GAUSSIAN_BLUR:
                return applyGaussianBlur(img);
            case FilterType::UNSHARP_MASK:
                return applyUnsharpMask(img);
            case FilterType::LAPLACIAN_SHARPEN:
                return applyLaplacianSharpen(img);
            case FilterType::BILATERAL_DENOISE:
                return applyBilateralDenoise(img);
            case FilterType::CLAHE_ENHANCE:
                return applyCLAHE(img);
            case FilterType::EDGE_ENHANCE:
                return applyEdgeEnhance(img);
            default:
                return applyGaussianBlur(img);
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_img> <output_img> [filter_type] [auto_assess]\n";
        std::cerr << "Filter types: blur, sharpen, laplacian, denoise, clahe, edge\n";
        std::cerr << "Auto assess: use 'auto' to automatically choose best filter\n";
        return 1;
    }
    
    std::string in = argv[1], out = argv[2];
    std::string filterStr = (argc > 3) ? argv[3] : "auto";
    bool autoAssess = (filterStr == "auto") || (argc > 4 && std::string(argv[4]) == "auto");
    
    // Load image
    cv::Mat img = cv::imread(in, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to open " << in << "\n";
        return 1;
    }
    
    std::cout << "=== Image Preprocessing Pipeline ===\n";
    std::cout << "Input: " << in << " (" << img.cols << "x" << img.rows << ")\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    ImagePreprocessor processor(true);
    FilterType selectedFilter;
    
    if (autoAssess) {
        selectedFilter = processor.assessImageQuality(img);
    } else {
        // Manual filter selection
        if (filterStr == "blur") selectedFilter = FilterType::GAUSSIAN_BLUR;
        else if (filterStr == "sharpen") selectedFilter = FilterType::UNSHARP_MASK;
        else if (filterStr == "laplacian") selectedFilter = FilterType::LAPLACIAN_SHARPEN;
        else if (filterStr == "denoise") selectedFilter = FilterType::BILATERAL_DENOISE;
        else if (filterStr == "clahe") selectedFilter = FilterType::CLAHE_ENHANCE;
        else if (filterStr == "edge") selectedFilter = FilterType::EDGE_ENHANCE;
        else selectedFilter = FilterType::GAUSSIAN_BLUR;
    }
    
    // Apply selected filter
    cv::Mat processed = processor.processImage(img, selectedFilter);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
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
    std::cout << "Processing time: " << duration.count() << " ms\n";
    std::cout << "Threads used: " << numThreads << "\n";
    std::cout << "Ready for YOLO detection!\n";
    
    return 0;
}
