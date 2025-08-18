#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>

/**
 * Gaussian Blur Convolution Algorithm
 * 
 * This implementation provides a custom Gaussian blur filter using convolution.
 * Gaussian blur is commonly used for noise reduction and image smoothing.
 */

class GaussianBlurFilter {
private:
    std::vector<std::vector<double>> kernel;
    int kernelSize;
    double sigma;
    
    /**
     * Generate a Gaussian kernel for convolution
     * @param size: Size of the kernel (should be odd)
     * @param sigma: Standard deviation for Gaussian distribution
     * @return: 2D vector representing the Gaussian kernel
     */
    std::vector<std::vector<double>> generateGaussianKernel(int size, double sigma) {
        std::vector<std::vector<double>> kernel(size, std::vector<double>(size));
        double sum = 0.0;
        int center = size / 2;
        
        // Generate Gaussian kernel values
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double x = i - center;
                double y = j - center;
                double value = (1.0 / (2.0 * M_PI * sigma * sigma)) * 
                              exp(-(x * x + y * y) / (2.0 * sigma * sigma));
                kernel[i][j] = value;
                sum += value;
            }
        }
        
        // Normalize the kernel so that sum equals 1
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                kernel[i][j] /= sum;
            }
        }
        
        return kernel;
    }
    
public:
    /**
     * Constructor
     * @param size: Size of the Gaussian kernel (must be odd, e.g., 3, 5, 7, 9)
     * @param sigma: Standard deviation for Gaussian distribution
     */
    GaussianBlurFilter(int size = 5, double sigma = 1.0) 
        : kernelSize(size), sigma(sigma) {
        if (size % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd");
        }
        kernel = generateGaussianKernel(size, sigma);
    }
    
    /**
     * Apply Gaussian blur using manual convolution
     * @param input: Input image (grayscale)
     * @return: Blurred output image
     */
    cv::Mat applyConvolution(const cv::Mat& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input image is empty");
        }
        
        // Convert to grayscale if necessary
        cv::Mat grayInput;
        if (input.channels() == 3) {
            cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
        } else {
            grayInput = input.clone();
        }
        
        // Convert to float for better precision during convolution
        cv::Mat floatInput;
        grayInput.convertTo(floatInput, CV_32F);
        
        cv::Mat output = cv::Mat::zeros(floatInput.size(), CV_32F);
        int padding = kernelSize / 2;
        
        // Apply convolution with zero padding
        for (int i = padding; i < floatInput.rows - padding; i++) {
            for (int j = padding; j < floatInput.cols - padding; j++) {
                double sum = 0.0;
                
                // Convolve with kernel
                for (int ki = 0; ki < kernelSize; ki++) {
                    for (int kj = 0; kj < kernelSize; kj++) {
                        int row = i - padding + ki;
                        int col = j - padding + kj;
                        sum += floatInput.at<float>(row, col) * kernel[ki][kj];
                    }
                }
                
                output.at<float>(i, j) = static_cast<float>(sum);
            }
        }
        
        // Convert back to 8-bit
        cv::Mat result;
        output.convertTo(result, CV_8U);
        return result;
    }
    
    /**
     * Apply separable Gaussian blur (more efficient)
     * Uses the separability property of Gaussian kernels
     */
    cv::Mat applySeparableConvolution(const cv::Mat& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input image is empty");
        }
        
        // Convert to grayscale if necessary
        cv::Mat grayInput;
        if (input.channels() == 3) {
            cv::cvtColor(input, grayInput, cv::COLOR_BGR2GRAY);
        } else {
            grayInput = input.clone();
        }
        
        // Generate 1D Gaussian kernel
        std::vector<double> kernel1D(kernelSize);
        double sum = 0.0;
        int center = kernelSize / 2;
        
        for (int i = 0; i < kernelSize; i++) {
            double x = i - center;
            double value = exp(-(x * x) / (2.0 * sigma * sigma));
            kernel1D[i] = value;
            sum += value;
        }
        
        // Normalize
        for (int i = 0; i < kernelSize; i++) {
            kernel1D[i] /= sum;
        }
        
        cv::Mat floatInput;
        grayInput.convertTo(floatInput, CV_32F);
        
        // First pass: horizontal convolution
        cv::Mat temp = cv::Mat::zeros(floatInput.size(), CV_32F);
        int padding = kernelSize / 2;
        
        for (int i = 0; i < floatInput.rows; i++) {
            for (int j = padding; j < floatInput.cols - padding; j++) {
                double sum = 0.0;
                for (int k = 0; k < kernelSize; k++) {
                    int col = j - padding + k;
                    sum += floatInput.at<float>(i, col) * kernel1D[k];
                }
                temp.at<float>(i, j) = static_cast<float>(sum);
            }
        }
        
        // Second pass: vertical convolution
        cv::Mat output = cv::Mat::zeros(temp.size(), CV_32F);
        for (int i = padding; i < temp.rows - padding; i++) {
            for (int j = 0; j < temp.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < kernelSize; k++) {
                    int row = i - padding + k;
                    sum += temp.at<float>(row, j) * kernel1D[k];
                }
                output.at<float>(i, j) = static_cast<float>(sum);
            }
        }
        
        // Convert back to 8-bit
        cv::Mat result;
        output.convertTo(result, CV_8U);
        return result;
    }
    
    /**
     * Print the Gaussian kernel for debugging
     */
    void printKernel() {
        std::cout << "Gaussian Kernel (" << kernelSize << "x" << kernelSize 
                  << ", σ=" << sigma << "):" << std::endl;
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                std::cout << std::fixed << std::setprecision(6) << kernel[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

/**
 * Example usage and test function
 */
int main() {
    try {
        // Create Gaussian blur filter
        GaussianBlurFilter gaussianFilter(5, 1.0);
        
        // Print the kernel
        gaussianFilter.printKernel();
        
        // Test with a sample image (replace with actual image path)
        std::string imagePath = "../images/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg";
        cv::Mat image = cv::imread(imagePath);
        
        if (image.empty()) {
            std::cout << "Could not load image. Testing with synthetic data..." << std::endl;
            
            // Create a test image with noise
            cv::Mat testImage = cv::Mat::zeros(200, 200, CV_8UC1);
            cv::randu(testImage, 0, 255);
            
            // Apply Gaussian blur
            cv::Mat blurred = gaussianFilter.applyConvolution(testImage);
            cv::Mat separableBlurred = gaussianFilter.applySeparableConvolution(testImage);
            
            // Save results
            cv::imwrite("original_noisy.jpg", testImage);
            cv::imwrite("gaussian_blurred.jpg", blurred);
            cv::imwrite("separable_blurred.jpg", separableBlurred);
            
            std::cout << "Test completed. Check output images." << std::endl;
        } else {
            // Apply blur to actual image
            cv::Mat blurred = gaussianFilter.applyConvolution(image);
            cv::Mat separableBlurred = gaussianFilter.applySeparableConvolution(image);
            
            // Save results
            cv::imwrite("original.jpg", image);
            cv::imwrite("gaussian_blurred.jpg", blurred);
            cv::imwrite("separable_blurred.jpg", separableBlurred);
            
            std::cout << "Gaussian blur applied successfully!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
