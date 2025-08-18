#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Gaussian Blur Filter Class
 * 
 * Provides implementation for Gaussian blur convolution filter
 * with both standard and separable convolution methods.
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
    std::vector<std::vector<double>> generateGaussianKernel(int size, double sigma);
    
public:
    /**
     * Constructor
     * @param size: Size of the Gaussian kernel (must be odd, e.g., 3, 5, 7, 9)
     * @param sigma: Standard deviation for Gaussian distribution
     */
    GaussianBlurFilter(int size = 5, double sigma = 1.0);
    
    /**
     * Apply Gaussian blur using manual convolution
     * @param input: Input image (grayscale or color)
     * @return: Blurred output image
     */
    cv::Mat applyConvolution(const cv::Mat& input);
    
    /**
     * Apply separable Gaussian blur (more efficient)
     * Uses the separability property of Gaussian kernels
     * @param input: Input image (grayscale or color)
     * @return: Blurred output image
     */
    cv::Mat applySeparableConvolution(const cv::Mat& input);
    
    /**
     * Print the Gaussian kernel for debugging
     */
    void printKernel();
    
    /**
     * Get kernel size
     * @return: Current kernel size
     */
    int getKernelSize() const { return kernelSize; }
    
    /**
     * Get sigma value
     * @return: Current sigma value
     */
    double getSigma() const { return sigma; }
};

#endif // GAUSSIAN_BLUR_H
