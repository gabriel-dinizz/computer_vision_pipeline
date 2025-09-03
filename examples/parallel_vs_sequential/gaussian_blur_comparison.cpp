#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

// Sequential Gaussian Blur
Mat gaussian_blur_sequential(const Mat& src, int kernel_size, double sigma) {
    Mat dst = src.clone();
    int radius = kernel_size / 2;
    
    // Create Gaussian kernel
    vector<double> kernel(kernel_size);
    double sum = 0.0;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - radius;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }
    
    // Apply horizontal pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3f pixel(0, 0, 0);
            for (int k = 0; k < kernel_size; k++) {
                int nx = x + k - radius;
                nx = max(0, min(src.cols - 1, nx));
                Vec3b src_pixel = src.at<Vec3b>(y, nx);
                pixel[0] += src_pixel[0] * kernel[k];
                pixel[1] += src_pixel[1] * kernel[k];
                pixel[2] += src_pixel[2] * kernel[k];
            }
            dst.at<Vec3b>(y, x) = Vec3b(pixel[0], pixel[1], pixel[2]);
        }
    }
    
    return dst;
}

// Parallel Gaussian Blur
Mat gaussian_blur_parallel(const Mat& src, int kernel_size, double sigma) {
    Mat dst = src.clone();
    int radius = kernel_size / 2;
    
    // Create Gaussian kernel
    vector<double> kernel(kernel_size);
    double sum = 0.0;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - radius;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }
    
    // Apply horizontal pass with OpenMP
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3f pixel(0, 0, 0);
            for (int k = 0; k < kernel_size; k++) {
                int nx = x + k - radius;
                nx = max(0, min(src.cols - 1, nx));
                Vec3b src_pixel = src.at<Vec3b>(y, nx);
                pixel[0] += src_pixel[0] * kernel[k];
                pixel[1] += src_pixel[1] * kernel[k];
                pixel[2] += src_pixel[2] * kernel[k];
            }
            dst.at<Vec3b>(y, x) = Vec3b(pixel[0], pixel[1], pixel[2]);
        }
    }
    
    return dst;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }
    
    Mat image = imread(argv[1]);
    if (image.empty()) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }
    
    cout << "Image size: " << image.cols << "x" << image.rows << endl;
    cout << "OpenMP threads: " << omp_get_max_threads() << endl;
    
    // Sequential version
    auto start = chrono::high_resolution_clock::now();
    Mat result_seq = gaussian_blur_sequential(image, 15, 2.0);
    auto end = chrono::high_resolution_clock::now();
    auto seq_time = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    // Parallel version
    start = chrono::high_resolution_clock::now();
    Mat result_par = gaussian_blur_parallel(image, 15, 2.0);
    end = chrono::high_resolution_clock::now();
    auto par_time = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "Sequential time: " << seq_time.count() << "ms" << endl;
    cout << "Parallel time: " << par_time.count() << "ms" << endl;
    cout << "Speedup: " << (double)seq_time.count() / par_time.count() << "x" << endl;
    
    imwrite("gaussian_sequential.jpg", result_seq);
    imwrite("gaussian_parallel.jpg", result_par);
    
    return 0;
}
