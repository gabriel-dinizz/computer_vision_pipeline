#include <omp.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Sequential Unsharp Masking
Mat unsharp_mask_sequential(const Mat& src, double amount, double sigma) {
  Mat blurred, mask, result;

  // Create Gaussian blur
  GaussianBlur(src, blurred, Size(0, 0), sigma);

  // Create mask
  mask = src - blurred;

  // Apply unsharp masking
  result = src.clone();
  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      Vec3b original = src.at<Vec3b>(y, x);
      Vec3b mask_pixel = mask.at<Vec3b>(y, x);

      for (int c = 0; c < 3; c++) {
        int enhanced = original[c] + amount * mask_pixel[c];
        result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(enhanced);
      }
    }
  }

  return result;
}

// Parallel Unsharp Masking
Mat unsharp_mask_parallel(const Mat& src, double amount, double sigma) {
  Mat blurred, mask, result;

  // Create Gaussian blur
  GaussianBlur(src, blurred, Size(0, 0), sigma);

  // Create mask
  mask = src - blurred;

  // Apply unsharp masking with OpenMP
  result = src.clone();
#pragma omp parallel for
  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      Vec3b original = src.at<Vec3b>(y, x);
      Vec3b mask_pixel = mask.at<Vec3b>(y, x);

      for (int c = 0; c < 3; c++) {
        int enhanced = original[c] + amount * mask_pixel[c];
        result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(enhanced);
      }
    }
  }

  return result;
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
  Mat result_seq = unsharp_mask_sequential(image, 1.5, 1.0);
  auto end = chrono::high_resolution_clock::now();
  auto seq_time = chrono::duration_cast<chrono::milliseconds>(end - start);

  // Parallel version
  start = chrono::high_resolution_clock::now();
  Mat result_par = unsharp_mask_parallel(image, 1.5, 1.0);
  end = chrono::high_resolution_clock::now();
  auto par_time = chrono::duration_cast<chrono::milliseconds>(end - start);

  cout << "Sequential time: " << seq_time.count() << "ms" << endl;
  cout << "Parallel time: " << par_time.count() << "ms" << endl;
  cout << "Speedup: " << (double)seq_time.count() / par_time.count() << "x"
       << endl;

  imwrite("sharpen_sequential.jpg", result_seq);
  imwrite("sharpen_parallel.jpg", result_par);

  return 0;
}
