#include <omp.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Simple convolution filter (sequential)
Mat apply_filter_sequential(const Mat& src,
                            const vector<vector<float>>& kernel) {
  Mat dst = Mat::zeros(src.size(), src.type());
  int ksize = kernel.size();
  int radius = ksize / 2;

  for (int y = radius; y < src.rows - radius; y++) {
    for (int x = radius; x < src.cols - radius; x++) {
      Vec3f sum(0, 0, 0);
      for (int ky = 0; ky < ksize; ky++) {
        for (int kx = 0; kx < ksize; kx++) {
          Vec3b pixel = src.at<Vec3b>(y + ky - radius, x + kx - radius);
          float weight = kernel[ky][kx];
          sum[0] += pixel[0] * weight;
          sum[1] += pixel[1] * weight;
          sum[2] += pixel[2] * weight;
        }
      }
      dst.at<Vec3b>(y, x) =
          Vec3b(saturate_cast<uchar>(sum[0]), saturate_cast<uchar>(sum[1]),
                saturate_cast<uchar>(sum[2]));
    }
  }
  return dst;
}

// Simple convolution filter (parallel)
Mat apply_filter_parallel(const Mat& src, const vector<vector<float>>& kernel) {
  Mat dst = Mat::zeros(src.size(), src.type());
  int ksize = kernel.size();
  int radius = ksize / 2;

#pragma omp parallel for
  for (int y = radius; y < src.rows - radius; y++) {
    for (int x = radius; x < src.cols - radius; x++) {
      Vec3f sum(0, 0, 0);
      for (int ky = 0; ky < ksize; ky++) {
        for (int kx = 0; kx < ksize; kx++) {
          Vec3b pixel = src.at<Vec3b>(y + ky - radius, x + kx - radius);
          float weight = kernel[ky][kx];
          sum[0] += pixel[0] * weight;
          sum[1] += pixel[1] * weight;
          sum[2] += pixel[2] * weight;
        }
      }
      dst.at<Vec3b>(y, x) =
          Vec3b(saturate_cast<uchar>(sum[0]), saturate_cast<uchar>(sum[1]),
                saturate_cast<uchar>(sum[2]));
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

  cout << "=== Parallel vs Sequential Filter Benchmark ===" << endl;
  cout << "Image size: " << image.cols << "x" << image.rows << endl;
  cout << "OpenMP threads: " << omp_get_max_threads() << endl << endl;

  // Test kernels
  vector<vector<float>> sharpen_kernel = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};

  vector<vector<float>> blur_kernel = {{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
                                       {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
                                       {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};

  // Benchmark sharpen filter
  cout << "Sharpen Filter:" << endl;
  auto start = chrono::high_resolution_clock::now();
  Mat sharp_seq = apply_filter_sequential(image, sharpen_kernel);
  auto end = chrono::high_resolution_clock::now();
  auto seq_time = chrono::duration_cast<chrono::milliseconds>(end - start);

  start = chrono::high_resolution_clock::now();
  Mat sharp_par = apply_filter_parallel(image, sharpen_kernel);
  end = chrono::high_resolution_clock::now();
  auto par_time = chrono::duration_cast<chrono::milliseconds>(end - start);

  cout << "  Sequential: " << seq_time.count() << "ms" << endl;
  cout << "  Parallel: " << par_time.count() << "ms" << endl;
  cout << "  Speedup: " << (double)seq_time.count() / par_time.count() << "x"
       << endl
       << endl;

  // Benchmark blur filter
  cout << "Blur Filter:" << endl;
  start = chrono::high_resolution_clock::now();
  Mat blur_seq = apply_filter_sequential(image, blur_kernel);
  end = chrono::high_resolution_clock::now();
  seq_time = chrono::duration_cast<chrono::milliseconds>(end - start);

  start = chrono::high_resolution_clock::now();
  Mat blur_par = apply_filter_parallel(image, blur_kernel);
  end = chrono::high_resolution_clock::now();
  par_time = chrono::duration_cast<chrono::milliseconds>(end - start);

  cout << "  Sequential: " << seq_time.count() << "ms" << endl;
  cout << "  Parallel: " << par_time.count() << "ms" << endl;
  cout << "  Speedup: " << (double)seq_time.count() / par_time.count() << "x"
       << endl;

  return 0;
}
