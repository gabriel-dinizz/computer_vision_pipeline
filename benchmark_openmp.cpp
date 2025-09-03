#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <vector>

class OpenMPBenchmark {
public:
    struct Result {
        int threads;
        double time_ms;
        double speedup;
        double efficiency;
    };
    
    static std::vector<Result> benchmark_gaussian_blur(const cv::Mat& img, int max_threads = 8) {
        std::vector<Result> results;
        double baseline_time = 0;
        
        for (int t = 1; t <= max_threads; t *= 2) {
            omp_set_num_threads(t);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            cv::Mat result = img.clone();
            #pragma omp parallel for
            for (int r = 0; r < img.rows; ++r) {
                cv::GaussianBlur(img.row(r), result.row(r), cv::Size(5,5), 1.0);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            if (t == 1) baseline_time = time_ms;
            
            results.push_back({
                t, 
                time_ms, 
                baseline_time / time_ms,
                (baseline_time / time_ms) / t
            });
        }
        return results;
    }
};
