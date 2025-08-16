#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <input_img> <output_img>\n";
        return 1;
    }
    std::string in = argv[1], out = argv[2];

    cv::Mat img = cv::imread(in, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Falha ao abrir " << in << "\n";
        return 1;
    }

    cv::Mat blurred = img.clone();
    // Exemplo simples: aplica blur em paralelo por linha
    #pragma omp parallel for
    for (int r = 0; r < img.rows; ++r) {
        cv::GaussianBlur(img.row(r), blurred.row(r), cv::Size(5,5), 1.0, 1.0);
    }

    if (!cv::imwrite(out, blurred)) {
        std::cerr << "Falha ao salvar " << out << "\n";
        return 1;
    }

    int nt = 1;
    #ifdef _OPENMP
    nt = omp_get_max_threads();
    #endif
    std::cout << "OK! Usando até " << nt << " threads.\n";
    return 0;
}
