# Módulo de Preprocessamento C++ - Documentação Técnica

Este módulo contém a implementação do preprocessador de imagens otimizado com OpenMP, o componente central do pipeline de visão computacional.

## 📁 Estrutura do Módulo

```
src/
└── preprocess_optimized.cpp    # Implementação completa do preprocessador (1463 linhas)
```

**Arquivo compilado:** `../bin/preprocess_optimized`

## 🎯 Objetivo do Módulo

O preprocessador C++ foi desenvolvido para:

1. **Processar imagens** antes da detecção YOLO para melhorar a qualidade de detecção
2. **Paralelizar operações** usando OpenMP para aproveitar múltiplos cores da CPU
3. **Analisar qualidade** da imagem e selecionar filtros automaticamente
4. **Aplicar múltiplos filtros** em pipeline para imagens severamente degradadas
5. **Medir performance** detalhada de cada operação

## 🏗️ Arquitetura do Código

### Componentes Principais

```
┌─────────────────────────────────────────────────────────┐
│                  preprocess_optimized.cpp                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Enumerações e Estruturas de Dados             │    │
│  │  • FilterType (enum)                           │    │
│  │  • ImageQualityMetrics                         │    │
│  │  • FilterOperation                             │    │
│  │  • FilterPipeline                              │    │
│  │  • PreprocessingConfig                         │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Classe OptimizedImagePreprocessor             │    │
│  │                                                │    │
│  │  1. Análise de Qualidade:                     │    │
│  │     • analyzeImage()                          │    │
│  │     • selectOptimalPipeline()                 │    │
│  │                                                │    │
│  │  2. Filtros Implementados (OpenMP):           │    │
│  │     • applyOptimizedGaussianBlur()            │    │
│  │     • applyOptimizedUnsharpMask()             │    │
│  │     • applyOptimizedBilateralFilter()         │    │
│  │     • applyOptimizedCLAHE()                   │    │
│  │     • applyOptimizedEdgeEnhance()             │    │
│  │                                                │    │
│  │  3. Conversão de Cores (Paralela):            │    │
│  │     • parallelBGR2Lab()                       │    │
│  │     • parallelLab2BGR()                       │    │
│  │                                                │    │
│  │  4. Execução de Pipeline:                     │    │
│  │     • processPipeline()                       │    │
│  │     • processImageAuto()                      │    │
│  │     • processImage()                          │    │
│  │                                                │    │
│  │  5. Medição de Performance:                   │    │
│  │     • PerformanceCounters                     │    │
│  │     • printPipelinePerformance()              │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  main() - Interface CLI                        │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 📊 Estruturas de Dados

### 1. FilterType (Enum)

Define os tipos de filtros disponíveis:

```cpp
enum class FilterType {
  GAUSSIAN_BLUR,        // Blur gaussiano para redução de ruído
  UNSHARP_MASK,         // Sharpening (realce de detalhes)
  LAPLACIAN_SHARPEN,    // Sharpening usando Laplaciano (não usado)
  BILATERAL_DENOISE,    // Denoising que preserva bordas
  CLAHE_ENHANCE,        // Realce de contraste adaptativo
  EDGE_ENHANCE          // Realce de bordas
};
```

### 2. ImageQualityMetrics

Estrutura que armazena métricas de qualidade da imagem:

```cpp
struct ImageQualityMetrics {
  // Métricas básicas
  double blurVariance;        // Variância Laplaciana (>100 = nítida)
  double brightness;          // Brilho médio (0-255)
  double noiseLevel;          // Nível de ruído estimado

  // Métricas avançadas
  double contrast;            // Desvio padrão do brilho
  double dynamicRange;        // Faixa de intensidade usada (0-1)
  bool hasHistogramClipping;  // Detecção de super/subexposição

  // Dimensões
  int width, height;

  // Métodos de avaliação
  bool isSeverelyDegraded() const;
  bool isExtremelyDark() const;
  bool isExtremelyBright() const;
  bool isVeryBlurry() const;
  bool isVeryNoisy() const;
  bool hasLowContrast() const;
};
```

**Critérios de degradação:**
- **Extremamente escura:** `brightness < 30`
- **Extremamente clara:** `brightness > 220`
- **Muito borrada:** `blurVariance < 50`
- **Muito ruidosa:** `noiseLevel > 20`
- **Baixo contraste:** `contrast < 30` ou `dynamicRange < 0.3`

### 3. FilterOperation

Representa uma única operação de filtro:

```cpp
struct FilterOperation {
  FilterType type;                      // Tipo de filtro
  std::map<std::string, double> parameters;  // Parâmetros configuráveis
  std::string reason;                   // Razão da seleção
};
```

### 4. FilterPipeline

Gerencia uma sequência de filtros:

```cpp
class FilterPipeline {
  std::vector<FilterOperation> operations;
  std::string pipelineName;

  // Métodos de fábrica para cenários comuns
  static FilterPipeline createSevereDarkPipeline();
  static FilterPipeline createSevereBrightPipeline();
  static FilterPipeline createNoisyBlurPipeline();
  static FilterPipeline createLowContrastPipeline();
  static FilterPipeline createStandardPipeline(FilterType);
};
```

**Pipelines pré-definidos:**

1. **Severe Dark Recovery** (Escura extrema)
   - CLAHE (clipLimit=3.0) → Edge Enhancement (strength=1.2)

2. **Severe Bright Recovery** (Clara extrema)
   - CLAHE (clipLimit=2.5) → Edge Enhancement (strength=1.0)

3. **Noisy & Blurry Recovery** (Ruidosa e borrada)
   - Bilateral Denoise → Unsharp Mask

4. **Low Contrast Recovery** (Baixo contraste)
   - CLAHE (clipLimit=2.0)

### 5. PreprocessingConfig

Configuração do comportamento do preprocessamento:

```cpp
struct PreprocessingConfig {
  bool enableMultiFilter = true;        // Habilitar múltiplos filtros
  bool saveIntermediateResults = false; // Salvar resultados intermediários
  int maxPipelineStages = 3;            // Limite de estágios no pipeline
  std::string intermediateDir = "./debug/";

  enum class Strategy {
    CONSERVATIVE,  // Processamento mínimo
    BALANCED,      // Padrão - inteligente
    AGGRESSIVE     // Máxima melhoria de qualidade
  };
  Strategy strategy = Strategy::BALANCED;
};
```

## 🔬 Algoritmos Implementados

### 1. Gaussian Blur (Convolução Separável)

**Complexidade:** O(n) vs O(n²) da convolução 2D tradicional

```cpp
cv::Mat applyOptimizedGaussianBlur(const cv::Mat& img, double sigma = 1.0)
```

**Otimizações:**
- **Convolução separável:** Divide kernel 2D em duas passadas 1D (horizontal + vertical)
- **Paralelização OpenMP:** Cada linha processada independentemente
- **Scheduling estático:** Melhor balanceamento para operações uniformes

**Código OpenMP:**
```cpp
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
for (int y = 0; y < img.rows; y++) {
  // Processa linha y
}
```

**Parâmetros:**
- `sigma` (default: 2.0): Controla a intensidade do blur

### 2. Unsharp Mask (Sharpening)

```cpp
cv::Mat applyOptimizedUnsharpMask(const cv::Mat& img, double sigma = 1.0,
                                  double strength = 1.5)
```

**Fórmula:** `resultado = original + strength × (original - blur)`

**Processo:**
1. Cria versão borrada da imagem (Gaussian Blur)
2. Calcula máscara: diferença entre original e blur
3. Adiciona máscara amplificada à imagem original
4. Clamp valores para [0, 1]

**Paralelização:** Processamento por linhas com scheduling estático

**Parâmetros:**
- `sigma` (default: 2.0): Controla blur da máscara
- `strength` (default: 1.5): Intensidade do sharpening

### 3. Bilateral Filter (Denoising)

```cpp
cv::Mat applyOptimizedBilateralFilter(const cv::Mat& img, int d = 9,
                                      double sigmaColor = 75,
                                      double sigmaSpace = 75)
```

**Características:**
- Reduz ruído **preservando bordas**
- Combina similaridade espacial e de cor
- Pre-computa pesos espaciais (Gaussiano)
- Calcula pesos de cor dinamicamente

**Peso total:** `w = spatialWeight × colorWeight`

**Parâmetros:**
- `d` (default: 9): Diâmetro da vizinhança
- `sigmaColor` (default: 75): Sigma do filtro de cor
- `sigmaSpace` (default: 75): Sigma do filtro espacial

### 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)

```cpp
cv::Mat applyOptimizedCLAHE(const cv::Mat& img, double clipLimit = 2.0,
                            cv::Size tileGridSize = cv::Size(8, 8))
```

**Processo:**
1. Converte BGR → LAB (paralelo customizado, não OpenCV serial!)
2. Divide canal L em tiles (8×8 grid)
3. **Fase 1 (paralela):** Calcula histograma e CDF para cada tile
4. **Fase 2 (paralela):** Aplica equalização usando CDFs
5. Converte LAB → BGR (paralelo)

**Por que conversão de cor paralela?**
- `cv::cvtColor()` do OpenCV é serial (gargalo!)
- Implementação customizada usa OpenMP para paralelizar pixel a pixel

**Otimizações:**
- Sem critical sections (cada thread trabalha em tile separado)
- Histogram clipping para evitar amplificação de ruído
- Redistribuição uniforme de pixels clipped

**Parâmetros:**
- `clipLimit` (default: 2.0): Limita amplificação de contraste
- `tileGridSize` (default: 8×8): Grade de tiles

### 5. Edge Enhancement (Realce de Bordas)

```cpp
cv::Mat applyOptimizedEdgeEnhance(const cv::Mat& img, double strength = 1.0)
```

**Processo:**
1. Converte para escala de cinza
2. Aplica **Sobel operators** (paralelo) para gradientes X e Y
3. Calcula magnitude do gradiente: `√(gx² + gy²)`
4. Normaliza magnitude
5. Adiciona gradiente ponderado à imagem original

**Kernels Sobel:**
```
X: [-1  0  1]    Y: [-1 -2 -1]
   [-2  0  2]       [ 0  0  0]
   [-1  0  1]       [ 1  2  1]
```

**Parâmetros:**
- `strength` (default: 1.0): Intensidade do realce

## 🧠 Sistema de Análise Inteligente

### Análise de Qualidade da Imagem

```cpp
ImageQualityMetrics analyzeImage(const cv::Mat& img)
```

**Métricas calculadas:**

1. **Blur Variance (Variância Laplaciana)**
   ```cpp
   cv::Laplacian(gray, laplacian, CV_64F);
   blurVariance = stddev² // >100 = nítida, <50 = borrada
   ```

2. **Brightness (Brilho)**
   ```cpp
   brightness = mean(gray) // 0-255
   ```

3. **Noise Level (Nível de Ruído)**
   ```cpp
   diff = original - gaussianBlur(original)
   noiseLevel = stddev(diff)
   ```

4. **Contrast (Contraste)**
   ```cpp
   contrast = stddev(gray)
   ```

5. **Dynamic Range (Faixa Dinâmica)**
   ```cpp
   dynamicRange = (max - min) / 255
   ```

6. **Histogram Clipping**
   ```cpp
   clippedPixels = count(pixels == 0) + count(pixels == 255)
   hasClipping = (clippedPixels > 1% total)
   ```

### Seleção Automática de Pipeline

```cpp
FilterPipeline selectOptimalPipeline(const ImageQualityMetrics& metrics)
```

**Lógica de decisão (em ordem de prioridade):**

```
SE (enableMultiFilter E isSeverelyDegraded):
    SE (isExtremelyDark):
        RETORNA SevereDarkPipeline
            → CLAHE (agressivo) + Edge Enhancement

    SENÃO SE (isExtremelyBright):
        RETORNA SevereBrightPipeline
            → CLAHE (moderado) + Edge Enhancement

    SENÃO SE (isVeryBlurry E isVeryNoisy):
        RETORNA NoisyBlurPipeline
            → Bilateral Denoise + Unsharp Mask

    SENÃO SE (hasLowContrast):
        RETORNA LowContrastPipeline
            → CLAHE

SENÃO (single filter mode):
    SE (blurVariance < 100):
        RETORNA Unsharp Mask

    SENÃO SE (noiseLevel > 15):
        RETORNA Bilateral Denoise

    SENÃO SE (brightness < 50 OU brightness > 200):
        RETORNA CLAHE

    SENÃO:
        RETORNA Edge Enhancement
```

## ⚡ Otimizações OpenMP

### Constantes de Cache

```cpp
static constexpr int CACHE_LINE_SIZE = 64;
static constexpr int OPTIMAL_TILE_SIZE = 64;  // 64×64 cabe em L1 cache
```

### Estratégias de Scheduling

**1. Static Scheduling (Uniforme)**
```cpp
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
```
- **Usado em:** Gaussian Blur, Unsharp Mask, Bilateral, CLAHE, Edge Enhancement
- **Vantagem:** Mínimo overhead, ideal para trabalho uniforme por linha
- **Trade-off:** Não adapta a variações de carga

**2. Dynamic Scheduling (Adaptativo)**
```cpp
#pragma omp parallel for schedule(dynamic, 4) num_threads(omp_get_max_threads())
```
- **Usado em:** Versões antigas (removido para reduzir overhead)
- **Vantagem:** Adapta a variações de carga
- **Desvantagem:** Maior overhead de sincronização

### Padrões de Acesso à Memória

**Row-based Processing (Orientado a Linhas):**
```cpp
#pragma omp parallel for schedule(static)
for (int y = 0; y < img.rows; y++) {
    const cv::Vec3f* srcRow = img.ptr<cv::Vec3f>(y);
    cv::Vec3f* dstRow = result.ptr<cv::Vec3f>(y);

    for (int x = 0; x < img.cols; x++) {
        // Acesso sequencial na memória (cache-friendly)
        dstRow[x] = process(srcRow[x]);
    }
}
```

**Vantagens:**
- Acesso sequencial à memória (ótimo para cache)
- Sem compartilhamento de dados entre threads
- Alinhamento natural com linhas de cache

## 📈 Sistema de Medição de Performance

### PerformanceCounters

```cpp
struct PerformanceCounters {
  double kernelGenTime = 0.0;      // Tempo de geração de kernel
  double convolutionTime = 0.0;    // Tempo de convolução
  double memoryAllocTime = 0.0;    // Tempo de alocação
  double totalTime = 0.0;          // Tempo total
  int threadsUsed = 1;             // Threads utilizadas
  size_t cacheHits = 0;            // Cache hits
  size_t cacheMisses = 0;          // Cache misses
};
```

### StagePerformance (Pipeline)

```cpp
struct StagePerformance {
  FilterType filter;          // Filtro aplicado
  double processingTime;      // Tempo em ms
  std::string reason;         // Razão da seleção
  int threadsUsed;           // Threads usadas
};
```

### Relatório de Performance

```
=== Pipeline Performance Summary ===
Total pipeline time: 845.23 ms (2 stages)
Max threads used: 8

Per-stage breakdown:
  Stage 1 (CLAHE): 601.14 ms (71.1%) [8 threads]
  Stage 2 (Edge Enhancement): 244.09 ms (28.9%) [8 threads]
============================
```

## 🖥️ Interface CLI

### Uso Básico

```bash
./preprocess_optimized <input_img> <output_img> [filter_type] [options]
```

### Filtros Disponíveis

```bash
blur       # Gaussian Blur
sharpen    # Unsharp Mask
denoise    # Bilateral Filter
clahe      # CLAHE Enhancement
edge       # Edge Enhancement
auto       # Seleção automática inteligente (padrão)
```

### Opções

```bash
--single   # Força modo single-filter (desabilita multi-filter)
```

### Exemplos

```bash
# Seleção automática com multi-filter
./preprocess_optimized input.jpg output.jpg auto

# CLAHE em modo single-filter
./preprocess_optimized input.jpg output.jpg clahe --single

# Sharpening
./preprocess_optimized input.jpg output.jpg sharpen

# Denoising
./preprocess_optimized input.jpg output.jpg denoise
```

### Controle de Threads

```bash
# Usar 4 threads
OMP_NUM_THREADS=4 ./preprocess_optimized input.jpg output.jpg auto

# Usar 1 thread (baseline)
OMP_NUM_THREADS=1 ./preprocess_optimized input.jpg output.jpg sharpen

# Usar todas as threads disponíveis (padrão)
./preprocess_optimized input.jpg output.jpg auto
```

## 📊 Exemplo de Saída

```
=== Optimized OpenMP Image Preprocessing Pipeline ===
Input: sample.jpg (960x505)

=== Image Quality Analysis ===
Brightness: 128.5
Blur Variance: 245.3
Noise Level: 8.2
Contrast: 45.6
Dynamic Range: 0.82
Assessment: STANDARD

=== Filter Selection ===
Selected: Edge Enhancement - Good quality, light processing

=== Pipeline Execution ===
Pipeline: Single Filter (1 stage)
Applied Optimized Parallel Sobel Edge Enhancement (OpenMP)
[Stage 1] Edge Enhancement: 45.23ms (8 threads) - User selected

=== Processing Complete ===
Output: output.jpg
Total processing time: 46 ms
Max threads available: 8

=== Pipeline Performance Summary ===
Total pipeline time: 45.23 ms (1 stage)
Max threads used: 8

Per-stage breakdown:
  Stage 1 (Edge Enhancement): 45.23 ms (100.0%) [8 threads]
============================

Ready for YOLO detection!
```

## 🔧 Compilação

### Dependências

```bash
# macOS
brew install opencv pkg-config libomp

# Linux
sudo apt-get install libopencv-dev pkg-config
```

### Flags de Compilação

```makefile
CXX = clang++  # ou g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -march=native
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`
```

### Compilar

```bash
# Via Makefile (do diretório core_pipeline)
make all

# Diretamente
clang++ -std=c++17 -O3 -fopenmp -march=native \
  `pkg-config --cflags --libs opencv4` \
  src/preprocess_optimized.cpp -o bin/preprocess_optimized
```

### Flags de Otimização

- **`-O3`**: Otimização máxima do compilador
- **`-fopenmp`**: Habilita suporte OpenMP
- **`-march=native`**: Usa instruções específicas da CPU (SIMD, AVX, etc.)
- **`-std=c++17`**: C++17 para features modernas

## 🧪 Testes e Validação

### Testar Filtros Individuais

```bash
# Testar cada filtro
for filter in blur sharpen denoise clahe edge; do
  echo "Testing $filter..."
  ./bin/preprocess_optimized images/sample.jpg temp/test_$filter.jpg $filter
done
```

### Testar Diferentes Thread Counts

```bash
# Benchmark com diferentes números de threads
for threads in 1 2 4 8; do
  echo "Testing with $threads threads..."
  OMP_NUM_THREADS=$threads ./bin/preprocess_optimized \
    images/sample.jpg temp/test_t${threads}.jpg sharpen
done
```

### Modo Debug (Resultados Intermediários)

Para habilitar salvamento de resultados intermediários, modifique o código:

```cpp
PreprocessingConfig config;
config.saveIntermediateResults = true;
config.intermediateDir = "./debug/";
```

Isso salvará cada estágio do pipeline em `debug/stage_1_*.jpg`, `debug/stage_2_*.jpg`, etc.

## 📐 Arquitetura de Memória

### Layout de Dados OpenCV

```
cv::Mat (CV_8UC3) - 3 canais BGR, 8-bit unsigned
┌────────────────────────────────────┐
│ Row 0: [B₀G₀R₀][B₁G₁R₁]...[BₙGₙRₙ]│  ← Linha contínua na memória
│ Row 1: [B₀G₀R₀][B₁G₁R₁]...[BₙGₙRₙ]│
│ ...                                 │
└────────────────────────────────────┘
```

### Alinhamento de Cache

- **Cache Line:** 64 bytes
- **Pixels por cache line:** 64 / 3 ≈ 21 pixels (BGR)
- **Tile size:** 64×64 pixels (cabe em L1 cache)

### Padrão de Acesso

```cpp
// Cache-friendly (acesso sequencial)
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        pixel = row[x];  // ✓ Prefetching funciona
    }
}

// Cache-unfriendly (acesso por colunas)
for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
        pixel = img.at(y, x);  // ✗ Cache misses
    }
}
```

## 🎓 Conceitos Avançados

### 1. Separable Convolution

**Problema:** Kernel 2D de tamanho K×K requer K² operações por pixel

**Solução:** Separar em duas passadas 1D (horizontal + vertical)

```
Kernel 2D (5×5):        Kernel 1D horizontal (1×5):
┌──────────────┐       ┌────────────┐
│ G₀₀ ... G₀₄ │       │ g₀ ... g₄ │
│ ... ... ... │  ==>  └────────────┘
│ G₄₀ ... G₄₄ │
└──────────────┘       Kernel 1D vertical (5×1):
                       ┌──┐
                       │g₀│
                       │..│
                       │g₄│
                       └──┘
```

**Complexidade:**
- 2D: O(n × m × k²) = O(n × m × 25) para kernel 5×5
- Separável: O(n × m × 2k) = O(n × m × 10) para kernel 5×5
- **Speedup teórico:** 2.5x

### 2. Unsharp Masking

**Conceito:** Realçar detalhes subtraindo uma versão borrada

```
High-frequency details = Original - Blur(Original)
Sharpened = Original + α × High-frequency details
```

**Parâmetros:**
- `σ` (sigma): Controla a escala dos detalhes (menor = detalhes finos)
- `α` (strength): Controla intensidade do realce

### 3. Bilateral Filtering

**Conceito:** Blur que preserva bordas combinando similaridade espacial e de cor

```
Weight(p, q) = Gaussian_spatial(||p - q||) × Gaussian_color(|I(p) - I(q)|)
```

**Intuição:**
- Pixels espacialmente próximos E com cores similares → peso alto
- Pixels próximos MAS com cores diferentes (borda) → peso baixo

### 4. CLAHE (Histogram Equalization Adaptativo)

**Problema com HE global:** Amplifica ruído, perde detalhes locais

**Solução CLAHE:**
1. Divide imagem em tiles
2. Aplica HE independente em cada tile
3. Limita amplificação (clip limit)
4. Interpola entre tiles para suavizar

**Vantagem:** Realça contraste local sem destruir informação global

## 📚 Referências Técnicas

### OpenMP

- **Diretivas:** `#pragma omp parallel for`
- **Scheduling:** `schedule(static|dynamic|guided, chunk_size)`
- **Controle:** `OMP_NUM_THREADS`, `omp_get_max_threads()`

### OpenCV

- **Tipos:** `CV_8UC3` (8-bit unsigned, 3 canais), `CV_32FC3` (32-bit float, 3 canais)
- **Acesso:** `mat.ptr<type>(y)[x]` (rápido), `mat.at<type>(y,x)` (com bounds checking)
- **Conversão:** `convertTo(dst, type, scale, offset)`

### Algoritmos

- **Gaussian Blur:** Separable convolution
- **Sobel Operator:** Gradient detection
- **Laplacian:** Second derivative (blur detection)
- **Histogram Equalization:** Contrast enhancement
- **Color Spaces:** BGR ↔ LAB (D65 illuminant)

## 🔍 Troubleshooting

### Problema: Performance ruim com muitas threads

**Causa:** Overhead de sincronização supera benefício da paralelização

**Solução:**
```bash
# Testar diferentes configurações
for t in 1 2 4 8; do
  OMP_NUM_THREADS=$t ./preprocess_optimized img.jpg out.jpg sharpen
done
```

### Problema: Imagem muito escura após CLAHE

**Causa:** clipLimit muito alto

**Solução:** Reduzir clipLimit no código (default: 2.0 → 1.5)

### Problema: Sharpening muito agressivo

**Causa:** strength muito alto

**Solução:** Reduzir strength no código (default: 1.5 → 1.0)

## 🚀 Performance Esperada

### Tempos Típicos (Imagem 960×505, 8 threads)

| Filtro            | Tempo (ms) | Speedup vs 1 thread |
|-------------------|------------|---------------------|
| Gaussian Blur     | 784.6      | 1.21x               |
| Unsharp Mask      | 796.8      | 1.32x               |
| Bilateral Denoise | 3599.1     | 1.97x               |
| CLAHE             | 636.1      | 1.10x               |
| Edge Enhancement  | 609.4      | 1.13x               |

**Observações:**
- **Bilateral Denoise** tem melhor speedup (mais computação por pixel)
- **CLAHE** tem menor speedup (conversão de cor serial no OpenCV padrão)
- Nossa implementação paralela de conversão BGR↔LAB melhora CLAHE significativamente

## 💡 Dicas para Desenvolvedores

### 1. Adicionar Novo Filtro

```cpp
// 1. Adicionar ao enum
enum class FilterType {
  // ...
  MEU_NOVO_FILTRO
};

// 2. Implementar método
cv::Mat applyOptimizedMeuFiltro(const cv::Mat& img, params...) {
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < img.rows; y++) {
    // Processamento paralelo
  }
  return result;
}

// 3. Adicionar ao switch em applySingleFilter()
case FilterType::MEU_NOVO_FILTRO:
  result = applyOptimizedMeuFiltro(img, params...);
  break;

// 4. Atualizar main() para aceitar novo filtro
```

### 2. Debugar Performance

```cpp
// Adicionar timing detalhado
auto start = std::chrono::high_resolution_clock::now();
// ... código ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration<double, std::milli>(end - start);
std::cout << "Fase X: " << duration.count() << " ms\n";
```

### 3. Otimizar Cache

```cpp
// Processar em tiles para melhor localidade de cache
const int TILE_SIZE = 64;
for (int tileY = 0; tileY < height; tileY += TILE_SIZE) {
  for (int tileX = 0; tileX < width; tileX += TILE_SIZE) {
    // Processar tile [tileY:tileY+TILE_SIZE, tileX:tileX+TILE_SIZE]
  }
}
```

---

**Autor:** Sistema de Preprocessamento do Pipeline de Visão Computacional
**Linguagem:** C++17
**Bibliotecas:** OpenCV 4.x, OpenMP 5.0+
**Licença:** Acadêmico/Pesquisa
