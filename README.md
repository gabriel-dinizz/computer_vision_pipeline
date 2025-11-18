# Pipeline de Visão Computacional - Preprocessamento OpenMP + YOLO

Pipeline de visão computacional focado em pesquisa, combinando preprocessamento C++ paralelizado com OpenMP e detecção de objetos YOLO usando PyTorch.

## 📐 Arquitetura do Sistema

```
┌─────────────┐      ┌──────────────────────┐      ┌─────────────────┐      ┌──────────┐
│   Imagem    │ ───> │  Preprocessamento    │ ───> │  Detecção YOLO  │ ───> │ Resultado│
│   de Input  │      │  C++ (OpenMP)        │      │  (PyTorch)      │      │          │
└─────────────┘      └──────────────────────┘      └─────────────────┘      └──────────┘
                              ↓                              ↓
                     • Blur Gaussiano              • Modelo YOLOv5
                     • Sharpening                  • Inferência CPU
                     • Denoising                   • Detecção de objetos
                     • CLAHE                       • Confiança configurável
                     • Edge Enhancement
                     • Paralelização OpenMP
```

### Componentes Principais

1. **`core_pipeline/src/preprocess_optimized.cpp`** - Preprocessamento paralelizado com OpenMP
   - Filtros: Blur Gaussiano, Sharpening, Bilateral Filtering, CLAHE, Edge Enhancement
   - Algoritmos otimizados para cache (tile-based processing)
   - Seleção automática de filtro baseada em análise de qualidade

2. **`core_pipeline/python/yolo_detector.py`** - Detecção de objetos com YOLO (PyTorch)
   - Inferência otimizada para CPU usando ultralytics YOLO
   - Limiar de confiança configurável
   - Saída em JSON e formato visual

3. **`core_pipeline/python/pipeline_integration.py`** - Orquestração do pipeline
   - Coordenação entre C++ e Python via subprocessos
   - Medição de performance e benchmarking
   - Agregação de resultados

4. **`core_pipeline/benchmark/benchmark_academic.py`** - Benchmark acadêmico
   - Análise de performance com 1, 2, 4 e 8 threads
   - Métricas de speedup e eficiência paralela
   - Relatórios de qualidade acadêmica

## 🚀 Configuração Inicial

### 1. Criar Ambiente Virtual

```bash
# Criar ambiente virtual Python
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate
```

### 2. Instalar Dependências do Sistema

**macOS:**
```bash
brew install opencv pkg-config libomp
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install build-essential pkg-config libopencv-dev python3 python3-pip
```

### 3. Instalar Dependências Python

```bash
pip3 install -r requirements.txt
```

**Principais bibliotecas instaladas:**
- `torch`, `torchvision` - PyTorch para YOLO
- `ultralytics` - Framework YOLO
- `opencv-python-headless` - Processamento de imagem
- `numpy`, `pandas` - Análise de dados
- `matplotlib`, `seaborn` - Visualização de resultados

### 4. Compilar o Preprocessador C++

```bash
make
```

Isso compila o binário `core_pipeline/bin/preprocess_optimized` com suporte a OpenMP.

## 📖 Como Usar

### Opção 1: Usando o Makefile (Recomendado)

O Makefile oferece comandos simplificados para executar o pipeline:

#### Testar o Sistema
```bash
# Testar com imagem de exemplo
make test
```

#### Executar Pipeline Completo
```bash
# Pipeline completo (preprocessamento + detecção YOLO)
make pipeline IMAGE=core_pipeline/images/sample.jpg

# Com controle de threads (escolha quantos cores usar)
OMP_NUM_THREADS=8 make pipeline IMAGE=core_pipeline/images/sample_4.jpg
OMP_NUM_THREADS=4 make pipeline IMAGE=core_pipeline/images/sample.jpg
OMP_NUM_THREADS=1 make pipeline IMAGE=core_pipeline/images/sample.jpg  # baseline single-thread
```

#### Executar Benchmark
```bash
# Benchmark completo com análise acadêmica
make benchmark IMAGE=core_pipeline/images/sample_4.jpg

# Benchmark com número específico de threads
OMP_NUM_THREADS=8 make benchmark IMAGE=core_pipeline/images/sample_4.jpg
```

#### Outros Comandos Úteis
```bash
make all           # Compila o preprocessador C++
make install-deps  # Instala dependências Python
make clean         # Remove arquivos compilados
make help          # Mostra todos os comandos disponíveis
```

### Opção 2: Usando o Script pipeline.sh

O script `pipeline.sh` oferece mais flexibilidade e opções para testar filtros individuais:

#### Aplicar Filtro Individual
```bash
# Executar apenas preprocessamento com filtro específico
./pipeline.sh preprocess core_pipeline/images/sample.jpg -f sharpen

# Com controle de threads
OMP_NUM_THREADS=4 ./pipeline.sh preprocess core_pipeline/images/sample.jpg -f sharpen

# Usando a flag -t (equivalente)
./pipeline.sh preprocess core_pipeline/images/sample.jpg -f sharpen -t 4
```

#### Filtros Disponíveis
- `auto` - Seleção automática baseada em análise de qualidade
- `blur` - Blur Gaussiano (redução de ruído)
- `sharpen` - Unsharp Mask (realce de detalhes)
- `denoise` - Filtro Bilateral (preserva bordas)
- `clahe` - Realce de contraste adaptativo
- `edge` - Detecção/realce de bordas

#### Pipeline Completo
```bash
# Pipeline completo (preprocessamento + YOLO)
./pipeline.sh process core_pipeline/images/sample.jpg -f sharpen

# Comparar com/sem preprocessamento
./pipeline.sh compare core_pipeline/images/sample.jpg -f denoise

# Apenas detecção YOLO (sem preprocessamento)
./pipeline.sh yolo-only core_pipeline/images/sample.jpg
```

#### Benchmark Rápido
```bash
# Benchmark rápido (5 execuções)
./pipeline.sh benchmark core_pipeline/images/sample.jpg

# Benchmark acadêmico completo (20-30 minutos)
./pipeline.sh academic-benchmark core_pipeline/images/sample_4.jpg
```

#### Avaliar Qualidade da Imagem
```bash
# Apenas avaliar qualidade sem processar
./pipeline.sh assess core_pipeline/images/sample.jpg
```

### Opção 3: Executar Scripts Python Diretamente

```bash
cd core_pipeline

# Pipeline completo
python3 python/pipeline_integration.py images/sample.jpg -f sharpen

# Apenas detecção YOLO
python3 python/yolo_detector.py images/sample.jpg --save

# Benchmark acadêmico
python3 benchmark/benchmark_academic.py --image images/sample_4.jpg
```

### Opção 4: Executar Binário C++ Diretamente

```bash
# Sintaxe: ./bin/preprocess_optimized <input> <output> <filtro>
./core_pipeline/bin/preprocess_optimized \
  core_pipeline/images/sample.jpg \
  core_pipeline/temp/output.jpg \
  sharpen
```

## ⚙️ Como Funciona a Execução

### 1. Fluxo de Execução do Pipeline

```
1. Imagem de entrada é carregada
   ↓
2. C++ preprocessa a imagem usando OpenMP
   - Análise de qualidade (blur, nitidez, contraste)
   - Seleção automática de filtro OU aplicação de filtro especificado
   - Processamento paralelo em múltiplos threads
   - Salva imagem preprocessada em core_pipeline/temp/
   ↓
3. Python carrega imagem preprocessada
   ↓
4. YOLO detecta objetos na imagem
   - Carrega modelo YOLOv5
   - Realiza inferência
   - Aplica threshold de confiança
   ↓
5. Resultados são salvos
   - Imagem com bounding boxes
   - JSON com detecções e coordenadas
   - Métricas de performance
```

### 2. Controle de Threads (OpenMP)

O número de threads usados pelo OpenMP pode ser controlado de 3 formas:

**A. Variável de ambiente (recomendado para testes):**
```bash
export OMP_NUM_THREADS=4
./pipeline.sh preprocess image.jpg -f blur
```

**B. Inline com o comando:**
```bash
OMP_NUM_THREADS=8 make pipeline IMAGE=image.jpg
```

**C. Flag `-t` no pipeline.sh:**
```bash
./pipeline.sh preprocess image.jpg -f sharpen -t 8
```

### 3. Paralelização OpenMP

O preprocessador C++ usa OpenMP para paralelizar operações em nível de linha/bloco:

```cpp
// Exemplo simplificado do código C++
#pragma omp parallel for schedule(dynamic, 4)
for (int y = 0; y < height; y++) {
    // Processar linha y em paralelo
    process_row(y);
}
```

**Benefícios da paralelização:**
- Speedup médio de 1.35x com 8 threads
- Melhor performance em filtros computacionalmente intensivos (denoise: 1.97x)
- Uso eficiente de CPUs multi-core

## 🗂️ Estrutura do Repositório

```
computer_vision_pipeline/
│
├── Makefile                           # Sistema de build principal
├── pipeline.sh                        # Script wrapper com opções avançadas
├── requirements.txt                   # Dependências Python
├── yolov5su.pt                       # Modelo YOLO pré-treinado (18MB)
├── README.md                          # Esta documentação
│
├── core_pipeline/                     # Código principal do pipeline
│   │
│   ├── src/
│   │   └── preprocess_optimized.cpp  # Preprocessamento OpenMP (algoritmo principal)
│   │
│   ├── python/
│   │   ├── pipeline_integration.py   # Orquestração C++ + Python
│   │   ├── yolo_detector.py          # Detecção YOLO
│   │   └── full_pipeline.py          # Wrapper de compatibilidade
│   │
│   ├── benchmark/
│   │   ├── benchmark_academic.py     # Benchmark de pesquisa
│   │   └── results_academic/         # Resultados e relatórios
│   │       ├── complete_results.json
│   │       ├── academic_summary.md
│   │       ├── speedup_analysis.png
│   │       └── preprocessing_performance.png
│   │
│   ├── bin/
│   │   └── preprocess_optimized      # Binário compilado (gerado por make)
│   │
│   ├── images/                        # Imagens de teste
│   │   ├── sample.jpg
│   │   ├── sample_4.jpg
│   │   ├── image.jpg
│   │   └── dataset/
│   │
│   ├── temp/                          # Saídas intermediárias (preprocessadas)
│   └── Makefile                       # Build do core_pipeline
│
└── venv/                              # Ambiente virtual Python (criado por você)
```

## 📊 Explicação do Makefile

O Makefile automatiza compilação e execução. Principais targets:

### Targets de Build
```makefile
make all          # Compila o preprocessador C++ com OpenMP
make clean        # Remove arquivos compilados e temporários
make install-deps # Instala dependências Python do requirements.txt
```

### Targets de Execução
```makefile
make test         # Testa com imagem de exemplo (sample.jpg)
make pipeline     # Executa pipeline completo (requer IMAGE=...)
make benchmark    # Executa benchmark acadêmico completo
```

### Variáveis Importantes
```makefile
IMAGE=...         # Caminho da imagem a ser processada
OMP_NUM_THREADS=N # Número de threads OpenMP (use como variável de ambiente)
```

### Exemplo de Uso Combinado
```bash
# Compilar + instalar + testar + executar em uma linha
make all install-deps test && OMP_NUM_THREADS=8 make pipeline IMAGE=core_pipeline/images/sample_4.jpg
```

## 📈 Benchmark Acadêmico

### O que é testado?

O benchmark acadêmico valida a hipótese de que paralelização OpenMP oferece ganhos mensuráveis de performance em preprocessamento de visão computacional.

**Testes realizados:**
- 5 filtros diferentes (blur, sharpen, denoise, clahe, edge)
- 4 configurações de threads (1, 2, 4, 8)
- Múltiplas iterações por configuração (warmup + benchmark)
- Análise estatística completa (média, desvio padrão, mediana, min, max)

### Como executar?

```bash
# Benchmark completo (20-30 minutos)
./pipeline.sh academic-benchmark core_pipeline/images/sample_4.jpg

# OU usando make
make benchmark IMAGE=core_pipeline/images/sample_4.jpg
```

### Resultados Gerados

Após a execução, os resultados são salvos em `core_pipeline/benchmark/results_academic/`:

1. **`complete_results.json`** - Dados brutos em JSON
2. **`academic_summary.md`** - Relatório formatado com tabelas
3. **`speedup_analysis.png`** - Gráfico de speedup por filtro
4. **`preprocessing_performance.png`** - Gráfico de performance comparativa

### Exemplo de Resultados

| Filtro   | 1 Thread (ms) | 8 Threads (ms) | Speedup | Eficiência |
|----------|---------------|----------------|---------|------------|
| Blur     | 948.6         | 784.6          | 1.21x   | 15%        |
| Sharpen  | 1051.1        | 796.8          | 1.32x   | 16%        |
| Denoise  | 7078.0        | 3599.1         | 1.97x   | 25%        |
| CLAHE    | 700.3         | 636.1          | 1.10x   | 14%        |
| Edge     | 690.0         | 609.4          | 1.13x   | 14%        |

**Conclusões:**
- Speedup médio com 8 threads: **1.35x**
- Melhor desempenho: **Denoise** (1.97x - quase 2x mais rápido)
- Melhoria média de performance: **34.6%** com paralelização

## 🎯 Imagens de Teste Incluídas

O repositório inclui várias imagens para teste em `core_pipeline/images/`:

```bash
# Imagem pequena para testes rápidos
core_pipeline/images/sample.jpg

# Imagem maior para benchmarks
core_pipeline/images/sample_4.jpg

# Outras imagens disponíveis
core_pipeline/images/image.jpg
core_pipeline/images/severely_degraded_demo.jpg
core_pipeline/images/dataset/  # Imagens adicionais
```

## 🔬 Aplicações de Pesquisa

Este pipeline serve como baseline para pesquisas em:

- **Algoritmos paralelos** de visão computacional
- **Comparação CPU vs GPU** em inferência
- **Impacto do preprocessamento** na acurácia de detecção
- **Arquiteturas híbridas C++/Python** para sistemas de CV
- **Otimização de cache** em processamento de imagens
- **Eficiência energética** em processamento paralelo

## 🛠️ Otimizações Implementadas

### No Preprocessador C++
- **Convolução separável** para Gaussian Blur (complexidade O(n) vs O(n²))
- **Processamento por blocos** otimizado para L1 cache (tiles 64x64)
- **Algoritmos memory-efficient** com mínima movimentação de dados
- **Scheduling dinâmico OpenMP** com balanceamento de carga

### Na Integração Python
- **Overhead mínimo** de subprocessos via I/O otimizado
- **YOLO otimizado para CPU** usando PyTorch sem dependências CUDA
- **Suporte a batch processing** para múltiplas imagens
- **Gerenciamento de memória** para processamento de imagens grandes

## 📝 Guia Rápido de Referência

### Primeiro Uso (Setup Completo)
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
make
make test
```

### Uso Diário
```bash
source venv/bin/activate  # Sempre ativar o ambiente virtual primeiro

# Pipeline completo
make pipeline IMAGE=core_pipeline/images/sample.jpg

# Testar filtro específico com 4 threads
OMP_NUM_THREADS=4 ./pipeline.sh preprocess core_pipeline/images/sample.jpg -f denoise

# Comparar diferentes configurações
./pipeline.sh compare core_pipeline/images/sample.jpg -f sharpen
```

### Análise de Performance
```bash
# Benchmark rápido (5 runs)
./pipeline.sh benchmark core_pipeline/images/sample.jpg

# Benchmark acadêmico completo
./pipeline.sh academic-benchmark core_pipeline/images/sample_4.jpg

# Ver resultados
cat core_pipeline/benchmark/results_academic/academic_summary.md
```

## 🎓 Citação Acadêmica

Se você usar este pipeline em trabalhos acadêmicos, por favor referencie:
- Paralelização OpenMP em preprocessamento de visão computacional
- Otimização de pipeline de detecção de objetos baseado em CPU
- Integração híbrida C++/Python para sistemas de CV

## 📄 Licença

Uso acadêmico e de pesquisa. Consulte as diretrizes da sua instituição para aplicações comerciais.

---

**Desenvolvido para pesquisa em algoritmos paralelos de visão computacional**
