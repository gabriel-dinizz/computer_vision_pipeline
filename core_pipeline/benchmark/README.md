# Módulo de Benchmark Acadêmico - Documentação Técnica

Este módulo contém ferramentas de benchmark acadêmico para validação científica do impacto da paralelização OpenMP no pipeline de visão computacional.

## 📁 Estrutura do Módulo

```
benchmark/
├── benchmark_academic.py          # Script principal de benchmark (514 linhas)
├── yolov5su.pt                   # Modelo YOLO para testes (18MB)
├── tests/                        # Diretório de testes
│   ├── FINAL_RESULTS.md         # Resultados finais de otimização
│   ├── RESULTS_clahe_fix.md     # Resultados da otimização CLAHE
│   ├── RESULTS_sharpen_fix.md   # Resultados da otimização Sharpen
│   ├── quick_test.py            # Script de teste rápido
│   └── variance_test.py         # Teste de variância estatística
├── results_academic/             # Resultados gerados (criado após execução)
│   ├── complete_results.json    # Dados brutos completos
│   ├── academic_summary.md      # Relatório formatado
│   ├── preprocessing_performance.svg  # Gráfico de performance (vetorizado)
│   └── speedup_analysis.svg     # Gráfico de speedup e eficiência (vetorizado)
└── README.md                     # Esta documentação
```

## 🎯 Objetivo do Módulo

O módulo de benchmark foi desenvolvido para:

1. **Validar cientificamente** a hipótese de que paralelização OpenMP oferece ganhos mensuráveis
2. **Medir performance** de preprocessamento em diferentes configurações de threads
3. **Calcular métricas acadêmicas**: speedup, eficiência paralela, variância
4. **Gerar visualizações** de qualidade acadêmica (gráficos, tabelas)
5. **Produzir relatórios** para publicação científica
6. **Benchmarking end-to-end** do pipeline completo

## 🔬 Metodologia de Benchmark

### Hipótese de Pesquisa

> **"A paralelização OpenMP em preprocessamento de visão computacional oferece ganhos mensuráveis de performance em pipelines de detecção de objetos baseados em CPU."**

### Configuração de Testes

```python
# Configuração padrão do benchmark
thread_counts = [1, 2, 4, 8]           # Contagens de threads testadas
filters = ['blur', 'sharpen', 'denoise', 'clahe', 'edge']  # Filtros testados
iterations_per_test = 10               # Iterações por configuração
warmup_iterations = 3                  # Warm-up antes do benchmark
```

### Variáveis Controladas

```python
# Variáveis de ambiente para controle rigoroso
os.environ['OMP_PROC_BIND'] = 'true'   # Thread affinity
os.environ['OMP_PLACES'] = 'cores'     # Bind em cores físicos
os.environ['OMP_NUM_THREADS'] = str(N) # Número de threads
```

**Por que controlar essas variáveis?**
- `OMP_PROC_BIND='true'`: Previne migração de threads entre cores
- `OMP_PLACES='cores'`: Garante uso de cores físicos (não hyperthreading)
- Minimiza variância e garante resultados reproduzíveis

### Métricas Calculadas

#### 1. **Speedup (S)**

```
S(n) = T(1) / T(n)

Onde:
  T(1) = tempo com 1 thread (baseline)
  T(n) = tempo com n threads
  S(n) = speedup com n threads
```

**Interpretação:**
- `S(8) = 4.0` → 4x mais rápido com 8 threads
- `S(8) = 8.0` → Speedup linear ideal (raro!)
- `S(8) = 1.0` → Sem ganho de paralelização

#### 2. **Eficiência Paralela (E)**

```
E(n) = S(n) / n = T(1) / (n × T(n))

Onde:
  E(n) = eficiência com n threads
  Valor ideal: 1.0 (100%)
```

**Interpretação:**
- `E(8) = 1.0` (100%) → Eficiência perfeita (ideal)
- `E(8) = 0.5` (50%) → Metade da eficiência ideal
- `E(8) = 0.15` (15%) → Baixa eficiência (overhead alto)

#### 3. **Estatísticas de Tempo**

```python
{
    "times_ms": [234.5, 236.1, 233.8, ...],  # Todas as medições
    "mean_ms": 234.8,                         # Média
    "std_ms": 1.2,                            # Desvio padrão
    "min_ms": 233.8,                          # Mínimo
    "max_ms": 236.1,                          # Máximo
    "median_ms": 234.7                        # Mediana
}
```

## 🏗️ Arquitetura do Código

### Classe `AcademicCVBenchmark`

```
┌─────────────────────────────────────────────────────────┐
│         AcademicCVBenchmark                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Configuração:                                          │
│  • thread_counts = [1, 2, 4, 8]                         │
│  • filters = ['blur', 'sharpen', 'denoise', ...]        │
│  • iterations_per_test = 10                             │
│  • warmup_iterations = 3                                │
│                                                          │
│  Métodos Principais:                                    │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. run_preprocessing_benchmark()               │    │
│  │    • Testa todos os filtros                    │    │
│  │    • Testa todas as contagens de threads       │    │
│  │    • Warm-up + benchmark runs                  │    │
│  │    • Calcula estatísticas                      │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ 2. run_end_to_end_benchmark()                  │    │
│  │    • Pipeline completo (preproc + YOLO)        │    │
│  │    • Valida impacto end-to-end                 │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ 3. generate_academic_report()                  │    │
│  │    • Gráficos de performance                   │    │
│  │    • Gráficos de speedup/eficiência            │    │
│  │    • Relatório Markdown                        │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ 4. run_complete_study()                        │    │
│  │    • Orquestra benchmark completo              │    │
│  │    • Valida pré-requisitos                     │    │
│  │    • Salva resultados                          │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Métodos Auxiliares:                                    │
│  • _run_single_preprocessing()                          │
│  • _calculate_parallel_metrics()                        │
│  • _generate_summary_report()                           │
└─────────────────────────────────────────────────────────┘
```

## 📊 Métodos Principais

### 1. `run_preprocessing_benchmark()`

Benchmark principal de preprocessamento isolado.

```python
benchmark = AcademicCVBenchmark()
results = benchmark.run_preprocessing_benchmark("image.jpg")
```

**Processo:**

```
Para cada FILTRO em [blur, sharpen, denoise, clahe, edge]:
    Para cada THREAD_COUNT em [1, 2, 4, 8]:

        1. Configurar OMP_NUM_THREADS = THREAD_COUNT

        2. WARM-UP (3 iterações):
           • Executar preprocessador sem medir
           • Estabilizar cache e sistema

        3. BENCHMARK (10 iterações):
           • Executar preprocessador
           • Medir tempo de execução
           • Armazenar em array times[]

        4. CALCULAR ESTATÍSTICAS:
           • mean, std, min, max, median

        5. Se THREAD_COUNT > 1:
           • Calcular speedup = T(1) / T(n)
           • Calcular efficiency = speedup / n
```

**Saída exemplo:**

```python
{
    "metadata": {
        "image_path": "images/sample_4.jpg",
        "thread_counts": [1, 2, 4, 8],
        "filters": ["blur", "sharpen", "denoise", "clahe", "edge"],
        "iterations_per_test": 10,
        "timestamp": "2024-11-17 19:45:23"
    },
    "preprocessing": {
        "blur": {
            1: {
                "times_ms": [948.2, 949.1, 947.8, ...],
                "mean_ms": 948.6,
                "std_ms": 2.3,
                "min_ms": 947.8,
                "max_ms": 951.2,
                "median_ms": 948.4
            },
            2: { ... },
            4: { ... },
            8: {
                "times_ms": [784.1, 785.3, ...],
                "mean_ms": 784.6,
                "std_ms": 1.8,
                "min_ms": 782.9,
                "max_ms": 786.4,
                "median_ms": 784.5,
                "speedup": 1.21,
                "efficiency": 0.15
            }
        },
        "sharpen": { ... },
        // ... outros filtros
    }
}
```

**Mensagens de progresso:**

```
🔬 Running Academic Preprocessing Benchmark
Image: images/sample_4.jpg
Thread counts: [1, 2, 4, 8]
Filters: ['blur', 'sharpen', 'denoise', 'clahe', 'edge']
Iterations per test: 10

📊 Testing blur filter...
  🧵 1 threads: ... 948.6ms
  🧵 2 threads: ... 812.3ms
  🧵 4 threads: ... 798.1ms
  🧵 8 threads: ... 784.6ms

📊 Testing sharpen filter...
  🧵 1 threads: ... 1051.1ms
  🧵 2 threads: ... 856.2ms
  🧵 4 threads: ... 812.5ms
  🧵 8 threads: ... 796.8ms

...
```

### 2. `run_end_to_end_benchmark()`

Benchmark do pipeline completo (preprocessamento + YOLO).

```python
results = benchmark.run_end_to_end_benchmark("image.jpg")
```

**Processo:**

```
Para cada THREAD_COUNT em [1, 2, 4, 8]:

    1. Configurar OMP_NUM_THREADS = THREAD_COUNT

    2. Executar CVPipeline.run_full_pipeline() 5 vezes:
       • Preprocessamento (C++)
       • Detecção YOLO (Python)
       • Medir timing total
       • Contar detecções

    3. Calcular estatísticas:
       • Tempo total médio
       • Desvio padrão
       • Contagem média de detecções
```

**Saída exemplo:**

```python
{
    "end_to_end": {
        1: {
            "total_time_ms": {
                "mean": 1182.5,
                "std": 23.4
            },
            "detection_count": {
                "mean": 5.0,
                "std": 0.0
            },
            "all_times": [1175.2, 1189.3, 1180.1, 1184.7, 1183.2]
        },
        8: {
            "total_time_ms": {
                "mean": 1019.3,
                "std": 18.9
            },
            "detection_count": {
                "mean": 5.0,
                "std": 0.0
            },
            "all_times": [1015.6, 1028.4, 1012.8, 1019.5, 1020.2]
        }
    }
}
```

### 3. `generate_academic_report()`

Gera visualizações e relatórios de qualidade acadêmica.

```python
benchmark.generate_academic_report(results, "image.jpg")
```

**Arquivos gerados:**

#### a. `preprocessing_performance.svg`

Gráfico de performance de preprocessamento (2×3 subplots, formato vetorizado):

```
┌────────────────────────────────────────────────────────┐
│  OpenMP Preprocessing Performance Analysis            │
├──────────────┬──────────────┬──────────────────────────┤
│ Blur Filter  │Sharpen Filter│ Denoise Filter          │
│              │              │                          │
│ [Gráfico]    │ [Gráfico]    │ [Gráfico]               │
│ Thread Count │ Thread Count │ Thread Count            │
│ vs Time(ms)  │ vs Time(ms)  │ vs Time(ms)             │
├──────────────┼──────────────┼──────────────────────────┤
│ CLAHE Filter │ Edge Filter  │                         │
│              │              │                          │
│ [Gráfico]    │ [Gráfico]    │                         │
│ Thread Count │ Thread Count │                          │
│ vs Time(ms)  │ vs Time(ms)  │                         │
└──────────────┴──────────────┴──────────────────────────┘
```

**Características:**
- Error bars mostrando desvio padrão
- Eixo X: Thread count [1, 2, 4, 8]
- Eixo Y: Processing time (ms)
- Grid para facilitar leitura
- Formato SVG vetorizado para qualidade de publicação

#### b. `speedup_analysis.svg`

Gráfico de métricas paralelas (1×2 subplots, formato vetorizado):

```
┌──────────────────────────────────────────────────────┐
│         Parallel Performance Metrics                 │
├────────────────────────┬─────────────────────────────┤
│ Speedup vs Thread Count│Parallel Efficiency vs Thread│
│                        │           Count              │
│ [Gráfico]              │ [Gráfico]                   │
│                        │                              │
│ • Blur                 │ • Blur                      │
│ • Sharpen              │ • Sharpen                   │
│ • Denoise              │ • Denoise                   │
│ • CLAHE                │ • CLAHE                     │
│ • Edge                 │ • Edge                      │
│ • Ideal Speedup (---)  │ • Perfect Efficiency (---)  │
│                        │                              │
│ Y: Speedup (x)         │ Y: Efficiency (0-1)         │
│ X: Thread Count        │ X: Thread Count             │
└────────────────────────┴─────────────────────────────┘
```

**Speedup Plot:**
- Linha tracejada preta: Speedup ideal (S = n)
- Linhas coloridas: Speedup real por filtro
- Quanto mais próximo da linha ideal, melhor

**Efficiency Plot:**
- Linha tracejada em y=1.0: Eficiência perfeita (100%)
- Linhas coloridas: Eficiência real por filtro
- Y-axis limitado a [0, 1.1]

#### c. `academic_summary.md`

Relatório Markdown formatado:

```markdown
# OpenMP Computer Vision Pipeline - Academic Benchmark Results

**Test Image:** images/sample_4.jpg
**Timestamp:** 2024-11-17 19:45:23
**Iterations per test:** 10

## Executive Summary

This benchmark validates the performance impact of OpenMP parallelization
in computer vision preprocessing pipelines for object detection.

## Preprocessing Performance Summary

| Filter   | 1 Thread (ms) | 8 Threads (ms) | Speedup | Efficiency |
|----------|---------------|----------------|---------|------------|
| Blur     | 948.6         | 784.6          | 1.21x   | 0.15       |
| Sharpen  | 1051.1        | 796.8          | 1.32x   | 0.16       |
| Denoise  | 7078.0        | 3599.1         | 1.97x   | 0.25       |
| Clahe    | 700.3         | 636.1          | 1.10x   | 0.14       |
| Edge     | 690.0         | 609.4          | 1.13x   | 0.14       |

## Key Findings

- **Average 8-thread speedup:** 1.35x
- **Average 8-thread efficiency:** 0.17
- **Performance gain:** 34.6% improvement with 8 threads

## Research Validation

The benchmark demonstrates that OpenMP parallelization provides measurable
performance improvements in CPU-based computer vision preprocessing,
validating the research hypothesis that classical parallelism techniques
remain effective in modern AI pipelines.

**Methodology:** Each test was repeated 10 times with 3 warmup iterations.
Thread affinity and controlled environment ensure reproducible results.
```

#### d. `complete_results.json`

JSON com dados brutos completos para análise posterior:

```json
{
  "metadata": { ... },
  "preprocessing": {
    "blur": {
      "1": { "times_ms": [...], "mean_ms": 948.6, ... },
      "2": { ... },
      "4": { ... },
      "8": { ... }
    },
    ...
  },
  "end_to_end": { ... }
}
```

### 4. `run_complete_study()`

Orquestra o estudo acadêmico completo.

```python
results = benchmark.run_complete_study("images/sample_4.jpg")
```

**Fluxo completo:**

```
┌──────────────────────────────────────────────────┐
│ 1. Validar Pré-requisitos                       │
│    • Binário C++ existe?                         │
│    • Imagem de teste existe?                     │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│ 2. Preprocessing Benchmark                       │
│    • 5 filtros × 4 thread counts                 │
│    • 10 iterações por configuração               │
│    • ~200 execuções total                        │
│    • Tempo estimado: 15-20 minutos               │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│ 3. End-to-End Benchmark                         │
│    • Pipeline completo                           │
│    • 4 thread counts × 5 iterações               │
│    • Tempo estimado: 5-10 minutos                │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│ 4. Gerar Relatório Acadêmico                    │
│    • Gráficos PNG (300 DPI)                      │
│    • Relatório Markdown                          │
│    • JSON com dados brutos                       │
└─────────────────┬────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────────┐
│ 5. Salvar Resultados                            │
│    • results_academic/complete_results.json      │
│    • results_academic/academic_summary.md        │
│    • results_academic/preprocessing_performance.svg│
│    • results_academic/speedup_analysis.svg       │
└──────────────────────────────────────────────────┘
```

## 🚀 Como Usar

### Uso Básico

```bash
# Do diretório core_pipeline/benchmark/
python3 benchmark_academic.py --image ../images/sample_4.jpg
```

### Opções CLI

```bash
python3 benchmark_academic.py [OPÇÕES]

Opções:
  --image PATH              # Imagem de teste (default: ../images/sample_4.jpg)
  --preprocess-bin PATH     # Binário C++ (default: ../bin/preprocess_optimized)
```

### Exemplos

```bash
# Benchmark padrão
python3 benchmark_academic.py

# Com imagem customizada
python3 benchmark_academic.py --image /path/to/image.jpg

# Com binário em local customizado
python3 benchmark_academic.py --preprocess-bin /path/to/bin/preprocess_optimized
```

### Uso Programático

```python
from benchmark_academic import AcademicCVBenchmark

# Criar instância
benchmark = AcademicCVBenchmark(
    preprocess_bin="../bin/preprocess_optimized"
)

# Executar estudo completo
results = benchmark.run_complete_study("../images/sample_4.jpg")

# OU executar partes individuais

# Apenas preprocessing benchmark
prep_results = benchmark.run_preprocessing_benchmark("image.jpg")

# Apenas end-to-end benchmark
e2e_results = benchmark.run_end_to_end_benchmark("image.jpg")

# Gerar relatório de resultados existentes
benchmark.generate_academic_report(results, "image.jpg")
```

## 📈 Resultados Esperados

### Tempos Típicos (Imagem 960×505)

| Filtro   | 1 Thread | 2 Threads | 4 Threads | 8 Threads | Speedup (8T) | Eficiência (8T) |
|----------|----------|-----------|-----------|-----------|--------------|-----------------|
| Blur     | 948.6 ms | 812.3 ms  | 798.1 ms  | 784.6 ms  | 1.21x        | 15%             |
| Sharpen  | 1051.1 ms| 856.2 ms  | 812.5 ms  | 796.8 ms  | 1.32x        | 16%             |
| Denoise  | 7078.0 ms| 3689.4 ms | 2133.2 ms | 3599.1 ms | 1.97x        | 25%             |
| CLAHE    | 700.3 ms | 668.2 ms  | 652.4 ms  | 636.1 ms  | 1.10x        | 14%             |
| Edge     | 690.0 ms | 639.5 ms  | 619.8 ms  | 609.4 ms  | 1.13x        | 14%             |

**Observações:**
- **Denoise** tem melhor speedup (mais computação por pixel)
- **Blur/Sharpen** têm speedup moderado
- **CLAHE/Edge** têm menor speedup (mais leves ou overhead serial)

### Análise de Amdahl's Law

A **Lei de Amdahl** explica por que o speedup não é linear:

```
Speedup_max = 1 / (s + (p / n))

Onde:
  s = fração serial do código (0-1)
  p = fração paralela do código (0-1), onde s + p = 1
  n = número de threads
```

**Exemplo - Denoise:**
- Speedup observado: 1.97x com 8 threads
- Estimativa de fração serial: ~10-15%
- Fração paralela: ~85-90%

**Exemplo - CLAHE (antes de otimização):**
- Speedup observado: 1.00x (sem ganho!)
- Fração serial: ~100% (conversão de cor era serial)
- **Solução:** Paralelizar conversão BGR↔LAB

## 🔬 Validação Estatística

### Variância e Reproduzibilidade

```python
# Exemplo de análise de variância
import statistics

times = [948.2, 949.1, 947.8, 948.5, 949.3, 948.0, 948.7, 949.0, 947.9, 948.4]

mean = statistics.mean(times)         # 948.69 ms
std = statistics.stdev(times)         # 0.52 ms
cv = (std / mean) * 100               # 0.055% (coef. de variação)

# CV < 2% indica alta reproduzibilidade
```

**Critérios de qualidade:**
- **CV < 1%:** Excelente reproduzibilidade
- **CV 1-5%:** Boa reproduzibilidade
- **CV > 5%:** Variância alta (investigar causas)

### Significância Estatística

Para determinar se diferenças são estatisticamente significativas:

```python
from scipy import stats

# Teste t para comparar 1 thread vs 8 threads
times_1t = [948.2, 949.1, 947.8, ...]
times_8t = [784.1, 785.3, 783.9, ...]

t_stat, p_value = stats.ttest_ind(times_1t, times_8t)

# p < 0.05 indica diferença estatisticamente significativa
if p_value < 0.05:
    print(f"Diferença significativa (p={p_value:.4f})")
```

## 📊 Interpretação de Resultados

### Como Ler o Gráfico de Speedup

```
  Speedup
    8│              ╱ Ideal (S = n)
     │            ╱
    6│          ╱
     │        ╱   • Denoise (1.97x)
    4│      ╱
     │    ╱  • Sharpen (1.32x)
    2│  ╱   • Blur (1.21x)
     │ • CLAHE (1.10x), Edge (1.13x)
    0└────────────────────────────>
      1    2    4    8    Thread Count
```

**Interpretação:**
- **Acima da linha ideal:** Impossível (violaria física!)
- **Próximo da linha ideal:** Excelente paralelização
- **Longe da linha ideal:** Overhead ou fração serial alta
- **Abaixo de 1.0:** Degradação (raro, indica problema)

### Como Ler o Gráfico de Eficiência

```
  Efficiency
   1.0│─────  Perfect (100%)
      │
   0.8│
      │
   0.6│
      │  • Denoise (25% em 8T)
   0.4│
      │
   0.2│  • Outros filtros (14-16% em 8T)
      │
   0.0└────────────────────────────>
       1    2    4    8    Thread Count
```

**Interpretação:**
- **E > 0.5 (50%):** Excelente eficiência
- **E 0.2-0.5 (20-50%):** Boa eficiência
- **E < 0.2 (20%):** Baixa eficiência (considerar otimizar)

## 🛠️ Configuração do Benchmark

### Ajustar Número de Iterações

```python
benchmark = AcademicCVBenchmark()

# Para testes rápidos (menos preciso)
benchmark.iterations_per_test = 3
benchmark.warmup_iterations = 1

# Para benchmark rigoroso (mais preciso, mais lento)
benchmark.iterations_per_test = 20
benchmark.warmup_iterations = 5
```

### Testar Thread Counts Customizados

```python
benchmark.thread_counts = [1, 2, 3, 4, 6, 8, 12, 16]
```

### Testar Filtros Específicos

```python
# Apenas CLAHE e Denoise
benchmark.filters = ['clahe', 'denoise']

# Todos os filtros (padrão)
benchmark.filters = ['blur', 'sharpen', 'denoise', 'clahe', 'edge']
```

## 🔍 Análise de Resultados Históricos

### FINAL_RESULTS.md - Principais Descobertas

Do arquivo `tests/FINAL_RESULTS.md`, as otimizações realizadas mostraram:

#### Fix #1: Sharpen - Static Scheduling

**Mudança:** `schedule(dynamic, 8)` → `schedule(static)`

**Resultado:**
- Fix de regressão em 8 threads (+7% de melhoria)
- Performance mais consistente entre thread counts

#### Fix #2: CLAHE - Parallel Color Conversion ⭐

**Mudança:** Conversão de cor serial → paralela customizada

**Resultado IMPRESSIONANTE:**
- 1 thread: **2.88x mais rápido** (90ms → 31ms)
- 8 threads: **10.4x mais rápido** (90ms → 9ms)
- Eficiência: 12.6% → 45.5% (**3.6x melhoria**)

**Por que funcionou?**
```python
# ANTES (serial):
cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab)  # Gargalo!

# DEPOIS (paralelo):
#pragma omp parallel for schedule(static)
for (int y = 0; y < rows; y++) {
    # Conversão BGR→LAB pixel por pixel
}
```

### Lições Aprendidas

1. **Amdahl's Law é real:**
   - CLAHE tinha 0% speedup porque conversão de cor era 100% serial
   - Paralelizar o gargalo destravou ganhos massivos

2. **Nem todo filtro beneficia igualmente:**
   - Denoise: 1.97x speedup (muita computação)
   - Edge: 1.13x speedup (overhead domina)

3. **Dynamic scheduling tem custo:**
   - Sharpen melhorou ao mudar para static
   - Overhead de sincronização > benefício de balanceamento

## 🧪 Scripts de Teste

### `tests/quick_test.py`

Teste rápido de sanidade:

```python
#!/usr/bin/env python3
"""Quick sanity test for benchmark"""

import sys
sys.path.append('..')
from benchmark_academic import AcademicCVBenchmark

benchmark = AcademicCVBenchmark()
benchmark.iterations_per_test = 2  # Rápido
benchmark.warmup_iterations = 1
benchmark.filters = ['blur']       # Apenas 1 filtro
benchmark.thread_counts = [1, 8]   # Apenas extremos

results = benchmark.run_complete_study("../../images/sample.jpg")
print("✅ Quick test passed!")
```

### `tests/variance_test.py`

Teste de variância estatística:

```python
#!/usr/bin/env python3
"""Test statistical variance of benchmark"""

import statistics
from benchmark_academic import AcademicCVBenchmark

benchmark = AcademicCVBenchmark()
benchmark.iterations_per_test = 30  # Muitas iterações

results = benchmark.run_preprocessing_benchmark("../../images/sample.jpg")

# Análise de variância
for filter_type, filter_data in results['preprocessing'].items():
    for threads, data in filter_data.items():
        cv = (data['std_ms'] / data['mean_ms']) * 100
        print(f"{filter_type:10} {threads}T: CV = {cv:.2f}%")

        if cv > 5:
            print(f"  ⚠️  Alta variância!")
```

## 📝 Boas Práticas para Benchmarking

### 1. Ambiente Controlado

```bash
# Desabilitar Turbo Boost (Intel)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Desabilitar CPU frequency scaling
sudo cpupower frequency-set --governor performance

# Fechar aplicações pesadas
# Desconectar rede (opcional)
```

### 2. Múltiplas Execuções

```bash
# Executar benchmark 3 vezes em dias diferentes
for i in 1 2 3; do
    echo "Run $i"
    python3 benchmark_academic.py
    mv results_academic results_academic_run$i
    sleep 3600  # 1 hora de intervalo
done

# Comparar resultados
diff results_academic_run1/academic_summary.md \
     results_academic_run2/academic_summary.md
```

### 3. Validação de Consistência

```python
# Verificar se detecções são consistentes
def validate_detection_consistency(results):
    """Verifica se número de detecções é o mesmo em todos os threads"""
    counts = []
    for tc, data in results['end_to_end'].items():
        counts.append(data['detection_count']['mean'])

    # Todos devem ser iguais
    if len(set(counts)) == 1:
        print("✅ Detection counts consistent across threads")
    else:
        print("⚠️  Detection counts vary!")
        print(counts)
```

## 🎓 Uso Acadêmico

### Para Papers

**Figuras para incluir:**
1. `preprocessing_performance.svg` - Performance por filtro (formato vetorizado)
2. `speedup_analysis.svg` - Speedup e eficiência (formato vetorizado)

**Tabelas para incluir:**
```markdown
Performance summary table from academic_summary.md
```

**Métricas para reportar:**
- Average speedup
- Average efficiency
- Best-case speedup (qual filtro?)
- Worst-case speedup
- Variância média

### Citação

```bibtex
@misc{cv_pipeline_benchmark,
  title={OpenMP Parallelization in Computer Vision Preprocessing},
  author={[Seu Nome]},
  year={2024},
  note={Academic benchmark validating OpenMP performance in CV pipelines}
}
```

## 💡 Dicas de Otimização

### Se Speedup é Baixo (<1.5x com 8 threads)

1. **Identificar gargalo:**
   ```bash
   # Profiling com perf
   perf record -g ./bin/preprocess_optimized input.jpg output.jpg filter
   perf report
   ```

2. **Verificar fração serial:**
   - Conversão de cor serial?
   - I/O de arquivo?
   - Alocação de memória?

3. **Paralelizar gargalo:**
   - Adicionar `#pragma omp parallel for`
   - Implementação customizada se necessário

### Se Variância é Alta (CV > 5%)

1. **Aumentar warm-up:**
   ```python
   benchmark.warmup_iterations = 10
   ```

2. **Controlar ambiente:**
   - Desabilitar Turbo Boost
   - Fechar outras aplicações
   - Usar governor performance

3. **Aumentar iterações:**
   ```python
   benchmark.iterations_per_test = 30
   ```

## 🐛 Troubleshooting

### Erro: "Preprocessing binary not found"

```bash
# Compilar binário
cd ../
make all

# Verificar se existe
ls -l bin/preprocess_optimized
```

### Erro: "Test image not found"

```bash
# Verificar caminho
ls -l ../images/sample_4.jpg

# Usar caminho absoluto
python3 benchmark_academic.py --image /full/path/to/image.jpg
```

### Erro: "Module 'matplotlib' not found"

```bash
# Instalar dependências
pip3 install matplotlib seaborn numpy scipy
```

### Erro: Timeout

```python
# Aumentar timeout em _run_single_preprocessing()
result = subprocess.run(cmd, timeout=60)  # 60 segundos
```

## 📊 Exemplo de Saída Completa

```
🎓 Starting Complete Academic Study
============================================================

🔬 Running Academic Preprocessing Benchmark
Image: ../images/sample_4.jpg
Thread counts: [1, 2, 4, 8]
Filters: ['blur', 'sharpen', 'denoise', 'clahe', 'edge']
Iterations per test: 10

📊 Testing blur filter...
  🧵 1 threads: ... 948.6ms
  🧵 2 threads: ... 812.3ms
  🧵 4 threads: ... 798.1ms
  🧵 8 threads: ... 784.6ms

📊 Testing sharpen filter...
  🧵 1 threads: ... 1051.1ms
  🧵 2 threads: ... 856.2ms
  🧵 4 threads: ... 812.5ms
  🧵 8 threads: ... 796.8ms

📊 Testing denoise filter...
  🧵 1 threads: ... 7078.0ms
  🧵 2 threads: ... 3689.4ms
  🧵 4 threads: ... 2133.2ms
  🧵 8 threads: ... 3599.1ms

📊 Testing clahe filter...
  🧵 1 threads: ... 700.3ms
  🧵 2 threads: ... 668.2ms
  🧵 4 threads: ... 652.4ms
  🧵 8 threads: ... 636.1ms

📊 Testing edge filter...
  🧵 1 threads: ... 690.0ms
  🧵 2 threads: ... 639.5ms
  🧵 4 threads: ... 619.8ms
  🧵 8 threads: ... 609.4ms

🔄 Running End-to-End Pipeline Benchmark
🧵 Testing 1 threads...
🧵 Testing 2 threads...
🧵 Testing 4 threads...
🧵 Testing 8 threads...

📊 Generating Academic Report...
📄 Summary report: results_academic/academic_summary.md
✅ Academic report generated in results_academic/

🏆 Complete study finished! Results in results_academic/

✅ Academic benchmark completed successfully!
```

---

**Autor:** Sistema de Benchmark Acadêmico do Pipeline de Visão Computacional
**Linguagem:** Python 3.8+
**Bibliotecas:** matplotlib, seaborn, numpy, scipy
**Licença:** Acadêmico/Pesquisa
