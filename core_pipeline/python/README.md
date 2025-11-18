# Módulo Python - Documentação Técnica

Este módulo contém a implementação Python do pipeline de visão computacional, incluindo detecção YOLO e integração com o preprocessador C++.

## 📁 Estrutura do Módulo

```
python/
├── yolo_detector.py           # Detector YOLO standalone (289 linhas)
├── pipeline_integration.py    # Integração C++ + Python (207 linhas)
├── yolov5su.pt               # Modelo YOLO pré-treinado (18MB)
└── README.md                  # Esta documentação
```

## 🎯 Objetivo do Módulo

O módulo Python foi desenvolvido para:

1. **Detectar objetos** em imagens usando YOLO (PyTorch)
2. **Integrar** o preprocessador C++ com o detector YOLO
3. **Orquestrar** o pipeline completo (preprocessing → detecção)
4. **Medir performance** de cada componente
5. **Salvar resultados** (imagens com bounding boxes, JSON, arquivos de texto)
6. **Benchmarking** de performance do pipeline completo

## 🏗️ Arquitetura do Código

### Visão Geral

```
┌─────────────────────────────────────────────────────────┐
│                  Módulo Python                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  yolo_detector.py                             │    │
│  │                                                │    │
│  │  Classe: YOLODetector                         │    │
│  │  • detect_objects()      - Detecção única     │    │
│  │  • benchmark_detection() - Benchmark YOLO     │    │
│  │  • _save_detection_results() - Salvar         │    │
│  │                                                │    │
│  │  Dependências:                                │    │
│  │  • ultralytics (YOLO)                         │    │
│  │  • PyTorch                                    │    │
│  │  • OpenCV                                     │    │
│  │  • NumPy                                      │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  pipeline_integration.py                      │    │
│  │                                                │    │
│  │  Classe: CVPipeline                           │    │
│  │  • run_full_pipeline()   - Pipeline completo  │    │
│  │  • benchmark_pipeline()  - Benchmark pipeline │    │
│  │                                                │    │
│  │  Integração:                                  │    │
│  │  • subprocess → C++ preprocessor              │    │
│  │  • import → YOLODetector                      │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  yolov5su.pt                                  │    │
│  │  • Modelo YOLOv5 Small (18MB)                 │    │
│  │  • 80 classes (COCO dataset)                  │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 📦 Arquivo 1: `yolo_detector.py`

### Classe YOLODetector

Detector YOLO standalone para inferência e benchmarking.

#### Inicialização

```python
detector = YOLODetector(
    weights="yolov5s.pt",    # Caminho do modelo
    device="cpu",             # cpu, cuda, ou mps
    conf_thresh=0.25,        # Limiar de confiança (0.0-1.0)
    iou_thresh=0.45          # Limiar de IoU para NMS
)
```

**Parâmetros:**
- `weights` (str): Caminho para arquivo de pesos do modelo YOLO
- `device` (str): Dispositivo de inferência (`cpu`, `cuda`, `mps`)
- `conf_thresh` (float): Limiar de confiança mínima para detecções (0.0-1.0)
- `iou_thresh` (float): Limiar de IoU para Non-Maximum Suppression

**Atributos:**
- `model`: Instância do modelo YOLO (ultralytics)
- `img_size`: Tamanho de entrada da imagem (640x640)

### Métodos Principais

#### 1. `detect_objects()`

Executa detecção YOLO em uma única imagem.

```python
results = detector.detect_objects(
    image_path="path/to/image.jpg",
    save_results=True,
    output_dir="runs/detect/exp1"
)
```

**Argumentos:**
- `image_path` (str|Path): Caminho da imagem de entrada
- `save_results` (bool): Se True, salva imagem com bounding boxes
- `output_dir` (str, opcional): Diretório para salvar resultados

**Retorno (Dict):**
```python
{
    "image_path": "path/to/image.jpg",
    "detections": [
        {
            "class": 0,                    # ID da classe
            "class_name": "person",        # Nome da classe
            "confidence": 0.89,            # Confiança (0-1)
            "bbox": [x1, y1, x2, y2],     # Bounding box
            "center": [cx, cy],            # Centro do bbox
            "area": 12345                  # Área do bbox (pixels)
        },
        # ... mais detecções
    ],
    "detection_count": 5,                  # Número de objetos detectados
    "timing": {
        "preprocess": 0.0,                 # ms (interno do ultralytics)
        "inference": 234.5,                # ms
        "postprocess": 12.3,               # ms
        "total": 246.8                     # ms
    },
    "model_info": {
        "weights": "yolov5s.pt",
        "device": "cpu",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    }
}
```

**Processo interno:**

1. **Carrega imagem** com OpenCV
2. **Inferência YOLO** usando `ultralytics` library
   ```python
   results = self.model(image_path, conf=conf_thresh, device=device)
   ```
3. **Extrai detecções:**
   - Bounding boxes (xyxy format)
   - Confidências
   - Classes
4. **Pós-processamento:**
   - Converte tensors PyTorch para NumPy
   - Calcula centros e áreas
   - Formata como dicionários
5. **Salva resultados** (opcional):
   - Imagem com bounding boxes desenhados
   - Arquivo `.txt` com detecções (formato YOLO)

#### 2. `benchmark_detection()`

Executa benchmark de performance do detector YOLO.

```python
results = detector.benchmark_detection(
    image_path="image.jpg",
    num_runs=10
)
```

**Argumentos:**
- `image_path` (str|Path): Caminho da imagem de teste
- `num_runs` (int): Número de execuções para média (padrão: 10)

**Retorno (Dict):**
```python
{
    "image_path": "image.jpg",
    "num_runs": 10,
    "detection_count": {
        "mean": 5.0,              # Média de detecções
        "std": 0.0,               # Desvio padrão
        "all_runs": [5, 5, 5, ...]  # Detecções por run
    },
    "timing_stats": {
        "preprocess": {
            "mean": 1.2,
            "std": 0.3,
            "min": 0.9,
            "max": 1.8,
            "median": 1.1
        },
        "inference": { ... },
        "postprocess": { ... },
        "total": { ... }
    },
    "model_info": { ... }
}
```

**Processo:**
1. **Warm-up run:** 1 execução para carregar modelo na memória
2. **Benchmark runs:** N execuções cronometradas
3. **Cálculo de estatísticas:** Média, desvio padrão, min, max, mediana
4. **Retorna resultados agregados**

#### 3. `_save_detection_results()` (Privado)

Salva resultados da detecção em disco.

**Arquivos gerados:**

1. **Imagem com bounding boxes** (`detected_<nome>.jpg`):
   - Retângulos verdes ao redor de objetos
   - Labels com classe e confiança
   - Formato: `classe: 0.89`

2. **Arquivo de texto** (`<nome>.txt`):
   - Formato YOLO (normalizado xywh):
   ```
   class_id x_center y_center width height confidence
   0 0.512345 0.678901 0.234567 0.345678 0.891234
   1 0.234567 0.123456 0.111111 0.222222 0.765432
   ```

**Desenho de bounding boxes:**
```python
# Retângulo verde (BGR)
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Label com fundo verde
label = f"{class_name}: {conf:.2f}"
cv2.rectangle(img, (x1, y1-label_height), (x1+label_width, y1), (0, 255, 0), -1)
cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
```

### Interface CLI de `yolo_detector.py`

```bash
# Uso básico
python3 yolo_detector.py <imagem> [opções]

# Opções
-w, --weights <path>        # Pesos do modelo (default: yolov5s.pt)
-d, --device <device>       # cpu, cuda, ou mps (default: cpu)
-c, --confidence <float>    # Limiar de confiança (default: 0.25)
-o, --output <dir>          # Diretório de saída
-b, --benchmark <N>         # Executar benchmark com N runs
--save                      # Salvar resultados de detecção
```

**Exemplos:**

```bash
# Detecção simples
python3 yolo_detector.py image.jpg --save

# Com confiança customizada
python3 yolo_detector.py image.jpg -c 0.5 --save -o results/

# Benchmark com 10 runs
python3 yolo_detector.py image.jpg -b 10

# Usando GPU (CUDA)
python3 yolo_detector.py image.jpg -d cuda --save
```

**Saída exemplo:**

```
YOLO model loaded: yolov5s.pt
Device: cpu

=== Detection Results ===
Image: image.jpg
Detections: 5
Total time: 246.8 ms

Detected objects:
  1. person (0.89) at [120, 45, 250, 380]
  2. car (0.76) at [300, 200, 500, 350]
  3. bicycle (0.82) at [50, 150, 180, 280]
  4. dog (0.91) at [400, 250, 520, 400]
  5. backpack (0.67) at [200, 100, 280, 200]
```

## 📦 Arquivo 2: `pipeline_integration.py`

### Classe CVPipeline

Orquestra o pipeline completo: preprocessamento C++ → detecção YOLO.

#### Inicialização

```python
pipeline = CVPipeline(
    preprocess_bin="bin/preprocess_optimized"
)
```

**Parâmetros:**
- `preprocess_bin` (str): Caminho para o binário de preprocessamento C++

**Atributos:**
- `preprocess_bin` (Path): Caminho do binário C++
- `temp_dir` (Path): Diretório temporário (`temp/`)

### Métodos Principais

#### 1. `run_full_pipeline()`

Executa o pipeline completo de ponta a ponta.

```python
results = pipeline.run_full_pipeline(
    image_path="image.jpg",
    filter_type="auto",         # blur, sharpen, denoise, clahe, edge, auto
    confidence=0.25,            # Limiar YOLO
    device="cpu",               # cpu, cuda, mps
    save_results=True,
    output_dir="results/"
)
```

**Argumentos:**
- `image_path` (str): Caminho da imagem de entrada
- `filter_type` (str): Tipo de filtro de preprocessamento (padrão: "auto")
- `confidence` (float): Limiar de confiança YOLO (0.0-1.0)
- `device` (str): Dispositivo para YOLO
- `save_results` (bool): Salvar resultados de detecção
- `output_dir` (str, opcional): Diretório de saída

**Retorno (Dict):**
```python
{
    "input_image": "image.jpg",
    "filter_type": "auto",
    "timing": {
        "preprocessing": 456.7,    # ms
        "detection": 234.5,        # ms
        "total": 691.2             # ms
    },
    "preprocessing": {
        "success": True,
        "output_path": "temp/preprocessed_image.jpg"
    },
    "detection": {
        "image_path": "temp/preprocessed_image.jpg",
        "detections": [...],
        "detection_count": 5,
        "timing": { ... },
        "model_info": { ... }
    }
}
```

**Fluxo de execução:**

```
┌──────────────────────────────────────────────────┐
│ 1. Validação da imagem de entrada               │
└────────────────┬─────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────┐
│ 2. Preprocessamento C++ (subprocess)             │
│    • Executa: bin/preprocess_optimized           │
│    • Input: image.jpg                            │
│    • Output: temp/preprocessed_image.jpg         │
│    • Filtro: auto/blur/sharpen/denoise/clahe     │
│    • Captura stdout/stderr                       │
│    • Timeout: 60 segundos                        │
└────────────────┬─────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────┐
│ 3. Detecção YOLO (Python)                        │
│    • Cria YOLODetector                           │
│    • Input: temp/preprocessed_image.jpg          │
│    • Executa detect_objects()                    │
│    • Salva resultados (opcional)                 │
└────────────────┬─────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────┐
│ 4. Agregação de resultados                       │
│    • Combina timings                             │
│    • Retorna dicionário completo                 │
└──────────────────────────────────────────────────┘
```

**Comunicação com C++ via subprocess:**

```python
cmd = [
    str(self.preprocess_bin),      # bin/preprocess_optimized
    str(image_path),               # input.jpg
    str(preprocessed_path),        # temp/preprocessed_input.jpg
    filter_type                    # auto/blur/sharpen/etc
]

result = subprocess.run(
    cmd,
    capture_output=True,    # Captura stdout/stderr
    text=True,              # Decodifica como texto
    timeout=60              # Timeout de 60s
)

if result.returncode != 0:
    # Erro no preprocessamento
    error_message = result.stderr
```

**Mensagens de progresso:**

```
🔄 Step 1: Preprocessing with auto filter...
🔄 Step 2: YOLO object detection...
✅ Pipeline completed:
   Preprocessing: 456.7 ms
   Detection: 234.5 ms
   Total: 691.2 ms
   Objects detected: 5
   Results saved to: results/
```

#### 2. `benchmark_pipeline()`

Executa benchmark do pipeline completo.

```python
results = pipeline.benchmark_pipeline(
    image_path="image.jpg",
    num_runs=5
)
```

**Argumentos:**
- `image_path` (str): Imagem de teste
- `num_runs` (int): Número de execuções (padrão: 5)

**Retorno (Dict):**
```python
{
    "benchmark_results": {
        "num_runs": 5,
        "timing_stats": {
            "preprocessing": {
                "mean": 456.7,
                "std": 23.4,
                "min": 432.1,
                "max": 489.3
            },
            "detection": {
                "mean": 234.5,
                "std": 12.1,
                "min": 220.3,
                "max": 251.2
            },
            "total": {
                "mean": 691.2,
                "std": 28.9,
                "min": 665.4,
                "max": 725.8
            }
        },
        "detection_count": {
            "mean": 5.0,
            "std": 0.0
        }
    }
}
```

**Processo:**
1. **Warm-up:** 1 execução para estabilizar sistema
2. **Benchmark runs:** N execuções cronometradas
3. **Sem salvar resultados** (para não poluir disco)
4. **Estatísticas agregadas** usando NumPy

### Interface CLI de `pipeline_integration.py`

```bash
# Uso básico
python3 pipeline_integration.py <imagem> [opções]

# Opções
-f, --filter <tipo>         # Filtro de preprocessamento (default: auto)
-c, --confidence <float>    # Limiar YOLO (default: 0.25)
-d, --device <device>       # cpu, cuda, mps (default: cpu)
-b, --benchmark <N>         # Benchmark com N runs
--no-save                   # Não salvar resultados
```

**Exemplos:**

```bash
# Pipeline completo com filtro automático
python3 pipeline_integration.py image.jpg

# Com filtro específico
python3 pipeline_integration.py image.jpg -f sharpen

# Benchmark com 10 runs
python3 pipeline_integration.py image.jpg -b 10

# Sem salvar resultados
python3 pipeline_integration.py image.jpg --no-save

# Confiança customizada
python3 pipeline_integration.py image.jpg -c 0.5
```

**Saída exemplo:**

```
🔄 Step 1: Preprocessing with auto filter...
🔄 Step 2: YOLO object detection...
✅ Pipeline completed:
   Preprocessing: 456.7 ms
   Detection: 234.5 ms
   Total: 691.2 ms
   Objects detected: 5
   Results saved to: temp/detection_results
📄 Results saved to: temp/pipeline_results.json
```

**Arquivo `pipeline_results.json`:**

```json
{
  "input_image": "image.jpg",
  "filter_type": "auto",
  "timing": {
    "preprocessing": 456.7,
    "detection": 234.5,
    "total": 691.2
  },
  "preprocessing": {
    "success": true,
    "output_path": "temp/preprocessed_image.jpg"
  },
  "detection": {
    "image_path": "temp/preprocessed_image.jpg",
    "detections": [
      {
        "class": 0,
        "class_name": "person",
        "confidence": 0.89,
        "bbox": [120, 45, 250, 380],
        "center": [185, 212],
        "area": 43550
      }
    ],
    "detection_count": 5,
    "timing": {
      "inference": 220.3,
      "postprocess": 14.2,
      "total": 234.5
    }
  }
}
```

## 🤖 Modelo YOLO

### `yolov5su.pt`

**Características:**
- **Arquitetura:** YOLOv5 Small Ultralytics
- **Tamanho:** 18 MB (compacto)
- **Dataset:** COCO (80 classes)
- **Input size:** 640×640 pixels
- **Framework:** PyTorch

**Classes detectadas (COCO dataset - 80 classes):**

```python
classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
```

**Performance típica (CPU):**
- **Inferência:** 200-300 ms (imagens ~960×505)
- **FPS:** ~3-5 frames/segundo
- **mAP@0.5:** ~56.8% (COCO val)

**Por que YOLOv5 Small?**
- ✅ Rápido em CPU (sem GPU)
- ✅ Leve (18 MB vs 150 MB do YOLOv5x)
- ✅ Boa acurácia para uso geral
- ✅ Baixo uso de memória

## 📊 Fluxo de Dados Completo

### Pipeline End-to-End

```
┌─────────────────────────────────────────────────────────────────┐
│                      Usuário/Script                             │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
                  CVPipeline.run_full_pipeline()
                         ↓
         ┌───────────────┴───────────────┐
         ↓                               ↓
┌─────────────────┐              ┌──────────────────┐
│  Preprocessador │              │  YOLODetector    │
│      C++        │              │    (Python)      │
│  (subprocess)   │              │                  │
│                 │              │                  │
│ Input:          │              │ Input:           │
│  image.jpg      │              │  preprocessed.jpg│
│                 │              │                  │
│ Processo:       │              │ Processo:        │
│  • Análise      │              │  • Carregar img  │
│  • Filtro(s)    │──── temp/ ───>│  • Inferência   │
│  • OpenMP       │              │  • NMS           │
│                 │              │  • Pós-proc.     │
│ Output:         │              │                  │
│  preprocessed   │              │ Output:          │
│  .jpg           │              │  • detections[]  │
│                 │              │  • timing        │
└─────────────────┘              └──────────────────┘
         ↓                               ↓
         └───────────────┬───────────────┘
                         ↓
                 Resultados agregados
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Saídas:                                                        │
│  • temp/preprocessed_<nome>.jpg  - Imagem preprocessada         │
│  • temp/detected_<nome>.jpg      - Imagem com bounding boxes    │
│  • temp/<nome>.txt               - Detecções (formato YOLO)     │
│  • temp/pipeline_results.json    - Resultados completos (JSON)  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Dependências Python

### Bibliotecas Requeridas

```python
# Core
import subprocess          # Execução do binário C++
import time               # Medição de tempo
import json               # Serialização de resultados
from pathlib import Path  # Manipulação de caminhos
from typing import Dict, List, Optional, Tuple, Union  # Type hints

# Computação numérica
import numpy as np        # Arrays e estatísticas

# Visão computacional
import cv2                # OpenCV (leitura/escrita de imagens)
import torch              # PyTorch (backend do YOLO)
from ultralytics import YOLO  # Framework YOLO

# Argumentos CLI
import argparse           # Parser de argumentos
```

### Instalação

```bash
pip3 install -r requirements.txt
```

**requirements.txt:**
```
torch==2.8.0
torchvision==0.23.0
ultralytics==8.3.213
opencv-python-headless==4.12.0.88
numpy==2.2.6
```

## 🎯 Casos de Uso

### Caso 1: Detecção Simples (Sem Preprocessamento)

```python
from yolo_detector import YOLODetector

detector = YOLODetector(conf_thresh=0.3)
results = detector.detect_objects(
    "image.jpg",
    save_results=True,
    output_dir="results/"
)

print(f"Detectados: {results['detection_count']} objetos")
for det in results['detections']:
    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
```

### Caso 2: Pipeline Completo

```python
from pipeline_integration import CVPipeline

pipeline = CVPipeline()
results = pipeline.run_full_pipeline(
    image_path="image.jpg",
    filter_type="sharpen",
    confidence=0.25
)

print(f"Tempo total: {results['timing']['total']:.1f} ms")
print(f"Objetos: {results['detection']['detection_count']}")
```

### Caso 3: Benchmark de Performance

```python
from pipeline_integration import CVPipeline
import numpy as np

pipeline = CVPipeline()
results = pipeline.benchmark_pipeline("image.jpg", num_runs=10)

stats = results["benchmark_results"]["timing_stats"]
print(f"Preprocessing: {stats['preprocessing']['mean']:.1f} ms")
print(f"Detection: {stats['detection']['mean']:.1f} ms")
print(f"Total: {stats['total']['mean']:.1f} ms")
```

### Caso 4: Comparar Filtros

```python
from pipeline_integration import CVPipeline

filters = ["blur", "sharpen", "denoise", "clahe", "edge"]
pipeline = CVPipeline()

for f in filters:
    result = pipeline.run_full_pipeline(
        "image.jpg",
        filter_type=f,
        save_results=False
    )
    print(f"{f:10}: {result['detection']['detection_count']} objetos, "
          f"{result['timing']['total']:.1f} ms")
```

### Caso 5: Processar Múltiplas Imagens

```python
from pathlib import Path
from pipeline_integration import CVPipeline

pipeline = CVPipeline()
image_dir = Path("images/")

for img_path in image_dir.glob("*.jpg"):
    print(f"\nProcessando: {img_path.name}")
    results = pipeline.run_full_pipeline(
        str(img_path),
        save_results=True,
        output_dir=f"results/{img_path.stem}"
    )
```

## 📈 Performance Esperada

### Timings Típicos (CPU - Intel/AMD 8 cores)

| Componente        | Tempo (ms) | Percentual |
|-------------------|------------|------------|
| Preprocessing     | 400-800    | 60-70%     |
| YOLO Inference    | 200-300    | 25-35%     |
| Postprocessing    | 10-20      | 2-3%       |
| **Total**         | **650-1100** | **100%**   |

### Imagem de Teste: 960×505 pixels

**Com filtro blur:**
- Preprocessing: ~785 ms
- Detection: ~234 ms
- **Total: ~1019 ms**

**Com filtro sharpen:**
- Preprocessing: ~797 ms
- Detection: ~234 ms
- **Total: ~1031 ms**

**Com filtro denoise:**
- Preprocessing: ~3599 ms (mais intensivo!)
- Detection: ~234 ms
- **Total: ~3833 ms**

### Fatores que Afetam Performance

1. **Tamanho da imagem:**
   - 640×480: ~500-700 ms total
   - 1920×1080: ~1500-2500 ms total

2. **Complexidade do filtro:**
   - Edge Enhancement: ~609 ms (mais rápido)
   - CLAHE: ~636 ms
   - Blur: ~785 ms
   - Sharpen: ~797 ms
   - Denoise: ~3599 ms (mais lento)

3. **Número de threads OpenMP:**
   - 1 thread: baseline
   - 4 threads: ~1.3x speedup
   - 8 threads: ~1.4x speedup

4. **Dispositivo YOLO:**
   - CPU: ~230 ms
   - GPU (CUDA): ~50-80 ms
   - Apple Silicon (MPS): ~100-150 ms

## 🐛 Tratamento de Erros

### Erros Comuns e Soluções

#### 1. Erro: "YOLO model not available"

**Causa:** Biblioteca ultralytics não instalada

**Solução:**
```bash
pip3 install ultralytics
```

#### 2. Erro: "Preprocessing failed"

**Causa:** Binário C++ não compilado ou caminho incorreto

**Solução:**
```bash
# Compilar preprocessador
cd core_pipeline
make all

# Verificar se existe
ls -l bin/preprocess_optimized
```

#### 3. Erro: "Could not load image"

**Causa:** Caminho de imagem incorreto ou formato não suportado

**Solução:**
```python
from pathlib import Path
img_path = Path("image.jpg")
if not img_path.exists():
    print(f"Arquivo não encontrado: {img_path}")
```

#### 4. Erro: "Preprocessing timeout"

**Causa:** Imagem muito grande ou filtro muito lento

**Solução:**
- Redimensionar imagem antes
- Aumentar timeout em `pipeline_integration.py`:
```python
result = subprocess.run(cmd, timeout=120)  # 2 minutos
```

### Validação de Entrada

```python
def validate_input(image_path, filter_type, confidence):
    """Valida parâmetros de entrada"""

    # Verifica se imagem existe
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # Valida filtro
    valid_filters = ["auto", "blur", "sharpen", "denoise", "clahe", "edge"]
    if filter_type not in valid_filters:
        raise ValueError(f"Filtro inválido. Use: {valid_filters}")

    # Valida confiança
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confiança deve estar entre 0.0 e 1.0, got: {confidence}")
```

## 🧪 Testes

### Testar YOLODetector

```bash
# Detecção simples
python3 yolo_detector.py ../images/sample.jpg --save

# Benchmark
python3 yolo_detector.py ../images/sample.jpg -b 10

# Com GPU
python3 yolo_detector.py ../images/sample.jpg -d cuda --save
```

### Testar Pipeline Completo

```bash
# Pipeline com auto filter
python3 pipeline_integration.py ../images/sample.jpg

# Com filtro específico
python3 pipeline_integration.py ../images/sample.jpg -f sharpen

# Benchmark
python3 pipeline_integration.py ../images/sample.jpg -b 5
```

### Testes Automatizados

```python
def test_detector():
    """Testa YOLODetector"""
    detector = YOLODetector()
    result = detector.detect_objects("test_image.jpg")

    assert "error" not in result
    assert "detections" in result
    assert result["detection_count"] >= 0
    print("✅ YOLODetector test passed")

def test_pipeline():
    """Testa CVPipeline"""
    pipeline = CVPipeline()
    result = pipeline.run_full_pipeline("test_image.jpg", save_results=False)

    assert "error" not in result
    assert "timing" in result
    assert result["timing"]["total"] > 0
    print("✅ CVPipeline test passed")

if __name__ == "__main__":
    test_detector()
    test_pipeline()
```

## 📝 Estrutura de Arquivos Gerados

### Diretório `temp/`

```
temp/
├── preprocessed_image1.jpg      # Imagem preprocessada
├── preprocessed_image2.jpg
├── detection_results/           # Resultados de detecção
│   ├── detected_image1.jpg      # Imagem com bounding boxes
│   ├── image1.txt               # Detecções (formato YOLO)
│   ├── detected_image2.jpg
│   └── image2.txt
└── pipeline_results.json        # Resultados do pipeline
```

### Formato do Arquivo `.txt` (YOLO)

```
# Formato: class_id x_center y_center width height confidence
# Coordenadas normalizadas [0.0, 1.0]

0 0.512345 0.678901 0.234567 0.345678 0.891234
2 0.234567 0.123456 0.111111 0.222222 0.765432
16 0.789012 0.456789 0.098765 0.123456 0.678901
```

**Conversão para coordenadas absolutas:**
```python
img_width, img_height = 960, 505

class_id, x_norm, y_norm, w_norm, h_norm, conf = line.split()
x_norm, y_norm, w_norm, h_norm = map(float, [x_norm, y_norm, w_norm, h_norm])

x_center = x_norm * img_width
y_center = y_norm * img_height
width = w_norm * img_width
height = h_norm * img_height

x1 = int(x_center - width / 2)
y1 = int(y_center - height / 2)
x2 = int(x_center + width / 2)
y2 = int(y_center + height / 2)
```

## 🚀 Otimizações Futuras

### 1. Batch Processing

```python
def process_batch(self, image_paths: List[str]) -> List[Dict]:
    """Processa múltiplas imagens em lote"""
    results = []
    for img in image_paths:
        results.append(self.run_full_pipeline(img))
    return results
```

### 2. Async Processing

```python
import asyncio

async def async_pipeline(self, image_path: str):
    """Pipeline assíncrono"""
    # Preprocessing em subprocess (já é async)
    # Detection pode ser paralelizado
    pass
```

### 3. GPU Acceleration

```python
# Usar GPU para YOLO
detector = YOLODetector(device="cuda")

# Speedup esperado: 3-5x
# CPU: 230 ms → GPU: 50-80 ms
```

### 4. Caching de Modelo

```python
# Singleton para não recarregar modelo
_detector_cache = None

def get_detector():
    global _detector_cache
    if _detector_cache is None:
        _detector_cache = YOLODetector()
    return _detector_cache
```

## 💡 Dicas para Desenvolvedores

### 1. Debug de Subprocess

```python
# Ver stdout/stderr do preprocessador
result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
```

### 2. Profiling de Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Código a ser perfilado
pipeline.run_full_pipeline("image.jpg")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 funções
```

### 3. Logging Detalhado

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Iniciando preprocessamento...")
logger.info(f"Detecções: {count}")
logger.warning("Poucas detecções encontradas")
logger.error(f"Erro: {e}")
```

---

**Linguagem:** Python 3.8+
**Frameworks:** PyTorch, Ultralytics, OpenCV
**Licença:** Acadêmico/Pesquisa
