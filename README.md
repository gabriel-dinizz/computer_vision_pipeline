# Pre-processamento de Imagens

Este projeto realiza o pré-processamento de imagens com C++ (OpenCV + OpenMP) e integra o YOLOv5 como submódulo para tarefas de detecção.

## Estrutura de Pastas

- `src/preprocess.cpp`: Código-fonte principal em C++. Aplica blur gaussiano em imagens, processando cada linha em paralelo.
- `bin/preprocess`: Binário gerado após compilação do código-fonte C++.
- `images/`: Pasta para imagens de entrada e saída.
    - Exemplos: `input.jpg`, `output.jpg`, `panda.jpg`
- `data/`: Pasta reservada para dados auxiliares.
- `logs/`: Pasta reservada para logs.
- `Makefile`: Script para compilar o projeto C++.
- `requirements.txt`: Dependências Python (usadas principalmente pelo submódulo YOLOv5).
- `external/`: Submódulo contendo o código do YOLOv5 para detecção de objetos.
    - `external/yolov5/`: Repositório oficial do YOLOv5.
- `.gitignore`: Arquivos e pastas ignorados pelo git.

## Como Compilar o Pré-processador C++

Certifique-se de ter o OpenCV e o OpenMP instalados. Para compilar:

```sh
make
```

O binário será gerado em `bin/preprocess`.

## Como Executar o Pré-processador

Execute o binário passando o caminho da imagem de entrada e o caminho para salvar a imagem de saída:

```sh
./bin/preprocess images/input.jpg images/output.jpg
```

## Funcionalidades do Pré-processador

- Aplica blur gaussiano em imagens coloridas.
- Processamento paralelo por linha usando OpenMP.
- Mensagens de erro para arquivos não encontrados ou falha ao salvar.
- Informa o número de threads utilizados.

## Integração com YOLOv5

O submódulo `external/yolov5` permite realizar detecção de objetos em imagens já pré-processadas.  
Para usar o YOLOv5, siga as instruções do próprio repositório em `external/yolov5/README.md`.

## Requisitos

### Sistema

- macOS (testado)
- OpenCV
- OpenMP
- gcc (opcional, para suporte nativo ao OpenMP)
- Ferramentas de build: `cmake`, `pkg-config`, `make`

Instalação recomendada via Homebrew:

```sh
brew install cmake pkg-config make
brew install opencv libomp
brew install gcc # opcional
```

### Python (para YOLOv5)

Veja `requirements.txt` para dependências Python. Instale com:

```sh
pip install -r requirements.txt
```

## Caso de Uso do Algoritmo Paralelizado

O pré-processamento paralelo é útil para acelerar pipelines de machine learning, processamento em lote de imagens industriais, ou qualquer cenário que exija manipulação rápida de grandes volumes de dados visuais.

## Observações

- O submódulo YOLOv5 deve ser inicializado/clonado com:
  ```sh
  git submodule update --init --recursive
  ```
- Para integração entre C++ e Python, pode-se salvar imagens pré-processadas e processá-las posteriormente com YOLOv5.
