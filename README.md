# Pre-processamento

Este projeto realiza o pré-processamento de imagens utilizando OpenCV e paralelização com OpenMP.

## Estrutura de Pastas

- `src/preprocess.cpp`: Código-fonte principal. Aplica blur gaussiano em imagens, processando cada linha em paralelo.
- `bin/preprocess`: Binário gerado após compilação do código-fonte. Executa o pré-processamento.
- `images/`: Pasta para imagens de entrada e saída.
- `input.jpg`: Exemplo de imagem de entrada.
- `output.jpg`: Exemplo de imagem de saída gerada pelo programa.
- `panda.jpg`: Outro exemplo de imagem para teste.
- `data/`: Pasta reservada para dados auxiliares (atualmente vazia).
- `logs/`: Pasta reservada para logs (atualmente vazia).
- `Makefile`: Script para compilar o projeto.

## Como Compilar

Certifique-se de ter o OpenCV e o OpenMP instalados. Para compilar, execute:

```sh
make
```

O binário será gerado em `bin/preprocess`.

## Como Executar

Execute o binário passando o caminho da imagem de entrada e o caminho para salvar a imagem de saída:

```sh
./bin/preprocess images/input.jpg images/output.jpg
```

## Funcionalidades

- Aplica blur gaussiano em imagens coloridas.
- Processamento paralelo por linha usando OpenMP.
- Mensagens de erro para arquivos não encontrados ou falha ao salvar.
- Informa o número de threads utilizados.

## Requisitos


## Autor

Gabriel Diniz

## Caso de Uso do Algoritmo Paralelizado



