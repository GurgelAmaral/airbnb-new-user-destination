# Airbnb_model.v1

Pipeline de classificação com dados do Airbnb provenientes da competição da plataforma Kaggle, desenvolvido com foco em modularidade e reprodutibilidade. O projeto abrange pré-processamento, tuning de hiperparâmetros, avaliação e exportação do modelo final.

## Visão Geral

- Com o auxílio da biblioteca Pandas, foi feita uma limpeza e transformação dos dados em notebooks colab. Valores NaN foram reparados, datatypes incongruentes foram corrigidos, colunas irrelevantes foram eliminadas.

- **Pré-processamento**: codificação ordinal automática de variáveis categóricas.
- **Modelo**: Random Forest com ajuste via GridSearchCV.
- **Métrica**: Acurácia.
- **Exportação**: Modelo final salvo com `joblib`.

## Execução

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Certifique-se de que os seguintes arquivos CSV estejam no diretório raiz:

   - `x_train_resampled.csv`
   - `y_train_resampled.csv`
   - `x_test_resampled.csv`
   - `y_test_resampled.csv`

3. Execute o pipeline:

   ```bash
   python main.py
   ```

**Observação**: conforme determinação da plataforma de origem, os arquivos de dataset não podem ser disponibilizados neste repositório.

## Estrutura

- `main.py`: execução do pipeline completo.
- `pipeline.py`: definição do pipeline de modelagem.
- `tuning.py`: ajuste de hiperparâmetros com GridSearch.
- `evaluate.py`: função de avaliação do modelo.
- `features.py`: extração de colunas categóricas.

## Requisitos

- Python 3.8+
- scikit-learn
- pandas
- joblib

