# Projeto TCC Data Science and Analytics

Este projeto foi desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) em Data Science and Analytics. Ele tem como objetivo realizar análises de dados relacionadas à depressão, utilizando técnicas de aprendizado de máquina, estatística e visualização de dados.

## Estrutura do Projeto

A estrutura do projeto está organizada em diferentes diretórios e arquivos para facilitar a modularidade e a manutenção do código:

### Diretórios Principais

- **`data/files`**: Contém os arquivos de entrada e saída de dados, como `out_dados.csv` e `dados_iniciais.csv`.
- **`processing`**: Contém os módulos responsáveis pelo processamento de dados e análises, como:
  - `Data`: Manipulação e preparação de dados.
  - `DepressionAnalysis`: Treinamento e avaliação de modelos relacionados à análise de depressão.
- **`utils`**: Contém utilitários para manipulação de arquivos, divisão de dados, normalização e outras funções auxiliares.
- **`model_statistics`**: Contém análises estatísticas e geração de relatórios, como a matriz de correlação e curvas de aprendizado.

### Arquivos Importantes

- **`main.py`**: Arquivo principal para execução do projeto. Ele permite selecionar diferentes etapas do pipeline, como preparação de dados, treinamento de modelos e análise de resultados.
- **`data/files/out_dados.csv`**: Arquivo de dados processados.
- **`data/files/dados_iniciais.csv`**: Arquivo de dados iniciais para análise.
- **`processing/Data/DataTable.py`**: Classe responsável pela manipulação e preparação dos dados.
- **`processing/DepressionAnalysis/TrainingDepression.py`**: Classe responsável pelo treinamento e avaliação de modelos de aprendizado de máquina.
- **`utils/FileUtils.py`**: Classe utilitária para leitura e escrita de arquivos.

## Funcionalidades

### 1. Preparação de Dados
- Manipulação e limpeza de dados.
- Geração de variáveis dummy.
- Escrita de tabelas processadas em arquivos CSV.

### 2. Treinamento de Modelos
- Treinamento de modelos de aprendizado de máquina para análise de depressão.
- Validação cruzada e cálculo de métricas como acurácia, R² e matriz de confusão.
- Geração de curvas de aprendizado.

### 3. Análise Estatística
- Geração de matrizes de correlação.
- Visualização de dados com gráficos, como heatmaps.

### 4. Geração de Relatórios
- Exportação de resultados em arquivos CSV.
- Geração de relatórios detalhados com métricas e gráficos.

## Como Executar

1. **Instale as dependências**:
   Certifique-se de que você tem o Python instalado e instale as bibliotecas necessárias:
   ```bash
   pip install -r requirements.txt