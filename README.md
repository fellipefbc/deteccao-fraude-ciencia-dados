# Trabalho Final - Detecção de Fraude em Cartão de Crédito

## Disciplina: Ciência de Dados - Uniube

Link Apresentação: https://youtu.be/hkPiYm87xSg

Este projeto implementa um sistema de detecção de fraude em transações de cartão de crédito usando Python, Pandas, Scikit-learn, XGBoost e Streamlit.

### Como Executar

1.  **Clone e configure o ambiente:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd trabalho-final-ciencia-dados
    python -m venv venv
    source venv/bin/activate  # ou .\venv\Scripts\activate no Windows
    pip install -r requirements.txt
    ```

2.  **Baixe o Dataset:** Baixe o [dataset do Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), extraia o `creditcard.csv` e coloque-o na pasta `data/`.

3.  **Treine o Modelo:** Execute o notebook para gerar os modelos.
    ```bash
    jupyter notebook notebooks/01_Analise_e_Modelagem.ipynb
    ```
    *Isso criará a pasta `models/` com os arquivos `fraud_detection_model.pkl` e `scaler.pkl`.*

4.  **Execute a Aplicação:**
    ```bash
    streamlit run app/app.py
    ```
