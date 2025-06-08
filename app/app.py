import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Título da Aplicação
st.set_page_config(page_title="Detecção de Fraude", layout="wide")
st.title("Sistema de Detecção de Fraude em Cartão de Crédito")


# --- Carregamento do Modelo e Scaler ---
# Adiciona um cache para não recarregar os modelos a cada interação
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/fraud_detection_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None


model, scaler = load_model()

# Verifica se os modelos foram carregados
if model is None or scaler is None:
    st.error("ERRO: Arquivos de modelo não encontrados!")
    st.warning(
        "Por favor, execute o notebook 'notebooks/01_Analise_e_Modelagem.ipynb' primeiro para treinar e salvar os artefatos do modelo na pasta 'models/'.")
    st.stop()

st.success("Modelo de detecção carregado com sucesso!")
st.write(
    "Esta aplicação interativa utiliza um modelo de Machine Learning (XGBoost) para prever se uma transação "
    "financeira é legítima ou fraudulenta. Insira os dados da transação na barra lateral para fazer uma previsão em tempo real."
)

# --- Barra Lateral para Entrada do Usuário ---
st.sidebar.header("Insira os Dados da Transação")


def user_input_features():
    v_features = {}
    st.sidebar.markdown("As features V1-V28 são anônimas (resultado de PCA).")
    # Criar sliders para as features mais importantes (baseado no notebook)
    important_features = ['V14', 'V12', 'V10', 'V17', 'V11', 'V4', 'V3', 'V16']
    with st.sidebar.expander("Features Mais Relevantes para Fraude"):
        for i in important_features:
            v_features[i] = st.slider(f'Feature {i}', -25.0, 15.0, 0.0, 0.1)

    with st.sidebar.expander("Outras Features (V1-V28)"):
        all_v = [f'V{i}' for i in range(1, 29)]
        for feature_name in all_v:
            if feature_name not in v_features:
                v_features[feature_name] = st.slider(f'Feature {feature_name}', -10.0, 10.0, 0.0, 0.1)

    time = st.sidebar.number_input('Tempo (segundos desde a primeira transação)', min_value=0, value=1000)
    amount = st.sidebar.number_input('Valor da Transação (Amount)', min_value=0.0, value=50.0, format="%.2f")

    data = {**v_features, 'Time': time, 'Amount': amount}
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# --- Exibição e Predição ---
st.subheader("Dados da Transação Inseridos")
st.write(input_df)

if st.sidebar.button("Verificar Transação", type="primary"):
    # 1. Normaliza as colunas Time e Amount
    scaled_features = scaler.transform(input_df[['Time', 'Amount']])

    # 2. Cria uma cópia do dataframe de entrada, removendo as colunas originais
    final_df = input_df.drop(['Time', 'Amount'], axis=1)

    # 3. Adiciona as colunas normalizadas com os nomes CORRETOS que o modelo espera
    final_df['scaled_Time'] = scaled_features[0, 0]
    final_df['scaled_Amount'] = scaled_features[0, 1]

    # 4. Garante que a ordem das colunas é EXATAMENTE a mesma do treinamento
    #    (O modelo é sensível à ordem das colunas)
    feature_order_from_training = [f'V{i}' for i in range(1, 29)] + ['scaled_Amount', 'scaled_Time']
    final_df = final_df[feature_order_from_training]

    # Predição
    st.write("---")  # Linha divisória para clareza
    st.subheader("Resultado da Análise")
    prediction = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)

    st.subheader("Resultado da Análise")

    if prediction[0] == 1:
        st.error("ALERTA: Transação Classificada como FRAUDULENTA!", icon="🚨")
    else:
        st.success("Transação Classificada como LEGÍTIMA.", icon="✅")

    st.write("Probabilidades:")
    prob_df = pd.DataFrame({
        'Classe': ['Legítima', 'Fraude'],
        'Probabilidade': prediction_proba[0]
    })
    st.dataframe(prob_df.style.format({'Probabilidade': "{:.2%}"}), use_container_width=True)