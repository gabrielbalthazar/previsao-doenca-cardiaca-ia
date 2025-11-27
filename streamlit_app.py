import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('./archive/train.csv')

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang']
LABEL = 'target'

X = df[FEATURES]
y = df[LABEL]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

hd_model = RandomForestRegressor(random_state=1)
hd_model.fit(X_train, y_train)

val_predictions = hd_model.predict(X_val)
mae = mean_absolute_error(y_val, val_predictions)

def get_prediction_result(probability):
    if probability >= 0.5:
        return "PREVISÃO DE PRESENÇA DE DOENÇA CARDÍACA", "error"
    else:
        return "PREVISÃO DE AUSÊNCIA DE DOENÇA CARDÍACA (Previsão: Negativa)", "success"

st.set_page_config(
    page_title="Predição de Doença Cardíaca",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Predição de Doença Cardíaca")
st.markdown("---")

st.header("Entrada de Dados do Paciente")
st.markdown("Insira os parâmetros clínicos para obter a predição de doença cardíaca.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Idade (anos)", min_value=20, max_value=100, value=50)

    sex_option = st.radio("Gênero", options=['Masculino', 'Feminino'], index=0)
    sex = 1 if 'Masculino' in sex_option else 0

    cp = st.selectbox("Tipo de Dor no Peito (cp)",
                      options=[
                          (0, "Assintomática"),
                          (1, "Angina Típica"),
                          (2, "Angina Atípica"),
                          (3, "Dor Não-Anginosa")
                      ],
                      format_func=lambda x: x[1])
    cp = cp[0]

with col2:
    trestbps = st.slider("Pressão Arterial em Repouso (trestbps - mm Hg)", min_value=80, max_value=200, value=120)

    thalach = st.slider("Frequência Cardíaca Máx. (thalach - batimentos/min)", min_value=60, max_value=220, value=150)

    exang_option = st.radio("Angina Induzida por Exercício (exang)", options=['Sim', 'Não'], index=1)
    exang = 1 if 'Sim' in exang_option else 0

st.markdown("---")
if st.button("Executar Predição", type="primary"):

    new_data = pd.DataFrame([[age, sex, cp, trestbps, thalach, exang]], columns=FEATURES)
    prediction_prob = hd_model.predict(new_data)[0]
    result_text, result_type = get_prediction_result(prediction_prob)
    st.subheader("Resultados da Predição")

    if result_type == "error":
        st.error(f"### {result_text}")
    else:
        st.success(f"### {result_text}")

    st.info(f"O modelo previu uma **probabilidade de {prediction_prob * 100:.2f}%** para a presença da doença cardíaca (sendo 50% o ponto de corte).")

    st.markdown("---")
    st.subheader("Conclusões e Interpretações")
    st.markdown(f"""
        Esta predição é baseada em seis características clínicas. É importante entender a contribuição dos fatores:

        * **Probabilidade ({prediction_prob:.2f}):** Quanto mais próximo de 1.0, mais forte é a indicação de que as características inseridas se assemelham ao perfil de pacientes com doença cardíaca no nosso conjunto de dados de treinamento.
        * **Idade e Frequência Cardíaca:** Geralmente, idades avançadas e uma capacidade de atingir uma frequência cardíaca máxima mais baixa estão associadas a maior risco.
        * **Tipo de Dor no Peito:** Se Assintomática, ou seja, sem dor no peito, frequentemente indica uma maior probabilidade de doença cardíaca significativa, enquanto as opções 1, 2 e 3 tendem a ser menos graves.
        * **Angina por Exercício:** A presença de angina ao se exercitar é um forte preditor de doença cardíaca.
    """)
