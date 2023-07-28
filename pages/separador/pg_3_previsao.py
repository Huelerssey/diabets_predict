import streamlit as st
import pandas as pd
import numpy as np
import joblib


# função que constroi a página 3
def prever_diabetes():
    st.markdown("<h1 style='text-align: center;'>📊 Modelo de Previsão de Diabetes 📊</h1>", unsafe_allow_html=True)

    # define colunas na pagina
    col1, col2 = st.columns(2)

    # coluna 1
    with col1:
        # Carregar o modelo treinado
        clf = joblib.load("modelo_treinado.pkl")

        # Função para fazer previsões
        def predict_diabetes(model, patient_info):
            prediction = model.predict(patient_info)
            return prediction

        # Mapeamento da coluna gender
        gender_mapping = {
            'Homem': 0,
            'Mulher': 1
        }
        gender = st.selectbox('Gênero', list(gender_mapping.keys()))
        gender = gender_mapping[gender]  # Converter para o código correspondente

        # idade
        age = st.number_input('Idade', min_value=0, max_value=100)

        # Mapeamento para as colunas hypertension e heart_disease
        binary_mapping = {
            'Não': 0,
            'Sim': 1
        }
        hypertension = st.selectbox('Hipertensão', list(binary_mapping.keys()))
        hypertension = binary_mapping[hypertension]  # Converter para o código correspondente
        heart_disease = st.selectbox('Doença Cardíaca', list(binary_mapping.keys()))
        heart_disease = binary_mapping[heart_disease]  # Converter para o código correspondente

        # Mapeamento da coluna smoking_history
        smoking_mapping = {
            'Nunca fumou': 3,
            'Fumante frequente': 0,
            'Ex-fumante': 2,
            'Pelo menos uma vez': 1,
            'Fumante ocasional': 4
        }
        smoking_history = st.selectbox('Histórico de Fumante', list(smoking_mapping.keys()))
        smoking_history = smoking_mapping[smoking_history]  # Converter para o código correspondente

        #IMC
        bmi = st.number_input('IMC', min_value=0.0)

        #nivel de hemoglobina glicada
        HbA1c_level = st.number_input('Nível de HbA1c', min_value=0.0)

        #nivel de glicose
        blood_glucose_level = st.number_input('Nível de Glicose no Sangue', min_value=0)

        # Botão para fazer previsão
        if st.button('Fazer previsão'):
            # Preparar os dados para a previsão
            patient_info = pd.DataFrame(np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]]),
                                        columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

            # Fazer a previsão
            prediction = predict_diabetes(clf, patient_info)
            if prediction[0] == 1:
                st.write('O paciente tem diabetes.')
            else:
                st.write('O paciente não tem diabetes.')