import streamlit as st
import pandas as pd
import numpy as np
from src.data_utility import carregar_modelo


# função que constroi a página 3
def prever_diabetes():
    st.markdown("<h1 style='text-align: center;'>📊 Modelo de Previsão de Diabetes 📊</h1>", unsafe_allow_html=True)

    # define colunas na pagina
    col1, col2 = st.columns(2)

    # coluna 1
    with col1:
        # Carregar o modelo treinado
        clf = carregar_modelo()

        # Função para fazer previsões
        def predict_diabetes(model, patient_info):
            prediction = model.predict(patient_info)
            prediction_proba = model.predict_proba(patient_info)[0, int(prediction[0])]
            return prediction, prediction_proba

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
            prediction, prediction_proba = predict_diabetes(clf, patient_info)
            if prediction[0] == 1:
                st.success(f'O modelo classificou o paciente como diabético com {prediction_proba*100:.2f}% de chance de acerto.')
            else:
                st.success(f'O modelo classificou o paciente como não diabético com {prediction_proba*100:.2f}% de chance de acerto.')
    
    # Coluna 2
    with col2:

        # Tabela de referência de IMC
        st.markdown("<h6 style='text-align: center;'>Tabela de Referência de IMC</h6>", unsafe_allow_html=True)
        imc_table = pd.DataFrame({
            'Categoria': ['Abaixo do peso', 'Peso normal', 'Sobrepeso', 'Obesidade Grau I', 'Obesidade Grau II', 'Obesidade Grau III'],
            'IMC': ['< 18.5', '18.5 - 24.9', '25 - 29.9', '30 - 34.9', '35 - 39.9', '≥ 40']
        })
        st.table(imc_table)

        # Tabela de referência de HbA1c
        st.markdown("<h6 style='text-align: center;'>Tabela de Referência de HbA1c</h6>", unsafe_allow_html=True)
        hba1c_table = pd.DataFrame({
            'Categoria': ['Normal', 'Pré-diabetes', 'Diabetes'],
            'HbA1c (%)': ['< 5.7', '5.7 - 6.4', '≥ 6.5']
        })
        st.table(hba1c_table)

        # Tabela de referência de glicose no sangue
        st.markdown("<h6 style='text-align: center;'>Tabela de Referência de Glicose no Sangue</h6>", unsafe_allow_html=True)
        glucose_table = pd.DataFrame({
            'Categoria': ['Normal', 'Pré-diabetes', 'Diabetes'],
            'Glicemia de Jejum (mg/dL)': ['< 100', '100 - 125', '≥ 126']
        })
        st.table(glucose_table)

    # cria 3 colunas
    col1, col2, col3 = st.columns(3)

    with col2:
        if st.button('⚠️ Clique aqui para um aviso importante ⚠️'):
            st.warning("""
            **As tabelas de referência para IMC, HbA1c e níveis de glicose no sangue fornecidas aqui são apenas para fins informativos e não devem ser usadas para autodiagnóstico ou para substituir o aconselhamento médico profissional.** 
            """)

    # # Tabela de referência de IMC
    # st.markdown("<h3 style='text-align: center;'>Tabela de Referência de IMC</h3>", unsafe_allow_html=True)
    # imc_table = pd.DataFrame({
    #     'Categoria': ['Abaixo do peso', 'Peso normal', 'Sobrepeso', 'Obesidade Grau I', 'Obesidade Grau II', 'Obesidade Grau III'],
    #     'IMC': ['< 18.5', '18.5 - 24.9', '25 - 29.9', '30 - 34.9', '35 - 39.9', '≥ 40']
    # })
    # st.table(imc_table)

    st.write("---")

    #footer
    with st.container():
        col1, col2, col3 = st.columns(3)

        col2.write("Developed By: [@Huelerssey](https://github.com/Huelerssey)")
