import streamlit as st
from src.data_utility import carregar_tabela_pkl
import plost

# função que constroi a página 4
def dashboard():

    # variável que armazena a tabela em csv
    df = carregar_tabela_pkl()

    st.markdown("<h1 style='text-align: center;'>📋 Dashboard 📋</h1>", unsafe_allow_html=True)

    # conteiner dos KPI's
    with st.container():
        st.write("---")
        # cria 3 colunas
        col1, col2, col3 = st.columns(3)

        # KPI total de pacientes da amostra
        #total_patients = len(df)
        col1.metric("Total de Pacientes válidos", value='63.247', delta='100 mil no dataset original', delta_color='inverse')

        # KPI idade mais afetada
        #velho = len(df[(df['age'] > 40) & (df['age'] <= 60)])
        col2.metric(label="Pacientes de 40 a 60 anos", value='21.646', delta="Faixa etária mais afetada")

        # KPI fator mais influente para o diagnóstico
        correlation = df['blood_glucose_level'].corr(df['diabetes'])
        correlation_formated = correlation * 100
        col3.metric(label="Correlação Glicose - Diabetes", value=f"{correlation_formated:.2f}%", delta="fator mais influente para o diagnóstico")
        st.write("---")

    # conteiner dos gráficos de pizza
    with st.container():

        # cria 3 colunas
        col1, col2 = st.columns([1, 3])

        with col1:

            # Criar uma nova coluna 'diabetes_status' que transforma os valores 0 e 1 em categorias
            df['diabetes_status'] = df['diabetes'].map({0: 'Não tem diabetes', 1: 'Tem diabetes'})

            # Contagem de valores de cada categoria
            diabetes_counts = df['diabetes_status'].value_counts().reset_index()

            # Renomear as colunas
            diabetes_counts.columns = ['diabetes_status', 'count']

            # Construir o gráfico de pizza
            plost.pie_chart(
                data=diabetes_counts,
                theta='count',
                color='diabetes_status',
                title="Distribuição de pacientes com/sem diabetes",
                width=400,
                height=400,
                use_container_width=False
            )

        with col2:

            # Criar uma nova coluna 'gender_status' que transforma os valores 0 e 1 em 'Homem' e 'Mulher'
            df['gender_status'] = df['gender'].map({0: 'Homem', 1: 'Mulher'})

            # Contagem de valores de cada categoria
            gender_counts = df['gender_status'].value_counts().reset_index()

            # Renomear as colunas
            gender_counts.columns = ['gender_status', 'count']

            # Construir o gráfico de rosca
            plost.pie_chart(
                data=gender_counts,
                theta='count',
                color='gender_status',
                title="Distribuição de gênero dos pacientes",
                width=400,
                height=400,
                use_container_width=False
            )


