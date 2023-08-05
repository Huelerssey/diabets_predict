import streamlit as st
from src.data_utility import carregar_tabela_pkl
import altair as alt
import pandas as pd


# função que constroi a página 4
def dashboard():

    # variável que armazena a tabela em csv
    df = carregar_tabela_pkl()

    st.markdown("<h1 style='text-align: center;'>📋 Dashboard 📋</h1>", unsafe_allow_html=True)

    # Definir as faixas etárias
    bins = [0, 20, 40, 60, 80]
    labels = ['<=20', '21-40', '41-60', '61-80']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    # Calcular as médias
    mean_hba1c = df.groupby('age_group')['HbA1c_level'].mean().reset_index()
    mean_glucose = df.groupby('age_group')['blood_glucose_level'].mean().reset_index()

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

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # conteiner da segunda linha horizontal
    with st.container():

        # cria 2 colunas
        col1, col2 = st.columns([1, 2])
        
        with col1:

            # Criar uma nova coluna 'paciente' que transforma os valores 0 e 1 em categorias
            df['paciente'] = df['diabetes'].map({0: 'Não tem diabetes', 1: 'Tem diabetes'})

            # Contagem de valores de cada categoria
            diabetes_counts = df['paciente'].value_counts().reset_index()

           # Cria o gráfico de pizza
            chart = alt.Chart(diabetes_counts).mark_arc(innerRadius=0).encode(
                theta='count:Q',
                color='paciente:N',
                tooltip=['paciente:N', 'count:Q']
            ).properties(
                title='Distribuição de pacientes com/sem diabetes',
                width=400,
                height=300
            )

            # Exibe o gráfico Distribuição de pacientes com/sem diabetes
            st.altair_chart(chart)

        with col2:

            # Gráfico de linha para HbA1c
            hba1c_chart = alt.Chart(mean_hba1c).mark_line().encode(
                x='age_group:O',
                y='HbA1c_level:Q',
                tooltip=['age_group', 'HbA1c_level']
            ).properties(
                title='Média de HbA1c por faixa etária',
                width=900,
                height=300
            )
            st.altair_chart(hba1c_chart)

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # conteiner da terceira linha horizontal
    with st.container():
        
        #cria 2 colunas
        col1, col2 = st.columns([1, 2])

        with col1:

            # Criar uma nova coluna genero que transforma os valores 0 e 1 em categorias
            df['genero'] = df['gender'].map({0: 'Feminino', 1: 'Masculino'})

            # Contagem de valores de cada categoria
            gender_counts = df['genero'].value_counts().reset_index()

            # Renomear as colunas
            gender_counts.columns = ['genero', 'count']

            # Cria o gráfico de pizza
            chart = alt.Chart(gender_counts).mark_arc(innerRadius=0).encode(
                theta='count:Q',
                color='genero:N',
                tooltip=['genero:N', 'count:Q']
            ).properties(
                title='Distribuição de pacientes por gênero',
                width=400,
                height=300
            )

            # Exibe o gráfico Distribuição de pacientes por gênero
            st.altair_chart(chart)

        with col2:

            # Gráfico de linha para glicose no sangue
            glucose_chart = alt.Chart(mean_glucose).mark_line().encode(
                x='age_group:O',
                y='blood_glucose_level:Q',
                tooltip=['age_group', 'blood_glucose_level']
            ).properties(
                title='Média de glicose no sangue por faixa etária',
                width=900,
                height=300
            )
            st.altair_chart(glucose_chart)