import streamlit as st
from src.data_utility import carregar_tabela_pkl
import altair as alt
import pandas as pd
import plotly.express as px
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards


# função que constroi a página 4
def dashboard():

    # variável que armazena a tabela em csv
    df = carregar_tabela_pkl()

    # titulo 
    st.markdown("<h1 style='text-align: center;'>📋 Dashboard 📋</h1>", unsafe_allow_html=True)
    
    # marcador azul
    colored_header(
    label="",
    description="",
    color_name="light-blue-70"
    )

    # Definir as faixas etárias
    bins = [0, 20, 40, 60, 80]
    labels = ['0-20', '21-40', '41-60', '61-80']
    df['idade'] = pd.cut(df['age'], bins=bins, labels=labels)

    # Calcular as médias
    # Renomear as colunas
    df = df.rename(columns={
        'HbA1c_level': 'hemoglobina_glicada',
        'blood_glucose_level': 'nivel_glicose_sangue'
    })
    mean_hba1c = df.groupby('idade')['hemoglobina_glicada'].mean().reset_index()
    mean_glucose = df.groupby('idade')['nivel_glicose_sangue'].mean().reset_index()

    # conteiner dos KPI's
    with st.container():

        # cria 3 colunas
        col1, col2, col3 = st.columns(3)

        # KPI total de pacientes da amostra
        #total_patients = len(df)
        col1.metric("Total de Pacientes válidos", value='63.247', delta='100 mil no dataset original', delta_color='inverse')

        # KPI idade mais afetada
        #velho = len(df[(df['age'] > 40) & (df['age'] <= 60)])
        col2.metric(label="Pacientes de 40 a 60 anos", value='21.646', delta="Faixa etária mais afetada")

        # KPI fator mais influente para o diagnóstico
        correlation = df['nivel_glicose_sangue'].corr(df['diabetes'])
        correlation_formated = correlation * 100
        col3.metric(label="Correlação Glicose - Diabetes", value=f"{correlation_formated:.2f}%", delta="fator mais influente para o diagnóstico")

        style_metric_cards(
            background_color='#000000',
            border_color='#FFFFFF',
            border_left_color='#0D98E2'
        )

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # conteiner da segunda linha horizontal
    with st.container():

        # cria 2 colunas
        col1, col2 = st.columns([1, 2])
        
        with col1:
            
            st.write("")
            st.write("")
            # Criar uma nova coluna 'paciente' que transforma os valores 0 e 1 em categorias
            df['paciente'] = df['diabetes'].map({0: 'Com diabetes', 1: 'Sem diabetes'})

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
                height=400
            )

            # Exibe o gráfico Distribuição de pacientes com/sem diabetes
            st.altair_chart(chart)

        with col2:
            
            # Gráfico de HbA1c
            fig_hba1c = px.line(mean_hba1c, x='idade', y='hemoglobina_glicada', title='Média de HbA1c por faixa etária')
            fig_hba1c.update_layout(title_font=dict(size=15), width=930, height=400)
            st.plotly_chart(fig_hba1c)

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # conteiner da terceira linha horizontal
    with st.container():
        
        #cria 2 colunas
        col1, col2 = st.columns([1, 2])

        with col1:
            
            st.write("")
            st.write("")
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
                height=400
            )

            # Exibe o gráfico Distribuição de pacientes por gênero
            st.altair_chart(chart)

        with col2:

            # Gráfico de glicose no sangue
            fig_glucose = px.line(mean_glucose, x='idade', y='nivel_glicose_sangue', title='Média de glicose no sangue por faixa etária')
            fig_glucose.update_layout(title_font=dict(size=15), width=930, height=400)
            st.plotly_chart(fig_glucose)
    
    # marcador azul
    colored_header(
    label="",
    description="",
    color_name="light-blue-70"
    )
