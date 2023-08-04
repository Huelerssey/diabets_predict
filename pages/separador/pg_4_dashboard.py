import streamlit as st
from src.data_utility import carregar_tabela


# função que constroi a página 4
def dashboard():
    df = carregar_tabela()
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

    # Count the number of patients with and without diabetes
    diabetes_counts = df['diabetes'].value_counts()

    # Plot a bar chart
    st.bar_chart(diabetes_counts)

    # Filter the data for patients with and without diabetes
    df_with_diabetes = df[df['diabetes'] == 1]
    df_without_diabetes = df[df['diabetes'] == 0]

    # Group the data by age and calculate the mean blood glucose level
    df_with_diabetes_grouped = df_with_diabetes.groupby('age')['blood_glucose_level'].mean()
    df_without_diabetes_grouped = df_without_diabetes.groupby('age')['blood_glucose_level'].mean()

    # Plot a line chart
    st.line_chart(df_with_diabetes_grouped)
    st.line_chart(df_without_diabetes_grouped)


