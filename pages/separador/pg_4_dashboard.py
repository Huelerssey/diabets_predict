import streamlit as st
from src.data_utility import carregar_tabela


# função que constroi a página 4
def dashboard():
    df = carregar_tabela()
    st.markdown("<h1 style='text-align: center;'>📋 Dashboard 📋</h1>", unsafe_allow_html=True)

    st.table(df.tail())

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


