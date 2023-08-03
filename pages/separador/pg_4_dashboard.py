import streamlit as st
from src.data_utility import carregar_dados


# função que constroi a página 4
def dashboard():
    st.markdown("<h1 style='text-align: center;'>📋 Dashboard 📋</h1>", unsafe_allow_html=True)

    st.table(carregar_dados().tail())