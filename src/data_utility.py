import streamlit as st
import pandas as pd


# função que otimiza o carregamento dos dados
@st.cache_data
def carregar_dados():
    tabela = pd.read_pickle("arquivos_pkl/dataframe_modelado.pkl")
    return tabela