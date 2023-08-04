import streamlit as st
import pandas as pd
from joblib import load


# função que otimiza o carregamento dos dados da tabela
@st.cache_data
def carregar_tabela():
    tabela = pd.read_pickle("arquivos_pkl/dataframe_modelado.pkl")
    return tabela

# função que otimiza o carregamento dos dados do modelo
@st.cache_data
def carregar_modelo():
    modelo = load("arquivos_pkl/modelo_treinado.pkl")
    return modelo
