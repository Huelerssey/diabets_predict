import json
import streamlit as st
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components


def home():
    # animação
    with open("animacoes/pagina_inicial1.json") as source:
        animacao_1 = json.load(source)
    
    # exibir animação
    st_lottie(animacao_1, height=400, width=400)

    st.markdown("<h2 style='text-align: justfy;'>Seja bem vindo a apresentação do projeto de previsão de diabetes. Aqui você pode acompanhar todas as etapas de como o projeto foi desenvolvido, um modelo de previsão de acordo com as caracteristicas do paciente e minha apresentação formal de resultados.</h2>", unsafe_allow_html=True)
    st.write("")
