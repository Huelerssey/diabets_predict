import json
import streamlit as st
from streamlit_lottie import st_lottie


# função que constroi a página 1
def home():

    # Colunas que organizam a página
    col1, col2 = st.columns(2)

    # animações
    with open("animacoes/pagina_inicial1.json") as source:
        animacao_1 = json.load(source)
    with open("animacoes/pagina_inicial2.json") as source:
        animacao_2 = json.load(source)
    
    # conteúdo a ser exibido na coluna 1
    with col1:
        st_lottie(animacao_1, height=350, width=400)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("<h5 style='text-align: justfy;'> Ao longo do projeto, vou compartilhar os resultados formais, destacando o desempenho e a precisão do modelo, bem como sua aplicabilidade prática no auxílio ao diagnóstico dessa condição. Prepare-se para embarcar em uma jornada fascinante pelo mundo da ciência de dados e inteligência artificial!</h5>", unsafe_allow_html=True)

    # conteúdo a ser exibido na coluna 2
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("<h5 style='text-align: justfy;'> Seja muito bem-vindo ao projeto de Previsão de Diabetes! Aqui, você terá a oportunidade de acompanhar todas as etapas envolvidas no desenvolvimento deste projeto, desde a concepção até a criação de um modelo de previsão personalizado, adaptado especificamente às características únicas de cada paciente.</h5>", unsafe_allow_html=True)
        st_lottie(animacao_2, height=400, width=440)
