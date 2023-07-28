import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import pages.separador.pg_1_home as PaginaInicial
import pages.separador.pg_2_projeto as ConstrucaoProjeto
import pages.separador.pg_3_previsao as PreverDiabetes
import pages.separador.pg_4_apresentacao as ApresentacaoProjeto


# configurações da pagina
st.set_page_config(
    page_title='Prever Diabetes',
    #https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app
    page_icon='⚕️',
    layout='wide'
)

#aplicar estilos de css a pagina
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# animações
with open("animacoes/pagina_inicial3.json") as source:
    animacao_3 = json.load(source)

# Menu de navegação lateral
with st.sidebar:
    st_lottie(animacao_3, height=100, width=270)
    st.write("---")
    opcao_selecionada = option_menu(
        menu_title="Menu Principal",
        menu_icon="justify",
        options=["Inicio", "Projeto", "Previsão", "Apresentação"],
        #https://icons.getbootstrap.com
        icons=['house', 'pin-angle', 'clipboard-data', 'journal-medical'],
        default_index=0,
        orientation='vertical',
    )
    

    st.write("---")
    st.markdown("<h5 style='text-align: center; color: lightgray;'>Developed By: Huelerssey Rodrigues</h5>", unsafe_allow_html=True)

# Retorna a pagina 1
if opcao_selecionada == "Inicio":
    PaginaInicial.home()

# Retorna a pagina 2
elif opcao_selecionada == "Projeto":
    ConstrucaoProjeto.construcao_projeto()

# Retorna a pagina 3
elif opcao_selecionada == "Previsão":
    PreverDiabetes.prever_diabetes()

# Retorna a pagina 3
elif opcao_selecionada == "Apresentação":
    ApresentacaoProjeto.apresentacao()

