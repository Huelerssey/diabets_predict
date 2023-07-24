import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pages.separador.pagina_1 as PaginaUm
import pages.separador.pagina_2 as PaginaDois
import pages.separador.pagina_3 as PaginaTres
import pages.separador.pagina_4 as PaginaQuatro


# configurações da pagina
st.set_page_config(
    page_title='MultiPagesAPP',
    #https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app
    page_icon='✅',
    layout='wide'
)

#aplicar estilos de css a pagina
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Menu de navegação lateral
with st.sidebar:
    opcao_selecionada = option_menu(
        menu_title="Menu Inicial",
        options=["Página 1", "Página 2", "Página 3", "Página 4"],
        #https://icons.getbootstrap.com
        icons=['bookmark', 'bookmark', 'bookmark', 'bookmark'],
        default_index=0,
        orientation='vertical',
    )

# Retorna a pagina 1
if opcao_selecionada == "Página 1":
    PaginaUm.pagina1()

# Retorna a pagina 2
elif opcao_selecionada == "Página 2":
    PaginaDois.pagina2()

# Retorna a pagina 3
elif opcao_selecionada == "Página 3":
    PaginaTres.pagina3()

# Retorna a pagina 3
elif opcao_selecionada == "Página 4":
    PaginaQuatro.pagina4()

