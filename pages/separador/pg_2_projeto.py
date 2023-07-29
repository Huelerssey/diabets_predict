import streamlit as st
import pandas as pd


# função que otimiza o carregamento dos dados
@st.cache_data
def carregar_dados():
    tabela = pd.read_csv("dataset/diabetes_prediction_dataset.csv")
    return tabela

# função que constroi a página 2
def construcao_projeto():
    st.markdown("<h1 style='text-align: center;'>📌 Construção do Projeto 📌</h1>", unsafe_allow_html=True)
    st.write("")

    st.header("📌 Introdução")
    st.write("Neste projeto, decidimos usar o poder da ciência de dados para prever quem poderia ser mais suscetível a desenvolver diabetes. Com a ajuda do machine learning, buscamos desenvolver um modelo preditivo que possa nos ajudar a identificar os indivíduos em risco, permitindo intervenções precoces e talvez até mesmo a prevenção da doença.")
    st.image("imagens/1.jpg")
    st.write("")

    st.header("📌 Obtenção dos dados")
    st.write("Nosso ponto de partida foi um conjunto de dados disponível no Kaggle que contém informações médicas e comportamentais de indivíduos e pode ser acessado através deste [link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Através dessas caracteristicas, vamos ser capazes de prever se o paciente pode ou não ter a doença e qual a porcentagem disso acontecer.")
    st.write("")

    st.header("📌 Entendimento da área/negócio")
    st.write("Vamos começar entendendo a base de dados, aqui está uma tabela com as 10 primeiras linhas do nosso dataset.")
    st.dataframe(carregar_dados().head(10), hide_index=True)
    st.write("gender: Sexo do paciente;")
    st.write("hypertension: Indica se o paciente tem hipertensão (1 para sim, 0 para não);")
    st.write("heart_disease: Indica se o paciente tem doença cardíaca (1 para sim, 0 para não);")
    st.write("smoking_history: Histórico de fumo do paciente (nunca, atual, ex-fumante, etc);")
    st.write("bmi: Índice de Massa Corporal do paciente. O BMI é uma medida que tenta quantificar a quantidade de tecido muscular, gordura e osso de um indivíduo, e categoriza esse indivíduo como subpeso, peso normal, sobrepeso ou obeso com base nesse valor;")
    st.write("HbA1c_level: Nível de Hemoglobina Glicada (HbA1c) no sangue do paciente. A hemoglobina glicada é uma forma de hemoglobina que está ligada quimicamente a um açúcar. O nível de HbA1c no sangue de uma pessoa pode indicar o nível médio de açúcar no sangue em um período de semanas/meses;")
    st.write("blood_glucose_level: Nível de glicose no sangue do paciente;")
    st.write("diabetes: Indica se o paciente tem diabetes (1 para sim, 0 para não).")
    st.write("")

    st.header("📌 Limpeza e tratamento dos dados")
    st.write("Antes de poder mergulhar de cabeça na análise, temos que garantir que nossos dados estejam limpos e prontos para uso. Isso incluiu a remoção de linhas duplicadas e a exclusão de registros que não continham informações suficientes.")
    codigo1 = """
    # deleta colunas duplicadas
    tabela = tabela.drop_duplicates()

    # remove os valores não significativos da coluna genero
    tabela = tabela[tabela["gender"] != 'Other']

    # remove os pacientes com registro não informado sobre tabagismo
    tabela = tabela[tabela["smoking_history"] != 'No Info']
    """
    st.code(codigo1, language="python")
    st.write("Como o nosso modelo de inteligência artificial não é capaz de trabalhar com texto diretamente, também convertemos os dados categóricos em numéricos para facilitar o uso posterior.")
    codigo2 = """
    # Inicializa o label encoder
    le = LabelEncoder()

    # reajusta as colunas de texto para números
    tabela['gender'] = le.fit_transform(tabela['gender'])
    tabela['smoking_history'] = le.fit_transform(tabela['smoking_history'])
    """
    st.code(codigo2, language="python")
    st.write("")

    st.header("📌 Análise exploratória de dados")
    st.write("")

    st.header("📌 Modelando a inteligência artificial")
    st.write("")