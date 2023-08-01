import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# função que otimiza o carregamento dos dados
@st.cache_data
def carregar_dados():
    tabela = pd.read_pickle("arquivos_pkl/dataframe_modelado.pkl")
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
    st.write("É importante notar que a etapa de limpeza de dados em um projeto de ciência de dados geralmente não é um processo linear. Frequentemente, pode ser necessário retornar a esta fase para ajustar ou remodelar os dados à medida que surgem novas necessidades ao longo do projeto. Felizmente, neste caso, o conjunto de dados que temos já veio relativamente limpo. Isso nos poupa tempo significativo e nos permite concentrar nossos esforços nas etapas subsequentes de análise e modelagem de dados.")
    st.write("")

    st.header("📌 Análise exploratória de dados")
    st.write("Nesta etapa, queremos entender como as diferentes variáveis estão correlacionadas com a diabetes. Nosso objetivo é identificar quais fatores são os mais influentes na previsão da doença. Além disso, usamos várias visualizações para entender a distribuição de nossos dados, identificar possíveis outliers e verificar se existe algum desequilíbrio em nossa variável de destino.")
    st.write("É uma prática recomendada dedicar uma seção inteira para a construção das funções que auxiliarão nas análises e na visualização de gráficos. Desta forma, antes de aplicá-las aos dados, garantimos uma organização e padronização rigorosas no projeto. A seguir, apresentamos as funções auxiliares desenvolvidas para este propósito:")
    codigo3 = """
    ## FUNÇÕES AUXILIARES ##

    # gráfico de correlação dos dados com a diabetes
    correlation = tabela.corr()[['diabetes']]
    correlation_sorted = correlation.sort_values(by='diabetes', ascending=False)
    plt.figure(figsize=(18, 7))
    sns.heatmap(correlation_sorted, cmap="Blues", annot=True, fmt='.0%')
    plt.show()

    # função que retorna todos os valores únicos de uma coluna
    def verificar_val_unicos(dataframe):
        valores_unicos = {}
        for col in dataframe.columns:
            valores_unicos[col] = list(dataframe[col].unique())
        for col, vals in valores_unicos.items():
            print(f"{col}: {vals}")
        return valores_unicos

    # retorna o limite inferior e o limite superior
    def limites(coluna):
        q1 = coluna.quantile(0.25)
        q3 = coluna.quantile(0.75)
        amplitude = q3 - q1
        return (q1 - 1.5 * amplitude, q3 + 1.5 * amplitude)

    # plota 2 gráficos sendo o primeiro com os outliers e o segundo, sem
    def box_plot(coluna):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(15,5)
        sns.boxplot(x=coluna, ax=ax1)
        ax2.set_xlim(limites(coluna))
        sns.boxplot(x=coluna, ax=ax2)
        return plt.show()

    # plota um gráfico de histograma
    def histograma(coluna):
        plt.figure(figsize=(15, 5))
        sns.histplot(coluna, kde=True)
        return plt.show()

    # plota um gráfico de pizza
    def grafico_pizza(coluna):
        plt.figure(figsize=(10, 6))
        coluna.value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.ylabel('')
        plt.show()

    # cada coluna do dataframe terá os gráficos plotados
    def plot_all_columns(df):
        for col in df.columns:
            print(f"Coluna: {col}")
            box_plot(df[col])
            histograma(df[col])
            grafico_pizza(df[col])

    # Exclui outliers e retorna o novo dataframe e também a quantidade de linhas removidas
    def excluir_outliers(df, nome_coluna):
        qtde_linhas = df.shape[0]
        lim_inf, lim_sup = limites(df[nome_coluna])
        df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), : ]
        linhas_removidas = qtde_linhas - df.shape[0]
        return df, linhas_removidas
    """
    st.code(codigo3, language='python')
    st.write("Com todas as funções construidas, vamos colocar a mão na massa nas análises! Vamos começar com um gráfico de calor da correlação entre as características e a variável de resultado diabetes. As cores mais escuras indicam uma correlação positiva mais forte.")
    
    # titulo do gráfico
    st.subheader('Correlação das caracteristicas com a Diabetes')

    # cacula a correlação
    correlation = carregar_dados().corr()[['diabetes']]
    correlation_sorted = correlation.sort_values(by='diabetes', ascending=False)
    
    # Calculate percentage and add % symbol
    correlation_percent = correlation_sorted * 100
    annotations = correlation_percent.applymap(lambda x: f'{x:.0f}%')

    # cria um gráfico de calor
    fig, ax = plt.subplots(figsize=(18, 7))
    sns.heatmap(correlation_sorted, cmap="Blues", annot=annotations, fmt='', ax=ax)
    st.pyplot(fig)
    st.write("")
    
    # variáveis categóricas
    st.write("Gráficos de barras para variáveis categóricas - Esses gráficos mostram a distribuição das variáveis categóricas 'gender', 'smoking_history' e 'diabetes'. Podemos ver quantas observações temos para cada categoria.")
    
    # grafico de distribuição
    st.subheader("Distribuição de diabetes por pacientes")
    fig, ax = plt.subplots()
    sns.countplot(x='diabetes', data=carregar_dados(), ax=ax)
    st.pyplot(fig)
    st.write("")

    # gráfico de distribuição para 'gender'
    st.subheader('Distribuição de gênero por pacientes')
    fig, ax = plt.subplots()
    sns.countplot(x='gender', data=carregar_dados(), ax=ax)
    st.pyplot(fig)
    st.write("")

    # gráfico de distribuição para 'smoking_history'
    st.subheader('Distribuição de histórico de tabagismo por pacientes')
    fig, ax = plt.subplots()
    sns.countplot(x='smoking_history', data=carregar_dados(), ax=ax)
    st.pyplot(fig)
    st.write("")

    st.write("Histogramas para variáveis numéricas - Esses gráficos mostram a distribuição das variáveis numéricas 'idade', 'bmi', 'HbA1c_level' e 'blood_glucose_level'. A linha suave (KDE) representa uma estimativa da densidade de probabilidade dos dados, que pode ser útil para identificar a forma da distribuição dos dados.")

    # Histograma para 'age'
    st.subheader('Distribuição de idade por pacientes')
    fig, ax = plt.subplots()
    sns.histplot(carregar_dados()['age'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    st.write("")

    # Histograma para 'bmi'
    st.subheader('Distribuição de indice de massa corporal por pacientes')
    fig, ax = plt.subplots()
    sns.histplot(carregar_dados()['bmi'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    st.write("")

    # Histograma para 'HbA1c_level'
    st.subheader('Distribuição de hemoglobina glicada por pacientes')
    fig, ax = plt.subplots()
    sns.histplot(carregar_dados()['HbA1c_level'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    st.write("")

    # Histograma para 'blood_glucose_level'
    st.subheader('Distribuição de nível de glicose no sangue por pacientes')
    fig, ax = plt.subplots()
    sns.histplot(carregar_dados()['blood_glucose_level'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    st.write("")

    st.write("Box plots para variáveis numéricas por 'diabetes' - Esses gráficos mostram a distribuição das variáveis numéricas divididas por 'diabetes'. Isso permite ver a diferença na distribuição dessas variáveis para pessoas com e sem diabetes, que nos auxilia a identificar outliers (valores extremos). Cada box plot mostra a mediana (a linha no meio da caixa), os quartis superior e inferior (as bordas da caixa) e os 'bigodes', que indicam a faixa dentro da qual a maioria dos dados se encontra.")

    # Boxplot para 'age'
    st.subheader('Distribuição de Idade por Diabetes')
    fig, ax = plt.subplots()
    sns.boxplot(x='diabetes', y='age', data=carregar_dados(), ax=ax)
    st.pyplot(fig)
    st.write("")

    # Boxplot para 'bmi'
    st.subheader('Distribuição de Indice de massa corporal por Diabetes')
    fig, ax = plt.subplots()
    sns.boxplot(x='diabetes', y='bmi', data=carregar_dados(), ax=ax)
    st.pyplot(fig)
    st.write("")

    # Boxplot para 'HbA1c_level'
    st.subheader('Distribuição de Hemoglobina glicada por Diabetes')
    fig, ax = plt.subplots()
    sns.boxplot(x='diabetes', y='HbA1c_level', data=carregar_dados(), ax=ax)
    st.pyplot(fig)
    st.write("")

    # Boxplot para 'blood_glucose_level'
    st.subheader('Distribuição de Nível de glicose no sangue por Diabetes')
    fig, ax = plt.subplots()
    sns.boxplot(x='diabetes', y='blood_glucose_level', data=carregar_dados(), ax=ax)
    st.pyplot(fig)
    st.write("")

    st.write("")

    st.header("📌 Modelando a inteligência artificial")
    st.write("")