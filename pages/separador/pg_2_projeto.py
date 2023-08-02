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
    st.write("---")

    st.header("📌 Introdução")
    st.write("Neste projeto, decidimos usar o poder da ciência de dados para prever quem poderia ser mais suscetível a desenvolver diabetes. Com a ajuda do machine learning, buscamos desenvolver um modelo preditivo que possa nos ajudar a identificar os indivíduos em risco, permitindo intervenções precoces e talvez até mesmo a prevenção da doença.")
    st.image("imagens/1.jpg")
    st.write("---")

    st.header("📌 Obtenção dos dados")
    st.write("Nosso ponto de partida foi um conjunto de dados disponível no Kaggle que contém informações médicas e comportamentais de indivíduos e pode ser acessado através deste [link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Através dessas caracteristicas, vamos ser capazes de prever se o paciente pode ou não ter a doença e qual a porcentagem disso acontecer.")
    st.write("---")

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
    st.write("---")

    st.header("📌 Limpeza e tratamento dos dados")
    st.write("Antes de poder mergulhar de cabeça na análise, temos que garantir que nossos dados estejam limpos e prontos para uso. Isso incluiu a remoção de linhas duplicadas e a exclusão de registros que não continham informações suficientes, transformar ou deletar dados irrelevantes e etc...")
    codigo0 = """
    # verificando dados da tabela
    print(tabela.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 9 columns):
    #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
    0   gender               100000 non-null  object 
    1   age                  100000 non-null  float64
    2   hypertension         100000 non-null  int64  
    3   heart_disease        100000 non-null  int64  
    4   smoking_history      100000 non-null  object 
    5   bmi                  100000 non-null  float64
    6   HbA1c_level          100000 non-null  float64
    7   blood_glucose_level  100000 non-null  int64  
    8   diabetes             100000 non-null  int64  
    dtypes: float64(3), int64(4), object(2)
    """
    st.code(codigo0, language='python')
    st.write("É importante notar que a etapa de limpeza de dados em um projeto de ciência de dados geralmente não é um processo linear. Frequentemente, pode ser necessário retornar a esta fase para ajustar ou remodelar os dados à medida que surgem novas necessidades ao longo do projeto. Felizmente, neste caso, o conjunto de dados que temos já veio relativamente limpo. Isso nos poupa tempo significativo e nos permite concentrar nossos esforços nas próximas etapas.")

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
    st.write("---")

    st.header("📌 Análise exploratória de dados")
    st.write("Nesta etapa, queremos entender como as diferentes variáveis estão correlacionadas com a diabetes. Nosso objetivo é identificar quais fatores são os mais influentes na previsão da doença. Além disso, usamos várias visualizações para entender a distribuição de nossos dados, identificar possíveis outliers e verificar se existe algum desequilíbrio em nossa variável de destino.")
    st.write("Pessoalmente gosto de dedicar uma seção inteira para a construção das funções que auxiliarão nas análises e na visualização de gráficos. Desta forma, antes de aplicá-las aos dados, garantimos uma organização e padronização rigorosas no projeto.")
    codigo3 = """
    ## FUNÇÕES AUXILIARES ##

    # gráfico de correlação dos dados com a diabetes
    correlacao = tabela.corr()[['diabetes']]
    correlacao_organizada = correlacao.sort_values(by='diabetes', ascending=False)
    plt.figure(figsize=(18, 7))
    sns.heatmap(correlacao_organizada, cmap="Blues", annot=True, fmt='.0%')
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
        return plt.show()
    
    # plota um gráfico de contagem
    def countplot(coluna):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=coluna, data=tabela)
    plt.ylabel('')
    return plt.show()

    # cada coluna do dataframe terá os gráficos plotados
    def plot_all_columns(df):
        for col in df.columns:
            print(f"Coluna: {col}")
            box_plot(df[col])
            histograma(df[col])
            grafico_pizza(df[col])
            countplot(df[col])

    # Exclui outliers e retorna o novo dataframe e também a quantidade de linhas removidas
    def excluir_outliers(df, nome_coluna):
        qtde_linhas = df.shape[0]
        lim_inf, lim_sup = limites(df[nome_coluna])
        df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), : ]
        linhas_removidas = qtde_linhas - df.shape[0]
        return df, linhas_removidas
    """
    st.code(codigo3, language='python')
    st.write("Com todas as funções construidas, vamos colocar a mão na massa nas análises!")
    
    st.write("---")
    st.write("")
    st.write("**Gráfico de calor** - para correlação entre as características e a variável de resultado diabetes. As cores mais escuras indicam uma correlação positiva mais forte.")
    
    # titulo do gráfico
    st.subheader('Correlação das caracteristicas com a Diabetes')

    # cacula a correlação
    correlation = carregar_dados().corr()[['diabetes']]
    correlation_sorted = correlation.sort_values(by='diabetes', ascending=False)
    
    # Calculate percentage and add % symbol
    correlation_percent = correlation_sorted * 100
    annotations = correlation_percent.applymap(lambda x: f'{x:.0f}%')

    # cria um gráfico de calor
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(correlation_sorted, cmap="Blues", annot=annotations, fmt='', ax=ax)
    st.pyplot(fig)
    st.write("")

    st.write("---")
    st.write("")
    st.write("")
    st.write("")
    # variáveis categóricas
    st.write("**Gráficos de barras para variáveis categóricas** - Esses gráficos mostram a distribuição das variáveis categóricas 'gender', 'smoking_history' e 'diabetes'. Podemos ver quantas observações temos para cada categoria.")
    
    # cria 3 colunas
    col1, col2, col3 = st.columns(3)

    # conteiner para organização
    with st.container():

        # grafico de distribuição para diabetes
        col1.subheader("Distribuição de diabetes")
        fig, ax = plt.subplots(figsize=(5, 4.06))
        sns.countplot(x='diabetes', data=carregar_dados(), ax=ax)
        col1.pyplot(fig)
        st.write("")

        # gráfico de distribuição para 'gender'
        col2.subheader('Distribuição de gênero')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='gender', data=carregar_dados(), ax=ax)
        col2.pyplot(fig)
        st.write("")

        # gráfico de distribuição para 'smoking_history'
        col3.subheader('Distribuição de tabagismo')
        fig, ax = plt.subplots(figsize=(5, 4.05))
        sns.countplot(x='smoking_history', data=carregar_dados(), ax=ax)
        col3.pyplot(fig)
        st.write("")

    st.write("---")
    st.write("")
    st.write("")
    st.write("")
    # vareáveis numéricas
    st.write("**Histogramas para variáveis numéricas** - Esses gráficos mostram a distribuição das variáveis numéricas 'idade', 'bmi', 'HbA1c_level' e 'blood_glucose_level'. A linha suave (KDE) representa uma estimativa da densidade de probabilidade dos dados, que pode ser útil para identificar a forma da distribuição dos dados.")

    # cria 2 colunas
    col1, col2 = st.columns(2)

    # conteiner para organização
    with st.container():

        # Histograma para 'age'
        col1.subheader('Distribuição de idade por pacientes')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(carregar_dados()['age'], bins=30, kde=True, ax=ax)
        col1.pyplot(fig)
        st.write("")

        # Histograma para 'bmi'
        col2.subheader('Distribuição de IMC por pacientes')
        fig, ax = plt.subplots(figsize=(5, 4.1))
        sns.histplot(carregar_dados()['bmi'], bins=30, kde=True, ax=ax)
        col2.pyplot(fig)
        st.write("")

        # Histograma para 'HbA1c_level'
        col1.subheader('Distribuição de hemoglobina glicada por pacientes')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(carregar_dados()['HbA1c_level'], bins=30, kde=True, ax=ax)
        col1.pyplot(fig)
        st.write("")

        # Histograma para 'blood_glucose_level'
        col2.subheader('Distribuição de nível de glicose no sangue por pacientes')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(carregar_dados()['blood_glucose_level'], bins=30, kde=True, ax=ax)
        col2.pyplot(fig)
        st.write("")

    st.write("---")
    st.write("")
    st.write("")
    st.write("")
    #variáveis numéricas
    st.write("**Box plots para variáveis numéricas por 'diabetes'** - Esses gráficos mostram a distribuição das variáveis numéricas divididas por 'diabetes'. Isso permite ver a diferença na distribuição dessas variáveis para pessoas com e sem diabetes, que nos auxilia a identificar outliers (valores extremos). Cada box plot mostra a mediana (a linha no meio da caixa), os quartis superior e inferior (as bordas da caixa) e os 'bigodes', que indicam a faixa dentro da qual a maioria dos dados se encontra.")

    # cria 2 colunas
    col1, col2 = st.columns(2)

    # conteiner para organização
    with st.container():

        # Boxplot para 'age'
        col1.subheader('Distribuição de Idade por Diabetes')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x='diabetes', y='age', data=carregar_dados(), ax=ax)
        col1.pyplot(fig)
        st.write("")

        # Boxplot para 'bmi'
        col2.subheader('Distribuição de IMC por Diabetes')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x='diabetes', y='bmi', data=carregar_dados(), ax=ax)
        col2.pyplot(fig)
        st.write("")

        # Boxplot para 'HbA1c_level'
        col1.subheader('Distribuição de Hemoglobina glicada por Diabetes')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x='diabetes', y='HbA1c_level', data=carregar_dados(), ax=ax)
        col1.pyplot(fig)
        st.write("")

        # Boxplot para 'blood_glucose_level'
        col2.subheader('Distribuição de Nível de glicose no sangue por Diabetes')
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sns.boxplot(x='diabetes', y='blood_glucose_level', data=carregar_dados(), ax=ax)
        col2.pyplot(fig)
        st.write("")

    st.write("---")
    st.write("Nossa análise cuidadosa dos gráficos nos levou a duas conclusões cruciais. A primeira é que a coluna do Índice de Massa Corporal (IMC) contém outliers - valores que se desviam significativamente do restante dos dados. Esses outliers precisarão ser tratados para evitar distorções em nossas futuras análises. A segunda conclusão é que a nossa variável de resposta, 'diabetes', está desbalanceada. Esse desbalanceamento pode ser prejudicial para o desempenho do nosso modelo de aprendizado de máquina, uma vez que ele poderia se tornar enviesado para a classe mais representada.")
    st.write("**Problema 1 - Outliers**: Uma questão-chave com o Índice de Massa Corporal (IMC) é que ele não reflete a distribuição de gordura corporal, um aspecto essencial para avaliar o sobrepeso, um fator de risco para diabetes. O IMC não diferencia entre massa magra (músculo) e massa gorda (gordura), que têm implicações de saúde muito diferentes. Considere dois indivíduos, ambos com 1,70 m de altura e pesando 100 kg. Um é fisiculturista e o outro leva um estilo de vida sedentário. Se calcularmos o IMC, ambos terão o mesmo resultado, indicando sobrepeso. No entanto, no caso do fisiculturista, esse fator específico não representa um risco para a sua saúde. Portanto, é crucial levar em conta as limitações do IMC ao usá-lo como indicador de saúde em nossa análise e por isso vamos excluir os valores discrepantes.")
    codigo4 = """
    #excluir outliers da coluna IMC
    tabela, linhas_removidas = excluir_outliers(tabela, 'bmi')
    print(f'{linhas_removidas} linhas removidas da coluna bmi')
    """
    st.code(codigo4, language='python')

    # cria duas colunas
    col1, col2 = st.columns(2)

    # conteiner
    with st.container():

        #coluna 1
        col1.write("**Problema 2 - Base desbalanceada**: Existem várias sub-categorias de problemas relacionadas ao desbalanceamento dos dados.")
        col1.write("**2.1 - Viés do Modelo**: Nosso modelo de aprendizado de máquina pode se tornar enviesado para a classe majoritária, neste caso, indivíduos não diabéticos. Isso acontece porque o modelo tende a se ajustar mais aos dados que aparecem com mais frequência para minimizar o erro durante o treinamento. Como resultado, o modelo pode prever muito bem a classe majoritária, mas pode ter um desempenho ruim na previsão da classe minoritária (indivíduos diabéticos).")
        col1.write("**2.2 - Dificuldade na avaliação**: Métricas comuns de avaliação de modelos, como acurácia, podem ser enganosas quando os dados estão desbalanceados. Por exemplo, um modelo que simplesmente prevê que todos os indivíduos são não diabéticos teria uma acurácia de 88% na sua base de dados, embora não seja útil para identificar indivíduos diabéticos.")
        col1.write("**Resolução**: Deu para perceber que é crucial lidar com o desbalanceamento dos dados antes de treinar o modelo. Uma abordagem comum é usar técnicas de reamostragem para equilibrar as classes. Isso pode ser feito reduzindo a classe majoritária, aumentando a classe minoritária ou uma combinação de ambas. Cada técnica tem suas próprias limitações e pode não ser adequada para todos os conjuntos de dados ou problemas. Por este motivo, na etapa seguinte vamos explorar ao máximo a modelagem da base de dados para garantir a maior assertividade da nossa inteligência artificial ao tentar prever se um paciente tem ou não diabetes, levando em consideração os falsos positivos, falso negativos e a probabilidade estatística para cada um deles.")

        #coluna 2
        # plota um gráfico de pizza
        fig, ax = plt.subplots(figsize=(5, 4))
        carregar_dados()['diabetes'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        col2.pyplot(fig)
    
    st.write("---")

    st.header("📌 Modelando a inteligência artificial")
    st.write("Chegou a hora de construir nossos modelos. Experimentamos vários algoritmos de aprendizado de máquina, incluindo Árvore de Decisão, Random Forest e Extra Trees. Conforme já explicado na etapa anterior, usamos diferentes métodos de reamostragem para lidar com o desequilíbrio em nossa variável de destino 'diabetes'. Após treinar os modelos, avaliamos seu desempenho usando métricas como a matriz de confusão, a pontuação de recall e a acurácia.")
    st.write("Novamente temos um toque pessoal em relação a como construir um projeto de machine learning. Com o objetivo de manter uma organização padrão e a otimização do tempo de entrega, construo uma função responsável por balancear, treinar, testar e avaliar todos os modelos de uma vez só! Após obter os resultados, podemos nos concentrar na análise e escolha do melhor modelo que irá subir para produção, ou seja, será utilizado pelo usuário final. Aqui está ela:")
    codigo5 = """
    # definindo dados de treino e de teste
    y = tabela['diabetes']
    x = tabela.drop('diabetes', axis=1)

    # dividindo a base entre treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

    # função para avaliar modelos
    def avaliar_modelos(modelos, x_treino, y_treino, x_teste, y_teste, resampling_methods):
        resultados = {}
        
        for nome, modelo in modelos.items():
            for resampling_method in resampling_methods:
                if resampling_method == 'Random Undersample':
                    rus = RandomUnderSampler(random_state=42)
                    x_res, y_res = rus.fit_resample(x_treino, y_treino)
                elif resampling_method == 'Undersample ClusterCentroid':
                    cc = ClusterCentroids(estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42)
                    x_res, y_res = cc.fit_resample(x_treino, y_treino)
                elif resampling_method == 'Undersample NearMiss':
                    nm = NearMiss()
                    x_res, y_res = nm.fit_resample(x_treino, y_treino)
                elif resampling_method == 'Random Oversample':
                    ros = RandomOverSampler(random_state=42, shrinkage=0.7)
                    x_res, y_res = ros.fit_resample(x_treino, y_treino)
                elif resampling_method == 'Oversample SMOTE':
                    sm = SMOTE(random_state=42)
                    x_res, y_res = sm.fit_resample(x_treino, y_treino)
                elif resampling_method == 'Oversample ADASYN':
                    ada = ADASYN(random_state=42)
                    x_res, y_res = ada.fit_resample(x_treino, y_treino)
                elif resampling_method == 'Combined Over/Undersample':
                    sme = SMOTEENN(random_state=42)
                    x_res, y_res = sme.fit_resample(x_treino, y_treino)
                else:
                    raise ValueError(f'Método de resampling desconhecido: {resampling_method}')
                
                modelo.fit(x_res, y_res)
                y_pred = modelo.predict(x_teste)
                cm = confusion_matrix(y_teste, y_pred)
                rs = recall_score(y_teste, y_pred)
                sa = accuracy_score(y_teste, y_pred)
                
                if nome not in resultados:
                    resultados[nome] = {}
                
                resultados[nome][resampling_method] = {
                    'Matriz de confusão': cm,
                    'Recall': rs,
                    'Acurácia': sa
                }
        
        return resultados

    # Criar o modelo de árvore de decisão
    clf = tree.DecisionTreeClassifier(random_state=42)

    # Criar o modelo de Random Forest
    clfrf = RandomForestClassifier(random_state=42)

    # Criar o modelo de Extra Trees
    clfet = ExtraTreesClassifier(random_state=42)

    # Criar o dicionário com os nomes dos modelos e as instâncias correspondentes
    modelos = {
        'Decision Tree': clf,
        'Random Forest': clfrf,
        'Extra Trees': clfet
    }

    # Definir os métodos de resampling a serem utilizados
    resampling_methods = ['Random Undersample', 'Undersample ClusterCentroid', 'Undersample NearMiss',
                        'Random Oversample', 'Oversample SMOTE', 'Oversample ADASYN',
                        'Combined Over/Undersample']

    # Chamar a função para avaliar os modelos
    resultados = avaliar_modelos(modelos, x_treino, y_treino, x_teste, y_teste, resampling_methods)

    # Imprimir os resultados
    for nome, resultado in resultados.items():
        print(f"Modelo: {nome}")
        for resampling_method, res in resultado.items():
            print(f"Método de resampling: {resampling_method}")
            print(f"Matriz de confusão: {res['Matriz de confusão']}")
            print(f"Recall: {res['Recall']:.2f}%")
            print(f"Acurácia: {res['Acurácia']:.2f}%")
    """
    st.code(codigo5, language='python')
    st.write("---")

    st.header("📌 Apresentação de Resultados")
    st.write("Os resultados foram interessantes. Cada modelo e método de reamostragem teve seus pontos fortes e fracos. Em alguns casos, obtivemos um recall muito alto, mas uma acurácia mais baixa. Em outros, a acurácia era alta, mas o recall não era tão impressionante. Para facilitar a visualização, disponibilizei todos eles a baixo. Utilize a legenda como guia e selecione o modelo desejado para ver seus resultados.")

    with st.container():

        #cria 3 colunas
        col1, col2, col3 = st.columns(3)

        # Lista de resultados
        lista_resultados = [
            {
                'modelo': 'Decision Tree',
                'metodo_resampling': 'Random Undersample',
                'matriz_confusao': [[14077, 2098], [249, 1606]],
                'recall': 0.87,
                'Acurácia': 0.87
            },
            {
                'modelo': 'Decision Tree',
                'metodo_resampling': 'Undersample ClusterCentroid',
                'matriz_confusao': [[11076, 5099], [56, 1799]],
                'recall': 0.97,
                'Acurácia': 0.71
            },
            {
                'modelo': 'Decision Tree',
                'metodo_resampling': 'Undersample NearMiss',
                'matriz_confusao': [[8536, 7639], [321, 1534]],
                'recall': 0.83,
                'Acurácia': 0.56
            },
            {
                'modelo': 'Decision Tree',
                'metodo_resampling': 'Random Oversample',
                'matriz_confusao': [[15499, 676], [458, 1397]],
                'recall': 0.75,
                'Acurácia': 0.94
            },
            {
                'modelo': 'Decision Tree',
                'metodo_resampling': 'Oversample SMOTE',
                'matriz_confusao': [[15461, 714], [471, 1384]],
                'recall': 0.75,
                'Acurácia': 0.93
            },
            {
                'modelo': 'Decision Tree',
                'metodo_resampling': 'Oversample ADASYN',
                'matriz_confusao': [[15497, 678], [487, 1368]],
                'recall': 0.74,
                'Acurácia': 0.94
            },
            {
                'modelo': 'Decision Tree',
                'metodo_resampling': 'Combined Over/Undersample',
                'matriz_confusao': [[14931, 1244], [333, 1522]],
                'recall': 0.82,
                'Acurácia': 0.91
            },
            {
                'modelo': 'Random Forest',
                'metodo_resampling': 'Random Undersample',
                'matriz_confusao': [[14373, 1802], [191, 1664]],
                'recall': 0.90,
                'Acurácia': 0.89
            },
            {
                'modelo': 'Random Forest',
                'metodo_resampling': 'Undersample ClusterCentroid',
                'matriz_confusao': [[11500, 4675], [46, 1809]],
                'recall': 0.98,
                'Acurácia': 0.74
            },
            {
                'modelo': 'Random Forest',
                'metodo_resampling': 'Undersample NearMiss',
                'matriz_confusao': [[10251, 5924], [375, 1480]],
                'recall': 0.80,
                'Acurácia': 0.65
            },
            {
                'modelo': 'Random Forest',
                'metodo_resampling': 'Random Oversample',
                'matriz_confusao': [[15798, 377], [516, 1339]],
                'recall': 0.72,
                'Acurácia': 0.95
            },
            {
                'modelo': 'Random Forest',
                'metodo_resampling': 'Oversample SMOTE',
                'matriz_confusao': [[15699, 476], [487, 1368]],
                'recall': 0.74,
                'Acurácia': 0.95
            },
            {
                'modelo': 'Random Forest',
                'metodo_resampling': 'Oversample ADASYN',
                'matriz_confusao': [[15502, 673], [458, 1397]],
                'recall': 0.75,
                'Acurácia': 0.94
            },
            {
                'modelo': 'Random Forest',
                'metodo_resampling': 'Combined Over/Undersample',
                'matriz_confusao': [[14926, 1249], [289, 1566]],
                'recall': 0.84,
                'Acurácia': 0.91
            },
            {
                'modelo': 'Extra Trees',
                'metodo_resampling': 'Random Undersample',
                'matriz_confusao': [[14305, 1870], [192, 1663]],
                'recall': 0.90,
                'Acurácia': 0.89
            },
            {
                'modelo': 'Extra Trees',
                'metodo_resampling': 'Undersample ClusterCentroid',
                'matriz_confusao': [[11885, 4290], [76, 1779]],
                'recall': 0.96,
                'Acurácia': 0.76
            },
            {
                'modelo': 'Extra Trees',
                'metodo_resampling': 'Undersample NearMiss',
                'matriz_confusao': [[11756, 4419], [368, 1487]],
                'recall': 0.80,
                'Acurácia': 0.73
            },
            {
                'modelo': 'Extra Trees',
                'metodo_resampling': 'Random Oversample',
                'matriz_confusao': [[15493, 682], [447, 1408]],
                'recall': 0.76,
                'Acurácia': 0.94
            },
            {
                'modelo': 'Extra Trees',
                'metodo_resampling': 'Oversample SMOTE',
                'matriz_confusao': [[15569, 606], [484, 1371]],
                'recall': 0.74,
                'Acurácia': 0.94
            },
            {
                'modelo': 'Extra Trees',
                'metodo_resampling': 'Oversample ADASYN',
                'matriz_confusao': [[15335, 840], [452, 1403]],
                'recall': 0.76,
                'Acurácia': 0.93
            },
            {
                'modelo': 'Extra Trees',
                'metodo_resampling': 'Combined Over/Undersample',
                'matriz_confusao': [[14816, 1359], [283, 1572]],
                'recall': 0.85,
                'Acurácia': 0.91
            },
        ]

        # Opções disponíveis no seletor
        opcoes = [f"{resultado['modelo']} - {resultado['metodo_resampling']}" for resultado in lista_resultados]

        # coluna do meio
        with col2:
            st.write("Inteligência Artificial - Método de Reajuste da Base de Dados")
            # Seletor de opções
            opcao_selecionada = st.selectbox("Selecione uma opção:", opcoes)

    with st.container():

        #cria 3 colunas
        col1, col2 = st.columns(2)

        # Encontrar o resultado correspondente à opção selecionada
        resultado_selecionado = None
        for resultado in lista_resultados:
            if f"{resultado['modelo']} - {resultado['metodo_resampling']}" == opcao_selecionada:
                resultado_selecionado = resultado
                break

        # Verificar se um resultado válido foi selecionado
        if resultado_selecionado is not None:
            # Dados da matriz de confusão
            matriz_confusao = resultado_selecionado['matriz_confusao']

            # Cores das fatias para cada gráfico
            cores_fatias1 = ['#00FF00', '#FF0000']
            cores_fatias2 = ['#FF0000', '#00FF00']

            # Legendas
            legenda1 = ['Errou - Modelo não classificou o paciente como Diabético', 'Acertou - Modelo classificou o paciente como Diabético']
            legenda2 = ['Acertou - Modelo não classificou o paciente como Diabético', 'Errou - Modelo classificou o paciente como Diabético']

            col1.subheader("Tentando prever Pacientes que eram Diabéticos")
            # Gráfico dos dados que eram Diabetes
            plt.figure(figsize=(10, 10))
            plt.pie(matriz_confusao[1], colors=cores_fatias2, autopct='%1.1f%%', startangle=90)
            plt.legend(legenda1, loc='upper left', bbox_to_anchor=(0.80, 0.80), bbox_transform=plt.gcf().transFigure)
            plt.axis('equal')

            # Exibir o gráfico no Streamlit
            col1.pyplot(plt)

            col2.subheader("Tentando prever Pacientes que não eram Diabéticos")
            # Gráfico dos dados que não eram Diabetes
            plt.figure(figsize=(10, 10.15))
            plt.pie(matriz_confusao[0], colors=cores_fatias1, autopct='%1.1f%%', startangle=90)
            plt.legend(legenda2, loc='upper left', bbox_to_anchor=(0.80, 0.80), bbox_transform=plt.gcf().transFigure)
            plt.axis('equal')

            # Exibir o gráfico no Streamlit
            col2.pyplot(plt)
            st.write(f"Acurácia: {resultado_selecionado['Acurácia']*100}% - A porcentagem total de previsões que o modelo acertou;")
            st.write(f"Recall: {resultado_selecionado['recall']*100}% - A porcentagem de pacientes verdadeiramente diabéticos que foram corretamente identificados pelo modelo.")
    st.write("---")

    st.header("📌 Escolhendo o melhor Modelo e colocando em Produção")
    st.write("Com todos os resultados em mãos, escolhemos o modelo Random Forest com o método de subamostragem Random Under Sampler. Esse modelo ofereceu um bom equilíbrio entre recall e acurácia, tornando-o uma escolha sólida para nossa aplicação. Treinamos o modelo final com todo o conjunto de dados, e agora ele está pronto para ser usado para prever se um indivíduo pode desenvolver diabetes e qual a probabilidade disso acontecer.")
    st.write("Aqui está o código do modelo responsável pelas previsões:")
    codigo6 = """
    # definindo dados de treino e de teste
    y = tabela['diabetes']
    x = tabela.drop('diabetes', axis=1)

    # dividindo a base entre treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state=42, stratify=y)

    # Criar o modelo de Random Forest
    clf = RandomForestClassifier(random_state=42)

    # Instanciar o RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)

    # Aplicar o resampling
    x_res, y_res = rus.fit_resample(x_treino, y_treino)

    # treina o modelo
    clf.fit(x_res, y_res)

    # testa o modelo
    y_pred = clf.predict(x_teste)

    # obtendo a probabilidade de ser da classe 1 (diabetes)
    prob_diabetes = clf.predict_proba(x_teste)[:, 1]

    # colocando modelo para produção
    joblib.dump(clf, "modelo_treinado.pkl")
    """
    st.code(codigo6, language='python')

    st.write("E aqui está o código responsável por captar as respostas do usuário que serão entregues ao modelo para que ele possa ser capaz de prever:")
    codigo7 = """
    # Carregar o modelo treinado
    clf = joblib.load("arquivos_pkl/modelo_treinado.pkl")

    # Função para fazer previsões
    def predict_diabetes(model, patient_info):
        prediction = model.predict(patient_info)
        prediction_proba = model.predict_proba(patient_info)[0, int(prediction[0])]
        return prediction, prediction_proba

    # Mapeamento da coluna gender
    gender_mapping = {
        'Homem': 0,
        'Mulher': 1
    }
    gender = st.selectbox('Gênero', list(gender_mapping.keys()))
    gender = gender_mapping[gender]  # Converter para o código correspondente

    # idade
    age = st.number_input('Idade', min_value=0, max_value=100)

    # Mapeamento para as colunas hypertension e heart_disease
    binary_mapping = {
        'Não': 0,
        'Sim': 1
    }
    hypertension = st.selectbox('Hipertensão', list(binary_mapping.keys()))
    hypertension = binary_mapping[hypertension]  # Converter para o código correspondente
    heart_disease = st.selectbox('Doença Cardíaca', list(binary_mapping.keys()))
    heart_disease = binary_mapping[heart_disease]  # Converter para o código correspondente

    # Mapeamento da coluna smoking_history
    smoking_mapping = {
        'Nunca fumou': 3,
        'Fumante frequente': 0,
        'Ex-fumante': 2,
        'Pelo menos uma vez': 1,
        'Fumante ocasional': 4
    }
    smoking_history = st.selectbox('Histórico de Fumante', list(smoking_mapping.keys()))
    smoking_history = smoking_mapping[smoking_history]  # Converter para o código correspondente

    #IMC
    bmi = st.number_input('IMC', min_value=0.0)

    #nivel de hemoglobina glicada
    HbA1c_level = st.number_input('Nível de HbA1c', min_value=0.0)

    #nivel de glicose
    blood_glucose_level = st.number_input('Nível de Glicose no Sangue', min_value=0)

    # Botão para fazer previsão
    if st.button('Fazer previsão'):
        # Preparar os dados para a previsão
        patient_info = pd.DataFrame(np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]]),
                                    columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

        # Fazer a previsão
        prediction, prediction_proba = predict_diabetes(clf, patient_info)
        if prediction[0] == 1:
            st.success(f'O modelo classificou o paciente como diabético com {prediction_proba*100:.2f}% de chance de acerto.')
        else:
            st.success(f'O modelo classificou o paciente como não diabético com {prediction_proba*100:.2f}% de chance de acerto.')
    """
    st.code(codigo7, language='python')
    st.write("---")

    #footer
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        col2.write("Developed By: [@Huelerssey](https://github.com/Huelerssey)")