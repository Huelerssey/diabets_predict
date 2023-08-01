import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix, recall_score
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


## 1 - INTRODUÇÃO - ##

"""
Projeto de ciência de dados com o objetivo de construir um modelo de previsão com
machine learning que seja capaz de prever se um paciente irá ou não ter diabetes e
qual a probabilidade disso acontecer.
"""

## 1 - INTRODUÇÃO - ##

## 2 - OBTENÇÃO DOS DADOS ##

#https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

# importando dados
tabela = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

## 2 - OBTENÇÃO DOS DADOS - ##

## 3 - ENTENDIMENTO DA ÁREA/NEGÓCIO - ##

"""
demonstra a tabela com as 10 primeiras colunas
print(tabela.head(10))
   gender   age  hypertension  heart_disease smoking_history    bmi  HbA1c_level  blood_glucose_level  diabetes
0  Female  80.0             0              1           never  25.19          6.6                  140         0      
1  Female  54.0             0              0         No Info  27.32          6.6                   80         0      
2    Male  28.0             0              0           never  27.32          5.7                  158         0      
3  Female  36.0             0              0         current  23.45          5.0                  155         0      
4    Male  76.0             1              1         current  20.14          4.8                  155         0      
5  Female  20.0             0              0           never  27.32          6.6                   85         0      
6  Female  44.0             0              0           never  19.31          6.5                  200         1      
7  Female  79.0             0              0         No Info  23.86          5.7                   85         0      
8    Male  42.0             0              0           never  33.64          4.8                  145         0      
9  Female  32.0             0              0           never  27.32          5.0                  100         0 

gender: Sexo do paciente (Masculino ou Feminino).
age: Idade do paciente.
hypertension: Indica se o paciente tem hipertensão (1 para sim, 0 para não).
heart_disease: Indica se o paciente tem doença cardíaca (1 para sim, 0 para não).
smoking_history: Histórico de fumo do paciente (nunca, atual, etc).
bmi: Índice de Massa Corporal do paciente. O BMI é uma medida que tenta quantificar a quantidade de tecido muscular, gordura e osso de um indivíduo, e categoriza esse indivíduo como subpeso, peso normal, sobrepeso ou obeso com base nesse valor.
HbA1c_level: Nível de Hemoglobina Glicada (HbA1c) no sangue do paciente. A hemoglobina glicada é uma forma de hemoglobina que está ligada quimicamente a um açúcar. O nível de HbA1c no sangue de uma pessoa pode indicar o nível médio de açúcar no sangue em um período de semanas/meses.
blood_glucose_level: Nível de glicose no sangue do paciente.
diabetes: Indica se o paciente tem diabetes (1 para sim, 0 para não).
"""

## 3 - ENTENDIMENTO DA ÁREA/NEGÓCIO - ##

## 4 - LIMPEZA E TRATAMENTO DE DADOS - ##

# verificando dados da tabela
# print(tabela.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 100000 entries, 0 to 99999
# Data columns (total 9 columns):
#  #   Column               Non-Null Count   Dtype  
# ---  ------               --------------   -----  
#  0   gender               100000 non-null  object 
#  1   age                  100000 non-null  float64
#  2   hypertension         100000 non-null  int64  
#  3   heart_disease        100000 non-null  int64  
#  4   smoking_history      100000 non-null  object 
#  5   bmi                  100000 non-null  float64
#  6   HbA1c_level          100000 non-null  float64
#  7   blood_glucose_level  100000 non-null  int64  
#  8   diabetes             100000 non-null  int64  
# dtypes: float64(3), int64(4), object(2)

# deleta colunas duplicadas
tabela = tabela.drop_duplicates()

# remove os valores não significativos da coluna genero
tabela = tabela[tabela["gender"] != 'Other']

# remove os pacientes com registro não informado sobre tabagismo
tabela = tabela[tabela["smoking_history"] != 'No Info']

# Inicializa o label encoder
le = LabelEncoder()

# reajusta as colunas de texto para números
tabela['gender'] = le.fit_transform(tabela['gender'])
tabela['smoking_history'] = le.fit_transform(tabela['smoking_history'])

# salva a base de dados modelada para uso no streamlit
tabela.to_pickle("dataframe_modelado.pkl")

## 4 - LIMPEZA E TRATAMENTO DE DADOS - ##

## 5 - ANÁLISE EXPLORATÓRIA DE DADOS - ##

# demosntra a correlação dos dados com a diabetes antes de tratar de outliers
# print(tabela.corr()['diabetes'])
# gender                 0.056472
# age                    0.262460
# hypertension           0.191534
# heart_disease          0.169040
# smoking_history       -0.016840
# bmi                    0.203960
# HbA1c_level            0.440930
# blood_glucose_level    0.451358
# diabetes               1.000000

# gráfico de correlação dos dados com a diabetes antes de tratar de outliers
# correlation = tabela.corr()[['diabetes']]
# correlation_sorted = correlation.sort_values(by='diabetes', ascending=False)
# plt.figure(figsize=(18, 7))
# sns.heatmap(correlation_sorted, cmap="Blues", annot=True, fmt='.0%')
# plt.show()

## FUNÇÕES AUXILIARES ##

# função que retorna todos os valores únicos de uma coluna
def verificar_val_unicos(dataframe):
    valores_unicos = {}
    for col in dataframe.columns:
        valores_unicos[col] = list(dataframe[col].unique())
    for col, vals in valores_unicos.items():
        print(f"{col}: {vals}\n")
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

## FUNÇÕES AUXILIARES ##

# plota todos os gráficos de todas as colunas para análise
# plot_all_columns(tabela)

#excluir outliers da coluna
tabela, linhas_removidas = excluir_outliers(tabela, 'bmi')
print(f'{linhas_removidas} linhas removidas da coluna bmi')

## 5 - ANÁLISE EXPLORATÓRIA DE DADOS - ##

## 6 - MODELANDO UMA INTELIGÊNCIA ARTIFICIAL - ##

# # definindo dados de treino e de teste
# y = tabela['diabetes']
# x = tabela.drop('diabetes', axis=1)

# # dividindo a base entre treino e teste
# x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

# # função para avaliar modelos
# def avaliar_modelos(modelos, x_treino, y_treino, x_teste, y_teste, resampling_methods):
#     resultados = {}
    
#     for nome, modelo in modelos.items():
#         for resampling_method in resampling_methods:
#             if resampling_method == 'Random Undersample':
#                 rus = RandomUnderSampler(random_state=42)
#                 x_res, y_res = rus.fit_resample(x_treino, y_treino)
#             elif resampling_method == 'Undersample ClusterCentroid':
#                 cc = ClusterCentroids(estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=42)
#                 x_res, y_res = cc.fit_resample(x_treino, y_treino)
#             elif resampling_method == 'Undersample NearMiss':
#                 nm = NearMiss()
#                 x_res, y_res = nm.fit_resample(x_treino, y_treino)
#             elif resampling_method == 'Random Oversample':
#                 ros = RandomOverSampler(random_state=42, shrinkage=0.7)
#                 x_res, y_res = ros.fit_resample(x_treino, y_treino)
#             elif resampling_method == 'Oversample SMOTE':
#                 sm = SMOTE(random_state=42)
#                 x_res, y_res = sm.fit_resample(x_treino, y_treino)
#             elif resampling_method == 'Oversample ADASYN':
#                 ada = ADASYN(random_state=42)
#                 x_res, y_res = ada.fit_resample(x_treino, y_treino)
#             elif resampling_method == 'Combined Over/Undersample':
#                 sme = SMOTEENN(random_state=42)
#                 x_res, y_res = sme.fit_resample(x_treino, y_treino)
#             else:
#                 raise ValueError(f'Método de resampling desconhecido: {resampling_method}')
            
#             modelo.fit(x_res, y_res)
#             y_pred = modelo.predict(x_teste)
#             cm = confusion_matrix(y_teste, y_pred)
#             rs = recall_score(y_teste, y_pred)
#             sa = accuracy_score(y_teste, y_pred)
            
#             if nome not in resultados:
#                 resultados[nome] = {}
            
#             resultados[nome][resampling_method] = {
#                 'Matriz de confusão': cm,
#                 'Recall': rs,
#                 'Acurácia': sa
#             }
    
#     return resultados

# # Criar o modelo de árvore de decisão
# clf = tree.DecisionTreeClassifier(random_state=42)

# # Criar o modelo de Random Forest
# clfrf = RandomForestClassifier(random_state=42)

# # Criar o modelo de Extra Trees
# clfet = ExtraTreesClassifier(random_state=42)

# # Criar o dicionário com os nomes dos modelos e as instâncias correspondentes
# modelos = {
#     'Decision Tree': clf,
#     'Random Forest': clfrf,
#     'Extra Trees': clfet
# }

# # Definir os métodos de resampling a serem utilizados
# resampling_methods = ['Random Undersample', 'Undersample ClusterCentroid', 'Undersample NearMiss',
#                       'Random Oversample', 'Oversample SMOTE', 'Oversample ADASYN',
#                       'Combined Over/Undersample']

# # Chamar a função para avaliar os modelos
# resultados = avaliar_modelos(modelos, x_treino, y_treino, x_teste, y_teste, resampling_methods)

# # Imprimir os resultados
# for nome, resultado in resultados.items():
#     print(f"Modelo: {nome}")
#     for resampling_method, res in resultado.items():
#         print(f"Método de resampling: {resampling_method}")
#         print(f"Matriz de confusão:\n {res['Matriz de confusão']}")
#         print(f"Recall: {res['Recall']:.2f}%")
#         print(f"Acurácia: {res['Acurácia']:.2f}%")

## 6 - MODELANDO UMA INTELIGÊNCIA ARTIFICIAL - ##

## 7 - RESULTADOS - ##

##                                      DECISION TREE                                 ##

# Método de resampling: Random Undersample
# Matriz de confusão:
#  [[14077  2098]
#  [  249  1606]]
# Recall: 0.87%
# Acurácia: 0.87%

# Método de resampling: Undersample ClusterCentroid
# Matriz de confusão:
#  [[11076  5099]
#  [   56  1799]]
# Recall: 0.97%
# Acurácia: 0.71%

# Método de resampling: Undersample NearMiss
# Matriz de confusão:
#  [[8536 7639]
#  [ 321 1534]]
# Recall: 0.83%
# Acurácia: 0.56%

# Método de resampling: Random Oversample
# Matriz de confusão:
#  [[15499   676]
#  [  458  1397]]
# Recall: 0.75%
# Acurácia: 0.94%

# Método de resampling: Oversample SMOTE
# Matriz de confusão:
#  [[15461   714]
#  [  471  1384]]
# Recall: 0.75%
# Acurácia: 0.93%

# Método de resampling: Oversample ADASYN
# Matriz de confusão:
#  [[15497   678]
#  [  487  1368]]
# Recall: 0.74%
# Acurácia: 0.94%

# Método de resampling: Combined Over/Undersample
# Matriz de confusão:
#  [[14931  1244]
#  [  333  1522]]
# Recall: 0.82%
# Acurácia: 0.91%

##                                      RANDOM FOREST                                 ##

# Método de resampling: Random Undersample
# Matriz de confusão:
#  [[14373  1802]
#  [  191  1664]]
# Recall: 0.90%
# Acurácia: 0.89%

# Método de resampling: Undersample ClusterCentroid
# Matriz de confusão:
#  [[11500  4675]
#  [   46  1809]]
# Recall: 0.98%
# Acurácia: 0.74%

# Método de resampling: Undersample NearMiss
# Matriz de confusão:
#  [[10251  5924]
#  [  375  1480]]
# Recall: 0.80%
# Acurácia: 0.65%

# Método de resampling: Random Oversample
# Matriz de confusão:
#  [[15798   377]
#  [  516  1339]]
# Recall: 0.72%
# Acurácia: 0.95%

# Método de resampling: Oversample SMOTE
# Matriz de confusão:
#  [[15699   476]
#  [  487  1368]]
# Recall: 0.74%
# Acurácia: 0.95%

# Método de resampling: Oversample ADASYN
# Matriz de confusão:
#  [[15502   673]
#  [  458  1397]]
# Recall: 0.75%
# Acurácia: 0.94%

# Método de resampling: Combined Over/Undersample
# Matriz de confusão:
#  [[14926  1249]
#  [  289  1566]]
# Recall: 0.84%
# Acurácia: 0.91%

##                                      EXTRA TREES                                   ##

# Método de resampling: Random Undersample
# Matriz de confusão:
#  [[14305  1870]
#  [  192  1663]]
# Recall: 0.90%
# Acurácia: 0.89%

# Método de resampling: Undersample ClusterCentroid
# Matriz de confusão:
#  [[11885  4290]
#  [   76  1779]]
# Recall: 0.96%
# Acurácia: 0.76%

# Método de resampling: Undersample NearMiss
# Matriz de confusão:
#  [[11756  4419]
#  [  368  1487]]
# Recall: 0.80%
# Acurácia: 0.73%

# Método de resampling: Random Oversample
# Matriz de confusão:
#  [[15493   682]
#  [  447  1408]]
# Recall: 0.76%
# Acurácia: 0.94%

# Método de resampling: Oversample SMOTE
# Matriz de confusão:
#  [[15569   606]
#  [  484  1371]]
# Recall: 0.74%
# Acurácia: 0.94%

# Método de resampling: Oversample ADASYN
# Matriz de confusão:
#  [[15335   840]
#  [  452  1403]]
# Recall: 0.76%
# Acurácia: 0.93%

# Método de resampling: Combined Over/Undersample
# Matriz de confusão:
#  [[14816  1359]
#  [  283  1572]]
# Recall: 0.85%
# Acurácia: 0.91%

## 7 - RESULTADOS - ##

## 8 - ESCOLHENDO O MELHOR MODELO E COLOCANDO EM PRODUÇÃO - ##

# MODELO ESCOLHIDO: Random Forest - Random Undersample

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
# joblib.dump(clf, "modelo_treinado.pkl")

## 8 - ESCOLHENDO O MELHOR MODELO E COLOCANDO EM PRODUÇÃO - ##