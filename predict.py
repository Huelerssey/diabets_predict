import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


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

# Inicializa o codificador para a coluna genero
enc_gender = OneHotEncoder(sparse_output=False)

# Ajusta o codificador e transforma os dados
gender_encoded = enc_gender.fit_transform(tabela[['gender']])
gender_encoded_df = pd.DataFrame(gender_encoded, columns=enc_gender.get_feature_names_out(['gender']))

# Inicializa o codificador para a coluna histórico de tabagismo
enc_smoking = OneHotEncoder(sparse_output=False)

# Ajusta o codificador e transforma os dados
smoking_encoded = enc_smoking.fit_transform(tabela[['smoking_history']])
smoking_encoded_df = pd.DataFrame(smoking_encoded, columns=enc_smoking.get_feature_names_out(['smoking_history']))

# deleta as colunas originais e insere as colunas codificadas no dataframe
tabela = tabela.drop(['gender', 'smoking_history'], axis=1)
tabela = pd.concat([tabela, gender_encoded_df, smoking_encoded_df], axis=1)

## 4 - LIMPEZA E TRATAMENTO DE DADOS - ##

## 5 - ANÁLISE EXPLORATÓRIA DE DADOS - ##

# demosntra a correlação dos dados com a diabetes antes de tratar de outliers
# print(tabela.corr()['diabetes'])
# age                            0.262460
# hypertension                   0.191534
# heart_disease                  0.169040
# bmi                            0.203960
# HbA1c_level                    0.440930
# blood_glucose_level            0.451358
# diabetes                       1.000000
# gender_Female                 -0.002422
# gender_Male                    0.002422
# smoking_history_current        0.003198
# smoking_history_ever          -0.009651
# smoking_history_former        -0.005625
# smoking_history_never          0.007710
# smoking_history_not current   -0.002097

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
def diagrama_caixa(coluna):
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

# Exclui outliers e retorna o novo dataframe e também a quantidade de linhas removidas
def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), : ]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas

## FUNÇÕES AUXILIARES ##



## 5 - ANÁLISE EXPLORATÓRIA DE DADOS - ##

## 6 - MODELANDO UMA INTELIGÊNCIA ARTIFICIAL - ##


## 6 - MODELANDO UMA INTELIGÊNCIA ARTIFICIAL - ##
