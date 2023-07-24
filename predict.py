import pandas as pd


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

# demonstra a tabela com as 10 primeiras colunas
# print(tabela.head(10))
#    gender   age  hypertension  heart_disease smoking_history    bmi  HbA1c_level  blood_glucose_level  diabetes
# 0  Female  80.0             0              1           never  25.19          6.6                  140         0      
# 1  Female  54.0             0              0         No Info  27.32          6.6                   80         0      
# 2    Male  28.0             0              0           never  27.32          5.7                  158         0      
# 3  Female  36.0             0              0         current  23.45          5.0                  155         0      
# 4    Male  76.0             1              1         current  20.14          4.8                  155         0      
# 5  Female  20.0             0              0           never  27.32          6.6                   85         0      
# 6  Female  44.0             0              0           never  19.31          6.5                  200         1      
# 7  Female  79.0             0              0         No Info  23.86          5.7                   85         0      
# 8    Male  42.0             0              0           never  33.64          4.8                  145         0      
# 9  Female  32.0             0              0           never  27.32          5.0                  100         0 

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

## 4 - LIMPEZA E TRATAMENTO DE DADOS - ##

## 5 - ANÁLISE EXPLORATÓRIA DE DADOS - ##


## 5 - ANÁLISE EXPLORATÓRIA DE DADOS - ##

## 6 - MODELANDO UMA INTELIGÊNCIA ARTIFICIAL - ##


## 6 - MODELANDO UMA INTELIGÊNCIA ARTIFICIAL - ##
