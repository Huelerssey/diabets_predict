import pandas as pd


# importando dados
tabela = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

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

