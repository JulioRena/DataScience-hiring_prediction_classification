import pickle
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Carregar o Modelo
filename = 'modelo_treinado.pkl' 
loaded_model = pickle.load(open(filename, 'rb'))

scaler_filename = 'scaler.pkl'
scaler = pickle.load(open(scaler_filename, 'rb'))

# 2. Criar Novo Input (Exemplo)
novo_input = {
    'Age': 35,
    'Gender': 1,
    'EducationLevel': 4,
    'ExperienceYears': 10,
    'PreviousCompanies': 2,
    'DistanceFromCompany': 15.5,
    'InterviewScore': 85,
    'SkillScore': 90,
    'PersonalityScore': 75,
}
novo_input_df = pd.DataFrame([novo_input])

# ---  Aplicar Transformações (MANTENDO A ORDEM DO TREINAMENTO) ---

# Engenharia de Features 
novo_input_df['ExperiencePerCompany'] = novo_input_df['ExperienceYears'] / novo_input_df['PreviousCompanies']
novo_input_df['CombinedScore'] = 0.5 * novo_input_df['InterviewScore'] + 0.3 * novo_input_df['SkillScore'] + 0.2 * novo_input_df['PersonalityScore']

poly = PolynomialFeatures(degree=2, interaction_only=True)
interaction_features = poly.fit_transform(novo_input_df[['ExperienceYears', 'EducationLevel']])
novo_input_df['Experience_x_Education'] = interaction_features[:, 2]
novo_input_df['LogDistance'] = np.log1p(novo_input_df['DistanceFromCompany'])

#Escalonar as Variáveis Numéricas

numerical_cols = ['Age', 'EducationLevel', 'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore', 'ExperiencePerCompany', 'CombinedScore', 'LogDistance']
novo_input_df[numerical_cols] = scaler.fit_transform(novo_input_df[numerical_cols])

previsao = loaded_model.predict(novo_input_df)
print(previsao)