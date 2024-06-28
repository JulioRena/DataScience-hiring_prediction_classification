import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
import pickle

df = pd.read_csv('./dataset/recruitment_data.csv')
df = df.drop('RecruitmentStrategy', axis=1)

#FEATURE ENGINEERING

df['ExperiencePerCompany'] = df['ExperienceYears'] / df['PreviousCompanies']

df['CombinedScore'] = df['InterviewScore'] + 0.3 * df['SkillScore'] + 0.2 * df['PersonalityScore']

df['LogDistance'] = np.log1p(df['DistanceFromCompany'])

poly = PolynomialFeatures(degree=2, interaction_only=True)
interaction_features = poly.fit_transform(df[['ExperienceYears', 'EducationLevel']])
df['Experience_x_education'] = interaction_features[:,2]

df.fillna(df.mean(), inplace=True)
df.dropna(inplace=True)

#print(df.head(10))
#PRE-PROCESSING
scaler = StandardScaler()


X = df.drop(['HiringDecision'], axis=1)
y = df['HiringDecision']
#feature RecruitmentStrategy may be a leakage

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=SEED)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {'LogisticRegression':LogisticRegression(),
          'KNN': KNeighborsClassifier(),
          'DecisionTree': DecisionTreeClassifier(),
          'RandomForest': RandomForestClassifier(),
          'SVM': SVC()}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(f'----- {name} -----')
#     print(f'Acurácia: {accuracy_score(y_test, y_pred)}')
#     print(f'Precisão: {precision_score(y_test, y_pred)}')
#     print(f'Recall: {recall_score(y_test, y_pred)}')
#     print(f'F1-Score: {f1_score(y_test, y_pred)}')
#     print(f'AUC: {roc_auc_score(y_test, y_pred)}')
    
model = RandomForestClassifier()

param_grid = {
    'n_estimators': [50,100,200], #trees n on forest
    'max_depth': [None, 10, 20], #tree max deep
    'min_samples_split': [2,5,10], #min samples to split
    'max_features': ['auto', 'sqrt'] #features numbe to each split
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1') #cv5 for 5 folds cross validation
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=5, scoring='f1', random_state=SEED)
random_search.fit(X_train, y_train)

best_params_random = random_search.best_params_
best_model_random = random_search.best_estimator_

y_pred = best_model.predict(X_test)

importances = best_model.feature_importances_

feature_importances = pd.DataFrame({'Feature':X_train.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# print(feature_importances)

# plt.figure(figsize=(10, 6))
# plt.barh(feature_importances['Feature'], feature_importances['Importance'])
# plt.xlabel('Importância da Feature')
# plt.ylabel('Feature')
# plt.title('Importância das Features com Random Forest')
# plt.show()

modelo = 'modelo_treinado.pkl'
pickle.dump(model, open(modelo, 'wb'))

scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)