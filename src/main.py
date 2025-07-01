import sys
sys.path.append('loadData')
from load_data import load_data_df_x_y
from features import get_cat_features
from pipeline import build_model_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from evaluate import evaluate_model
from tuning import get_best_estimators
import joblib as jb

#art ascii Airbnb_model.v1
with open('gv1.txt', 'r', encoding='utf-8') as file:
    art = file.read()
print(art)
print('Airbnb_model.v1')

#Leitura dos datasets de treino e teste
x_train = pd.read_csv('x_train_resampled.csv', index_col=0)
y_train = pd.read_csv('y_train_resampled.csv', index_col=0)
x_test = pd.read_csv('x_test_resampled.csv', index_col=0) 
y_test = pd.read_csv('y_test_resampled.csv', index_col=0)
print(f'linhas de treino: {x_train.shape[0]}')

#captura das colunas categóricas
cat_cols = get_cat_features(x_train)

#instância do modelo pelo seu pipeline
model = build_model_pipeline(cat_cols=cat_cols)

#busca com gridsearch para encontrar o melhor modelo
best_model = get_best_estimators(model=model, x_train=x_train, y_train=y_train, cv=2)

#lista de predições do melhor modelo para uso do cálculo do acurácia
pred_list = best_model.predict(x_test)
print(f'acurácia: {evaluate_model(y_test, pred_list)}')

#salvar e exportar melhor modelo já treinado com joblib
try:
    jb.dump(best_model, filename='best_model_trained.joblib')
    print('-'*128)
    print('Modelo salvo e exportado como "best_model_trained.joblib"')
except Exception as e:
    print(f'Algo deu errado durante o saving e exportação do modelo. Rode o código de treino novamente | {e}')