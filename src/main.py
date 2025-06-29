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
x = pd.read_csv('x_train_resampled.csv', index_col=0)
y = pd.read_csv('y_train_resampled.csv', index_col=0)

#captura das colunas categóricas
cat_cols = get_cat_features(x)

#separação para utilização de apenas 15% do dataset (15% de ~1.23e+6 linhas)
x_fraction, _, y_fraction, _ = train_test_split(x, y, train_size=0.15, stratify=y)
#separação de treino e teste para uso no modelo
x_train, x_test, y_train, y_test = train_test_split(x_fraction, y_fraction, test_size=0.2, stratify=y_fraction)

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
    print('Modelo salvo e exportado como "best_model_trained.joblib"')
except Exception as e:
    print(f'Algo deu errado durante o saving e exportação do modelo. Rode o código de treino novamente | {e}')