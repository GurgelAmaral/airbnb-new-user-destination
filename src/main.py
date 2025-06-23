import sys
sys.path.append('loadData')
from load_data import load_data_df_x_y
from features import get_cat_features
from pipeline import build_model_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from evaluate import evaluate_model
from tuning import get_best_estimators


df, x, y = load_data_df_x_y(
    path='train_users_clean_upd.csv', 
    target_col=['country_destination'], 
    remove_col=['id', 'Unnamed: 0', 'timestamp_first_active'])

cat_cols = get_cat_features(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = build_model_pipeline(cat_cols=cat_cols)


model

best_model = get_best_estimators(model=model, x_train=x_train, y_train=y_train, cv=2)


pred_list = best_model.predict(x_test)

print(f'acurácia: {evaluate_model(y_test, pred_list)}')

#print(pred_list[:5])

