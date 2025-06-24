import sys
sys.path.append('loadData')
from load_data import load_data_df_x_y
from features import get_cat_features, get_num_features
from pipeline import build_model_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from evaluate import evaluate_model
from tuning import get_best_estimators
from sklearn.preprocessing import OrdinalEncoder

df, x, y = load_data_df_x_y(
    path='train_users_clean_upd.csv', 
    target_col=['country_destination'], 
    remove_col=['id', 'Unnamed: 0', 'timestamp_first_active'])

cat_cols = get_cat_features(x)
num_cols = get_num_features(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = build_model_pipeline(cat_cols=cat_cols,
                              num_cols=num_cols, 
                              hidden_layers_set=(80,40,20), 
                              learning_rate_init=0.1,
                              epochs=3100)

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=20)
y_train_encoded = ordinal_encoder.fit_transform(y_train)
y_test_encoded = ordinal_encoder.transform(y_test)

model.fit(x_train, y_train)

pred_list = model.predict(x_test)

print(f'acurácia: {evaluate_model(y_test, pred_list)}')

#print(pred_list[:5])

