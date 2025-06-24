from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def build_model_pipeline(cat_cols=None, num_cols=None, n_estimators=200, max_depth=None):
    #verificação se cat_cols está nulo
    if cat_cols is None or num_cols is None:
        raise ValueError('cat_cols e num_cols devem ser preenchidos deve ser preenchido')

    #criação de column_transformer e automação do processo de ordinal encoding paraa variáveis categóricas
    transf = ColumnTransformer(
        transformers=[
            ('cat_trans', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
            ('num_trans', StandardScaler(), num_cols)
        ],
    
        remainder='drop'
    )

    #pipeline do modelo 
    model_pipeline = Pipeline([
        ('prep', transf),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth))
    ])

    return model_pipeline

