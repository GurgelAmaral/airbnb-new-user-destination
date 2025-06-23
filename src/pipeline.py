from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model_pipeline(cat_cols=None, n_estimators=200, max_depth=None):
    #verificação se cat_cols está nulo
    if cat_cols is None:
        raise ValueError('cat_cols deve ser preenchido')

    #criação de column_transformer e automação do processo de ordinal encoding paraa variáveis categóricas
    transf = ColumnTransformer(
        transformers=[
            ('cat_trans', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ],

        #variáveis numéricas não passam por encoding e continuam como estão
        remainder='passthrough'
    )

    #pipeline do modelo 
    model_pipeline = Pipeline([
        ('prep', transf),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth))
    ])

    return model_pipeline

