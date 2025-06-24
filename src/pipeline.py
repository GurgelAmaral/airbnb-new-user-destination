from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def build_model_pipeline(cat_cols=None, 
                         num_cols=None, 
                         hidden_layers_set=(40,20),
                         activation_func='relu',
                         learning_rate_init=0.001,
                         solve_method='sgd',
                         epochs=300):
    
    #verificação se cat_cols está nulo
    if cat_cols is None or num_cols is None:
        raise ValueError('cat_cols e num_cols devem ser preenchidos como parâmetros da função build_model_pipeline')

    #criação de column_transformer e automação do processo de ordinal encoding para variáveis categóricas e StdScaler -> num
    #caso se ambas as colunas forem vazias
    if not cat_cols and not num_cols:
        raise ValueError('não há colunas para transformar')
    
    #se cat_cols for vazia
    elif not cat_cols:
        transf = ColumnTransformer(
            transformers=[
                ('num_trans', StandardScaler(), num_cols)
            ],
    
            remainder='drop'
        )

    #se num_cols for vazia
    elif not num_cols:
        transf = ColumnTransformer(
            transformers=[
                ('cat_trans', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
            ],
    
            remainder='drop'
        )
        
    #caso contrário
    else:
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
        ('classifier', MLPClassifier(
            hidden_layer_sizes=hidden_layers_set, 
            activation=activation_func, 
            learning_rate_init=learning_rate_init, 
            solver=solve_method,
            max_iter=epochs))
    ])

    return model_pipeline

