#captura das colunas categóricas para uso no pipeline/columnTransformer
def get_cat_features(df):
    return df.select_dtypes(include='object').columns.to_list()

