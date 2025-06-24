

def get_cat_features(df):
    return df.select_dtypes(include='object').columns.to_list()

def get_num_features(df):
    return df.select_dtypes(include='number').columns.to_list()
