import pandas as pd


def load_data_df_x_y(path, target_col=None, remove_col=None):
    if target_col is None:
        raise ValueError('Coluna target deve ser especificada')
    
    df = pd.read_csv(path)

    if remove_col is not None:
        x = df.drop(columns=remove_col).drop(columns=target_col)
    else:
        x = df.drop(columns=target_col)

    y = df[target_col]
    
    return df, x, y