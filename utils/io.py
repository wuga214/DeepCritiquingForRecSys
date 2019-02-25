import pandas as pd


def get_dataframe_json(path):
    return pd.read_json(path, lines=True)