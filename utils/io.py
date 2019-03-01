import pandas as pd
import json
import ast
from tqdm import tqdm

def get_dataframe_json(path):
    try:
        return pd.read_json(path, lines=True)
    except:
        with open(path) as f:
            lines = f.readlines()

        df_list = []

        for line in tqdm(lines):
            big_dict = ast.literal_eval(line)
            import ipdb; ipdb.set_trace()
            df_list.append({k: big_dict[k] for k in ('asin', 'reviewerID', 'reviewText')})

        return pd.DataFrame(df_list)
