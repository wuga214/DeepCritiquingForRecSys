import os

import pandas as pd
import json
import ast
from tqdm import tqdm
import yaml
import stat
from os import listdir
from os.path import isfile, join
from ast import literal_eval


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name, index=False)


def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)


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


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)


def find_best_hyperparameters(folder_path, meatric):
    csv_files = [join(folder_path, f) for f in listdir(folder_path)
                 if isfile(join(folder_path, f)) and f.endswith('.csv')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df[meatric+'_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
        best_settings.append(df.loc[df[meatric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings).drop(meatric+'_Score', axis=1)

    return df


def get_file_names(folder_path, extension='.yml'):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(extension)]

def write_file(folder_path, file_name, content, exe=False):
    full_path = folder_path+'/'+file_name
    with open(full_path, 'w') as the_file:
        the_file.write(content)

    if exe:
        st = os.stat(full_path)
        os.chmod(full_path, st.st_mode | stat.S_IEXEC)