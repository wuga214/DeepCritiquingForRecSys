import ast
import os
import pandas as pd
import stat
import yaml


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name, index=False)


def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)


def find_best_hyperparameters(folder_path, metric):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df[metric+'_Score'] = df[metric].map(lambda x: ast.literal_eval(x)[0])
        best_settings.append(df.loc[df[metric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings).drop(metric+'_Score', axis=1)

    return df


def load_dataframe_folder(folder_path):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
    dfs = []
    for record in csv_files:
        df = pd.read_csv(record)
        dfs.append(df)

    df = pd.concat(dfs)

    return df


def get_file_names(folder_path, extension='.yml'):
    return [f for f in os.listdir(folder_path) if
            not (not os.path.isfile(os.path.join(folder_path, f)) or not f.endswith(extension))]


def write_file(folder_path, file_name, content, exe=False):
    full_path = folder_path+'/'+file_name
    with open(full_path, 'w') as the_file:
        the_file.write(content)

    if exe:
        st = os.stat(full_path)
        os.chmod(full_path, st.st_mode | stat.S_IEXEC)
