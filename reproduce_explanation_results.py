from experiment.explanation import explain
from providers.split import leave_one_out_split
from utils.io import save_dataframe_csv, load_yaml
from utils.modelnames import explanable_models

import ast
import argparse
import numpy as np
import pandas as pd


def main(args):

    params = load_yaml(args.grid)

    num_users = pd.read_csv(args.path + args.user_id + '.csv')[args.user_id].nunique()
    num_items = pd.read_csv(args.path + args.item_id + '.csv')[args.item_id].nunique()

    df_train = pd.read_csv(args.path + 'Train.csv')
    df_train = df_train[df_train[args.rating_col] == 1]
    df_train[args.key_col] = df_train[args.key_col].apply(ast.literal_eval)

    df_valid = pd.read_csv(args.path + 'Valid.csv')

    keyPhrase = pd.read_csv(args.path + 'KeyPhrases.csv')['Phrases'].values

    explain(num_users, num_items, df_train, df_valid, keyPhrase, params,
            load_path=args.load, save_path=args.save, gpu_on=args.gpu)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('-gpu', dest='gpu', action='store_true')
    parser.add_argument('-b', dest='rating_col', default="Binary")
    parser.add_argument('-d', dest='path', default="data/CDsVinyl/")
    parser.add_argument('-i', dest='item_id', default="ItemIndex")
    parser.add_argument('-key-col', dest='key_col', default="keyVector")
    parser.add_argument('-l', dest='load', default="explanation/CDsVinyl/hyper_parameters.csv")
    parser.add_argument('-s', dest='save', default="explanation/CDsVinyl/explanation.csv")
    parser.add_argument('-u', dest='user_id', default="UserIndex")
    parser.add_argument('-y', dest='grid', default='config/explanation.yml')
    args = parser.parse_args()

    main(args)