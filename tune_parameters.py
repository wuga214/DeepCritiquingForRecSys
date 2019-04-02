from experiment.tuning import hyper_parameter_tuning
from providers.split import leave_one_out_split
from utils.io import save_dataframe_csv, load_yaml
from utils.modelnames import models

import argparse
import numpy as np
import pandas as pd


def main(args):
    params = load_yaml(args.grid)

    params['models'] = {params['models']: models[params['models']]}

    df_data = pd.read_csv(args.path + 'Data.csv')
    df_train = pd.read_csv(args.path + 'Train.csv')
    df_valid = pd.read_csv(args.path + 'Valid.csv')

    keyPhrase = pd.read_csv(args.path + 'KeyPhrases.csv')['Phrases'].values

    hyper_parameter_tuning(df_data, df_train, df_valid, keyPhrase, params, save_path=args.name, gpu_on=args.gpu)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('-gpu', dest='gpu', action='store_true')
    parser.add_argument('-d', dest='path', default="data/Beer/")
    parser.add_argument('-m', dest='model', default="NCF")
    parser.add_argument('-n', dest='name', default="ncf_tuning.csv")
    parser.add_argument('-y', dest='grid', default='config/default.yml')
    args = parser.parse_args()

    main(args)

