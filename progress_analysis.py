import argparse
import numpy as np
import pandas as pd
from experiment.converge import converge
from providers.split import leave_one_out_split
from utils.io import find_best_hyperparameters, load_yaml
from plots.rec_plots import show_training_progress


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_best_hyperparameters(table_path+args.param, 'NDCG')

    df_data = pd.read_csv(args.path + args.param + '/' + 'Data.csv')
    df_train, df_test = leave_one_out_split(df_data, 'UserIndex', 0.3)

    KeyPhrase = pd.read_csv(args.path + args.param + '/' + 'KeyPhrases.csv')['Phrases'].values

    results = converge(df_data, df_train, df_test, keyPhrase, df, table_path, args.name, epochs=50, gpu_on=args.gpu)

    show_training_progress(results, hue='model', metric='NDCG', name="epoch_vs_ndcg")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="progress_analysis")

    parser.add_argument('-gpu', dest='gpu', action='store_true')
    parser.add_argument('-type', dest='type', default='optimizer')
    parser.add_argument('-d', dest='path', default="data/")
    parser.add_argument('-n', dest='name', default="convergence_analysis.csv")
    parser.add_argument('-p', dest='param', default='Beer')
    args = parser.parse_args()

    main(args)

