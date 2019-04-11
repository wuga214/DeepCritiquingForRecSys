import ast
import argparse
import numpy as np
import pandas as pd
from experiment.converge import converge, explanation_converge
from providers.split import leave_one_out_split
from utils.argcheck import check_float_positive, check_int_positive, shape
from utils.io import find_best_hyperparameters, load_yaml
from plots.rec_plots import show_training_progress


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_best_hyperparameters(table_path + args.param, 'NDCG')

    num_users = pd.read_csv(args.path + args.param + '/' + args.user_id + '.csv')[args.user_id].nunique()
    num_items = pd.read_csv(args.path + args.param + '/' + args.item_id + '.csv')[args.item_id].nunique()

    df_train = pd.read_csv(args.path + args.param + '/' + 'Train.csv')
    df_train = df_train[df_train[args.rating_col] == 1]
    df_train[args.key_col] = df_train[args.key_col].apply(ast.literal_eval)

    df_test = pd.read_csv(args.path + args.param + '/' + 'Test.csv')

    keyPhrase = pd.read_csv(args.path + args.param + '/' + 'KeyPhrases.csv')['Phrases'].values

    if args.explanation:
        results = explanation_converge(num_users, num_items, df_train, df_test, keyPhrase, df, table_path, args.name,
                                       epochs=args.epochs, gpu_on=args.gpu)
    else:
        results = converge(num_users, num_items, df_train, df_test, keyPhrase, df, table_path, args.name,
                           epochs=args.epochs, gpu_on=args.gpu)

    show_training_progress(results, hue='model', metric='NDCG', name="epoch_vs_ndcg")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="progress_analysis")

    parser.add_argument('-gpu', dest='gpu', action='store_true')
    parser.add_argument('-type', dest='type', default='optimizer')
    parser.add_argument('-b', dest='rating_col', default="Binary")
    parser.add_argument('-d', dest='path', default="data/")
    parser.add_argument('-e', dest='epochs', type=check_int_positive, default=200)
    parser.add_argument('-i', dest='item_id', default="ItemIndex")
    parser.add_argument('-key-col', dest='key_col', default="keyVector")
    parser.add_argument('-n', dest='name', default="convergence_analysis.csv")
    parser.add_argument('-p', dest='param', default='beer')
    parser.add_argument('-u', dest='user_id', default="UserIndex")
    parser.add_argument('--explanation', dest='explanation', action="store_true")

    args = parser.parse_args()

    main(args)

