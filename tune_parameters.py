from experiment.tuning import explanation_parameter_tuning, hyper_parameter_tuning
from utils.io import load_yaml
from utils.modelnames import models

import argparse
import ast
import pandas as pd


def main(args):
    params = load_yaml(args.parameters)

    params['models'] = {params['models']: models[params['models']]}

    num_users = pd.read_csv(args.data_dir + args.user_col + '.csv')[args.user_col].nunique()
    num_items = pd.read_csv(args.data_dir + args.item_col + '.csv')[args.item_col].nunique()

    df_train = pd.read_csv(args.data_dir + args.train_set)
    df_train = df_train[df_train[args.rating_col] == 1]
    df_train[args.keyphrase_vector_col] = df_train[args.keyphrase_vector_col].apply(ast.literal_eval)

    df_valid = pd.read_csv(args.data_dir + args.valid_set)

    keyphrase_names = pd.read_csv(args.data_dir + args.keyphrase_set)[args.keyphrase_col].values

    if args.explanation:
        explanation_parameter_tuning(num_users,
                                     num_items,
                                     args.user_col,
                                     args.item_col,
                                     args.rating_col,
                                     args.keyphrase_vector_col,
                                     df_train,
                                     df_valid,
                                     keyphrase_names,
                                     params,
                                     save_path=args.save_path)
    else:
        hyper_parameter_tuning(num_users,
                               num_items,
                               args.user_col,
                               args.item_col,
                               args.rating_col,
                               args.keyphrase_vector_col,
                               df_train,
                               df_valid,
                               keyphrase_names,
                               params,
                               save_path=args.save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/")
    parser.add_argument('--explanation', dest='explanation', action="store_true")
    parser.add_argument('--item_col', dest='item_col', default="ItemIndex")
    parser.add_argument('--keyphrase', dest='keyphrase_set', default="KeyPhrases.csv")
    parser.add_argument('--keyphrase_col', dest='keyphrase_col', default="Phrases")
    parser.add_argument('--keyphrase_vector_col', dest='keyphrase_vector_col', default="keyVector")
    parser.add_argument('--parameters', dest='parameters', default='config/default.yml')
    parser.add_argument('--rating_col', dest='rating_col', default="Binary")
    parser.add_argument('--save_path', dest='save_path', default="ncf_tuning.csv")
    parser.add_argument('--train', dest='train_set', default="Train.csv")
    parser.add_argument('--user_col', dest='user_col', default="UserIndex")
    parser.add_argument('--valid', dest='valid_set', default="Valid.csv")

    args = parser.parse_args()

    main(args)
