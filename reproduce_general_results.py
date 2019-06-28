from experiment.general import general

import argparse
import ast
import pandas as pd


def main(args):

    num_users = pd.read_csv(args.data_dir + args.user_col + '.csv')[args.user_col].nunique()
    num_items = pd.read_csv(args.data_dir + args.item_col + '.csv')[args.item_col].nunique()

    df_train = pd.read_csv(args.data_dir + args.train_set)
    df_train = df_train[df_train[args.rating_col] == 1]
    df_train[args.keyphrase_vector_col] = df_train[args.keyphrase_vector_col].apply(ast.literal_eval)

    df_test = pd.read_csv(args.data_dir + args.test_set)

    keyphrase_names = pd.read_csv(args.data_dir + args.keyphrase_set)[args.keyphrase_col].values

    params = dict()
    params['tuning_result_path'] = args.tuning_result_path

    general(num_users,
            num_items,
            args.user_col,
            args.item_col,
            args.rating_col,
            args.keyphrase_vector_col,
            df_train,
            df_test,
            keyphrase_names,
            params,
            save_path=args.save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce Final General Recommendation Performance")

    parser.add_argument('--data_dir', dest='data_dir', default="data/CDsVinyl/")
    parser.add_argument('--item_col', dest='item_col', default="ItemIndex")
    parser.add_argument('--keyphrase', dest='keyphrase_set', default="KeyPhrases.csv")
    parser.add_argument('--keyphrase_col', dest='keyphrase_col', default="Phrases")
    parser.add_argument('--keyphrase_vector_col', dest='keyphrase_vector_col', default="keyVector")
    parser.add_argument('--rating_col', dest='rating_col', default="Binary")
    parser.add_argument('--save_path', dest='save_path', default="CD_final_result.csv")
    parser.add_argument('--test', dest='test_set', default="Test.csv")
    parser.add_argument('--train', dest='train_set', default="Train.csv")
    parser.add_argument('--tuning_result_path', dest='tuning_result_path', default="CDsVinyl")
    parser.add_argument('--user_col', dest='user_col', default="UserIndex")

    args = parser.parse_args()

    main(args)
