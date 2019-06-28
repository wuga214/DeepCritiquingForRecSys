from experiment.convergence import converge, explanation_converge
from utils.argcheck import check_int_positive
from utils.io import find_best_hyperparameters, load_yaml
from utils.plot import show_training_progress

import argparse
import ast
import pandas as pd


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    if args.explanation:
        df = find_best_hyperparameters(table_path + args.tuning_result_path, 'NDCG@10')
    else:
        df = find_best_hyperparameters(table_path + args.tuning_result_path, 'NDCG')

    num_users = pd.read_csv(args.data_dir + args.dataset_name + '/' + args.user_col + '.csv')[args.user_col].nunique()
    num_items = pd.read_csv(args.data_dir + args.dataset_name + '/' + args.item_col + '.csv')[args.item_col].nunique()

    df_train = pd.read_csv(args.data_dir + args.dataset_name + '/' + args.train_set)
    df_train = df_train[df_train[args.rating_col] == 1]
    df_train[args.keyphrase_vector_col] = df_train[args.keyphrase_vector_col].apply(ast.literal_eval)

    df_test = pd.read_csv(args.data_dir + args.dataset_name + '/' + args.test_set)

    keyphrase_names = pd.read_csv(args.data_dir + args.dataset_name + '/' + args.keyphrase_set)[args.keyphrase_col].values

    if args.explanation:
        results = explanation_converge(num_users,
                                       num_items,
                                       args.user_col,
                                       args.item_col,
                                       args.rating_col,
                                       args.keyphrase_vector_col,
                                       df_train,
                                       df_test,
                                       keyphrase_names,
                                       df,
                                       table_path,
                                       args.save_path,
                                       epoch=args.epoch)
    else:
        results = converge(num_users,
                           num_items,
                           args.user_col,
                           args.item_col,
                           args.rating_col,
                           args.keyphrase_vector_col,
                           df_train,
                           df_test,
                           keyphrase_names,
                           df,
                           table_path,
                           args.save_path,
                           epoch=args.epoch)

    show_training_progress(results, hue='model', metric='NDCG', name="epoch_vs_ndcg")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Progress Analysis")

    parser.add_argument('--data_dir', dest='data_dir', default="data/")
    parser.add_argument('--dataset_name', dest='dataset_name', default='beer')
    parser.add_argument('--epoch', dest='epoch', type=check_int_positive, default=300)
    parser.add_argument('--explanation', dest='explanation', action="store_true")
    parser.add_argument('--item_col', dest='item_col', default="ItemIndex")
    parser.add_argument('--keyphrase', dest='keyphrase_set', default="KeyPhrases.csv")
    parser.add_argument('--keyphrase_col', dest='keyphrase_col', default="Phrases")
    parser.add_argument('--keyphrase_vector_col', dest='keyphrase_vector_col', default="keyVector")
    parser.add_argument('--rating_col', dest='rating_col', default="Binary")
    parser.add_argument('--save_path', dest='save_path', default="convergence_analysis.csv")
    parser.add_argument('--test', dest='test_set', default="Test.csv")
    parser.add_argument('--train', dest='train_set', default="Train.csv")
    parser.add_argument('--tuning_result_path', dest='tuning_result_path', default="beer/")
    parser.add_argument('--user_col', dest='user_col', default="UserIndex")

    args = parser.parse_args()

    main(args)
