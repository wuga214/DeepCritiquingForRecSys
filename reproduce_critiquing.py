from experiment.critiquing import critiquing
from utils.argcheck import check_float_positive, check_int_positive, shape
import ast
import argparse
import pandas as pd


def main(args):

    num_users = pd.read_csv(args.path + args.user_id + '.csv')[args.user_id].nunique()
    num_items = pd.read_csv(args.path + args.item_id + '.csv')[args.item_id].nunique()

    df_train = pd.read_csv(args.path + 'Train.csv')
    df_train = df_train[df_train[args.rating_col] == 1]
    df_train[args.key_col] = df_train[args.key_col].apply(ast.literal_eval)

    keyPhrase = pd.read_csv(args.path + 'KeyPhrases.csv')['Phrases'].values

    params = dict()
    params['problem'] = args.problem

    critiquing(num_users, num_items, df_train, keyPhrase, params, args.critiquing,
               save_path=args.save, figure_path=args.figure)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('-b', dest='rating_col', default="Binary")
    parser.add_argument('-d', dest='path', default="data/CDsVinyl/")
    parser.add_argument('-i', dest='item_id', default="ItemIndex")
    parser.add_argument('-key-col', dest='key_col', default="keyVector")
    parser.add_argument('-p', dest='problem', default="CDsVinyl")
    parser.add_argument('-s', dest='save', default="CD_Critiquing.csv")
    parser.add_argument('-f', dest='figure', default="CD_Falling_Rank")
    parser.add_argument('-u', dest='user_id', default="UserIndex")
    parser.add_argument('-c', dest='critiquing', type=check_int_positive, default=300)
    args = parser.parse_args()

    main(args)