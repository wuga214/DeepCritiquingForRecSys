from providers.split import leave_one_out_split
from utils.reformat import to_sparse_matrix
import scipy.sparse as sparse
import argparse
import pandas as pd


def main(args):
    df = pd.read_csv(args.path + 'Data.csv')

    num_users = df[args.user_id].nunique()
    num_items = df[args.item_id].nunique()

    df_train, df_test = leave_one_out_split(df, args.user_id, 0.2, random_state=args.seed)

    if args.validation:
        df_train, df_valid = leave_one_out_split(df_train, args.user_id, 0.2, random_state=args.seed)
        df_valid.to_csv(args.path + 'Valid.csv')
        R_valid = to_sparse_matrix(df_valid, num_users, num_items, args.user_id, args.item_id, args.rating)
        sparse.save_npz(args.path+'Rvalid.npz', R_valid)

    df_train.to_csv(args.path + 'Train.csv')
    R_train = to_sparse_matrix(df_train, num_users, num_items, args.user_id, args.item_id, args.rating)
    sparse.save_npz(args.path + 'Rtrain.npz', R_train)

    df_test.to_csv(args.path + 'Test.csv')
    R_test = to_sparse_matrix(df_test, num_users, num_items, args.user_id, args.item_id, args.rating)
    sparse.save_npz(args.path + 'Rtest.npz', R_test)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="DatasetPreSplit")

    parser.add_argument('-disable-validation', dest='validation', action='store_false')
    parser.add_argument('-d', dest='path', default="data/beer/advocate/")
    parser.add_argument('-s', dest='seed', default=8292)
    parser.add_argument('-u', dest='user_id', default="UserIndex")
    parser.add_argument('-i', dest='item_id', default="ItemIndex")
    parser.add_argument('-b', dest='rating', default="Binary")
    args = parser.parse_args()

    main(args)
