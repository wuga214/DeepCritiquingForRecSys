from utils.reformat import to_sparse_matrix
from utils.split import leave_one_out_split

import argparse
import pandas as pd
import scipy.sparse as sparse


def main(args):
    df = pd.read_csv(args.data_dir + 'Data.csv')

    num_users = df[args.user_col].nunique()
    num_items = df[args.item_col].nunique()

    df_train, df_test = leave_one_out_split(df, args.user_col, 0.2, random_state=args.seed)

    if args.enable_validation:
        df_train, df_valid = leave_one_out_split(df_train, args.user_col, 0.2, random_state=args.seed)

        df_valid.to_csv(args.data_dir + 'Valid.csv')
        R_valid = to_sparse_matrix(df_valid, num_users, num_items, args.user_col, args.item_col, args.rating_col)
        sparse.save_npz(args.data_dir+'Rvalid.npz', R_valid)

    df_train.to_csv(args.data_dir + 'Train.csv')
    R_train = to_sparse_matrix(df_train, num_users, num_items, args.user_col, args.item_col, args.rating_col)
    sparse.save_npz(args.data_dir + 'Rtrain.npz', R_train)

    df_test.to_csv(args.data_dir + 'Test.csv')
    R_test = to_sparse_matrix(df_test, num_users, num_items, args.user_col, args.item_col, args.rating_col)
    sparse.save_npz(args.data_dir + 'Rtest.npz', R_test)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Dataset Split")

    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/",
                        help='Directory path to the dataset. (default: %(default)s)')

    parser.add_argument('--disable_validation', dest='enable_validation',
                        action='store_false',
                        help='Boolean flag indicating if validation is disabled.')

    parser.add_argument('--item_col', dest='item_col', default="ItemIndex",
                        help='Item column name in the dataset. (default: %(default)s)')

    parser.add_argument('--rating_col', dest='rating_col', default="Binary",
                        help='Rating column name in the dataset. (default: %(default)s)')

    parser.add_argument('--seed', dest='seed', default=8292,
                        help='Seed used to split dataset. (default: %(default)s)')

    parser.add_argument('--user_col', dest='user_col', default="UserIndex",
                        help='User column name in the dataset. (default: %(default)s)')

    args = parser.parse_args()

    main(args)
