from providers.split import leave_one_out_split

import argparse
import pandas as pd


def main(args):
    df = pd.read_csv(args.path + 'Data.csv')

    df_train, df_test = leave_one_out_split(df, args.user_id, 0.3, random_state=args.seed)

    if args.validation:
        df_train, df_valid = leave_one_out_split(df_train, args.user_id, 0.3, random_state=args.seed)
        df_valid.to_csv(args.path + 'Valid.csv')

    df_train.to_csv(args.path + 'Train.csv')
    df_test.to_csv(args.path + 'Test.csv')


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="DatasetPreSplit")

    parser.add_argument('-disable-validation', dest='validation', action='store_false')
    parser.add_argument('-d', dest='path', default="data/beer/advocate/")
    parser.add_argument('-s', dest='seed', default=8292)
    parser.add_argument('-u', dest='user_id', default="UserIndex")
    args = parser.parse_args()

    main(args)
