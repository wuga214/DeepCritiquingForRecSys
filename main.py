import sparse
import argparse
import time
import pandas as pd
from models.incf import INCF
from predicts.topk import elementwisepredictor
from utils.reformat import to_sparse_matrix
from utils.progress import WorkSplitter, inhour
from utils.argcheck import check_float_positive, check_int_positive, shape
from metrics.general_performance import evaluate

def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")

    df = pd.read_csv(args.path + 'data.csv')

    # df['Value'] = (df['Value']>3)*1
    # df.to_csv(args.path + 'data.csv', index=False)

    data = df.as_matrix()

    incf = INCF(num_users=df['UserID'].nunique(),
                 num_items=df['ItemID'].nunique(),
                 label_dim=1,
                 text_dim=data.shape[1]-3,
                 embed_dim=args.rank,
                 num_layers=2,
                 batch_size=1000,
                 lamb=args.lamb)

    incf.train_model(data, epoch=args.epoch)

    prediction = elementwisepredictor(incf, df, df['UserID'].nunique(), df['ItemID'].nunique(),
                                      args.topk, batch_size=1000)

    metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision']

    R_valid = to_sparse_matrix(df, df['UserID'].nunique(), df['ItemID'].nunique(), 'UserID', 'ItemID', 'Value', 3)

    result = evaluate(prediction[:, :, 1], R_valid, metric_names, [args.topk])

    print("-")
    for metric in result.keys():
        print("{0}:{1}".format(metric, result[metric]))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=100.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=1)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-d', dest='path', default="data/video/")
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=500)
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)