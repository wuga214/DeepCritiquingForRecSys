import scipy.sparse as sparse
import argparse
import time
import numpy as np
import pandas as pd
from models.ncf import NCF
from models.incf import INCF
from models.cncf import CNCF
from models.vncf import VNCF
from models.ivncf import IVNCF
from models.cvncf import CVNCF
from predicts.topk import elementwisepredictor
from providers.split import leave_one_out_split
from utils.reformat import to_sparse_matrix
from utils.progress import WorkSplitter, inhour
from utils.argcheck import check_float_positive, check_int_positive, shape
from metrics.general_performance import evaluate

def main(args):
    # Progress bar
    progress = WorkSplitter()

    progress.section("Parameter Setting")

    df = pd.read_csv(args.path + 'Data.csv')

    df_train, df_valid = leave_one_out_split(df, 'UserIndex', 0.1)

    df_train.to_csv(args.path + 'Train.csv')
    df_valid.to_csv(args.path + 'Valid.csv')

    incf = CVNCF(num_users=df['UserIndex'].nunique(),
                 num_items=df['ItemIndex'].nunique(),
                 label_dim=1,
                 text_dim=100,
                 embed_dim=args.rank,
                 num_layers=1,
                 batch_size=2000,
                 lamb=args.lamb)

    incf.train_model(df_train, epoch=args.epoch)

    df_key = pd.read_csv(args.path + 'KeyPhrases.csv')
    keyPhrase = np.array(df_key['Phrases'].tolist())

    prediction, explanation = elementwisepredictor(incf, df_train, 'UserIndex', 'ItemIndex',
                                                   args.topk, batch_size=1000, explain=True, key_names=keyPhrase)

    metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision']

    R_valid = to_sparse_matrix(df_valid, df['UserIndex'].nunique(), df['ItemIndex'].nunique(),
                               'UserIndex', 'ItemIndex', 'Binary')

    result = evaluate(prediction, R_valid, metric_names, [args.topk])

    print("-")
    for metric in result.keys():
        print("{0}:{1}".format(metric, result[metric]))

    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="INCF")

    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=100.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=10.0)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-d', dest='path', default="data/beer/advocate/")
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=10)
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)