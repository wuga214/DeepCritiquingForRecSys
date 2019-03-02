import sparse
import argparse
import time
import pandas as pd
from models.incf import INCF
from utils.progress import WorkSplitter, inhour
from utils.argcheck import check_float_positive, check_int_positive, shape
from metrics.general_performance import evaluate

def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")

    df = pd.read_csv(args.path + 'data.csv')

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




if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=100.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=1)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-d', dest='path', default="data/video/")
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)