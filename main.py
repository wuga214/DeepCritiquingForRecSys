import sparse
import argparse
import time
from utils.progress import WorkSplitter, inhour
from utils.argcheck import check_float_positive, check_int_positive, shape

from models.interpautorec import InterpretableAutoRec
from predicts.itembased import topk_predict
from metrics.general_performance import evaluate

def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")

    progress.section("Load Data")
    Rtrain = sparse.load_npz(args.path + args.train)
    Rvalid = sparse.load_npz(args.path + args.valid)

    n, m, k = Rtrain.shape # Item, User, Text Feature

    progress.section("Train Model")
    iae = InterpretableAutoRec([m, k], args.rank, 100, args.lamb) # I-AutoRec
    iae.train_model(Rtrain, args.epoch)

    progress.section("Prediction")
    predicted = topk_predict(iae, Rvalid, args.topk, n, m, k)

    metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision']
    result = evaluate(predicted[:, :, 0], Rtrain[:, :, 0].tocsr().transpose(), metric_names, [args.topk])
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
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)