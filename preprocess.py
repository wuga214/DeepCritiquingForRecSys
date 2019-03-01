import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
import sparse
from utils.io import get_dataframe_json
from providers.bow import get_bow_dataframe
from providers.split import split_seed_randomly
from providers.subset import getsubset
from utils.argcheck import check_float_positive, check_int_positive, shape, ratio



def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Load Data
    progress.section("Loading Data")
    df = getsubset(args.path+args.name, args.user_col, args.item_col, args.review_col, [100, 2000], [100, 2000], 50)
    # df = get_dataframe_json(args.path+args.name)
    print("Number of Records: {0}".format(len(df)))
    print("Sparsity: {0}".format(float(len(df))/(df[args.user_col].nunique()*df[args.item_col].nunique())))

    progress.section("Tensorfy")
    df, topk, item_map = get_bow_dataframe(df, args.user_col, args.item_col, args.review_col,
                                           args.rating_col, args.topk, implicit=False)

    df.to_csv(args.path+'data.csv')
    item_map.to_csv(args.path+'item_index.csv')

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Preprocessing")

    parser.add_argument('-ratio', dest='ratio', type=ratio, default='0.8,0.1,0.1')
    parser.add_argument('-uid', dest='user_col', default='reviewerID')
    parser.add_argument('-iid', dest='item_col', default='asin')
    parser.add_argument('-review', dest='review_col', default='reviewText')
    parser.add_argument('-rating', dest='rating_col', default='overall')
    parser.add_argument('-topk', dest='topk', default=50)
    parser.add_argument('-path', dest='path', default='data/video/')
    parser.add_argument('-name', dest='name', default='Amazon_Instant_Video_5.json')
    args = parser.parse_args()

    main(args)