import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import get_dataframe_json
from providers.bow import get_bow_tensor
from utils.argcheck import check_float_positive, check_int_positive, shape



def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Load Data
    progress.section("Loading Data")
    df = get_dataframe_json(args.path)

    df = df[:300]

    topk, tensor, keywords = get_bow_tensor(df, args.user_col, args.item_col, args.review_col,
                                            args.rating_col, [], args.topk, implicit=True)


    import ipdb;ipdb.set_trace()




if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Preprocessing")

    # parser.add_argument('--disable-item-item', dest='item', action='store_false')
    # parser.add_argument('--disable-validation', dest='validation', action='store_false')
    # parser.add_argument('-i', dest='iter', type=check_int_positive, default=1)
    parser.add_argument('-uid', dest='user_col', default='reviewerID')
    parser.add_argument('-iid', dest='item_col', default='asin')
    parser.add_argument('-review', dest='review_col', default='reviewText')
    parser.add_argument('-rating', dest='rating_col', default='overall')
    parser.add_argument('-topk', dest='topk', default=50)
    parser.add_argument('-path', dest='path', default='data/video/Amazon_Instant_Video_5.json')
    args = parser.parse_args()

    main(args)