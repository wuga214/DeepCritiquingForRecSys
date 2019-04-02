from metrics.general_performance import evaluate, evaluate_explanation
from predicts.topk import elementwisepredictor, predict_explanation
from providers.split import leave_one_out_split
from tqdm import tqdm
from utils.argcheck import check_float_positive, check_int_positive, shape
from utils.modelnames import models
from utils.progress import WorkSplitter, inhour
from utils.reformat import to_sparse_matrix
from critique.critique import critique_keyphrase
from metrics.critiquing_performance import falling_rank
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import time


def main(args):
    # Progress bar
    progress = WorkSplitter()

    progress.section("Parameter Setting")
    print("Data Path: {}".format(args.path))
    print("Algorithm: {}".format(args.model))
    print("GPU: {}".format(args.gpu))
    print("Learning Rate: {}".format(args.alpha))
    print("Epoch: {}".format(args.epoch))
    print("Evaluation Ranking Topk: {}".format(args.topk))
    print("Lambda: {}".format(args.lamb))
    print("Rank: {}".format(args.rank))
    print("Validation: {}".format(args.validation))

    progress.section("Loading Data")
    df = pd.read_csv(args.path + 'Data.csv')

    num_users=df[args.user_id].nunique()
    num_items=df[args.item_id].nunique()
    print("Dataset U-I Dimensions: ({}, {})".format(num_users, num_items))

    df_train = pd.read_csv(args.path + 'Train.csv')
    if args.validation:
        df_valid = pd.read_csv(args.path + 'Valid.csv')
    else:
        df_valid = pd.read_csv(args.path + 'Test.csv')

    keyPhrase = pd.read_csv(args.path + 'KeyPhrases.csv')[args.phrase].values

    progress.section("Train")
    model = models[args.model](num_users=num_users,
                               num_items=num_items,
                               text_dim=len(keyPhrase),
                               embed_dim=args.rank,
                               num_layers=1,
                               batch_size=1024,
                               lamb=args.lamb,
                               learning_rate=args.alpha)

    model.train_model(df_train, epoch=args.epoch)
    """
    progress.section("Predict")
    prediction, explanation = elementwisepredictor(model, df_train, args.user_id,
                                                   args.item_id, args.topk,
                                                   batch_size=1024, explain=True,
                                                   key_names=keyPhrase,
                                                   topk_key=args.topk_key)
    """
    #
    # metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
    #
    # R_valid = to_sparse_matrix(df_valid, num_users, num_items, args.user_id,
    #                            args.item_id, args.rating)
    #
    # result = evaluate(prediction, R_valid, metric_names, [args.topk])
    #
    # print("-- General Performance")
    # for metric in result.keys():
    #     print("{0}:{1}".format(metric, result[metric]))
    #
    # df_valid_explanation = predict_explanation(model, df_valid, args.user_id,
    #                                            args.item_id, topk_key=args.topk_key)
    #
    # explanation_result = evaluate_explanation(df_valid_explanation, df_valid,
    #                                           ['Recall', 'Precision'])
    #
    # print("-- Explanation Performance")
    # for metric in explanation_result.keys():
    #     print("{0}:{1}".format(metric, explanation_result[metric]))

    falling_rank_result = []
    for user in tqdm(range(num_users)):
        r_b, r_f, k = critique_keyphrase(model, user, num_items, topk_key=10)
        falling_rank_result.append(falling_rank(r_b.tolist(), r_f.tolist(), k))
    mean_falling_rank = np.mean(falling_rank_result)
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="INCF")

    parser.add_argument('-gpu', dest='gpu', action='store_true')
    parser.add_argument('-disable-validation', dest='validation', action='store_false')
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=0.0001)
    parser.add_argument('-b', dest='rating', default="Binary")
    parser.add_argument('-d', dest='path', default="data/beer/advocate/")
    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1)
    parser.add_argument('-i', dest='item_id', default="ItemIndex")
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=10)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=1.0)
    parser.add_argument('-m', dest='model', default="NCF")
    parser.add_argument('-p', dest='phrase', default="Phrases")
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=200)
    parser.add_argument('-topk-key', dest='topk_key', type=check_int_positive, default=10)
    parser.add_argument('-u', dest='user_id', default="UserIndex")
    args = parser.parse_args()

    main(args)

