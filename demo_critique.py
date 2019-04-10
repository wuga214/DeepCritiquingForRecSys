from metrics.general_performance import evaluate, evaluate_explanation
from predicts.topk import elementwisepredictor, predict_explanation
from providers.sampler import Negative_Sampler
from providers.split import leave_one_out_split
from tqdm import tqdm
from utils.argcheck import check_float_positive, check_int_positive, shape
from utils.modelnames import explanable_models
from utils.progress import WorkSplitter, inhour
from utils.reformat import to_sparse_matrix
from critique.critique import critique_keyphrase
from metrics.critiquing_performance import falling_rank

import argparse
import ast
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
    print("Train Batch Size: {}".format(args.train_batch_size))
    print("Predict Batch Size: {}".format(args.predict_batch_size))
    print("Negative Sampling Size: {}".format(args.negative_sampling_size))
    print("Number of Keyphrases in Explanation: {}".format(args.topk_key))
    print("Validation: {}".format(args.validation))

    progress.section("Loading Data")
    num_users = pd.read_csv(args.path + args.user_col + '.csv')[args.user_col].nunique()
    num_items = pd.read_csv(args.path + args.item_col + '.csv')[args.item_col].nunique()
    print("Dataset U-I Dimensions: ({}, {})".format(num_users, num_items))

    df_train = pd.read_csv(args.path + 'Train.csv')
    df_train = df_train[df_train[args.rating_col] == 1]
    df_train[args.key_col] = df_train[args.key_col].apply(ast.literal_eval)

    if args.validation:
        df_valid = pd.read_csv(args.path + 'Valid.csv')
    else:
        df_valid = pd.read_csv(args.path + 'Test.csv')

    keyPhrase = pd.read_csv(args.path + 'KeyPhrases.csv')[args.phrase].values

    progress.section("Initializing Negative Sampler")
    negative_sampler = Negative_Sampler(df_train[[args.user_col, args.item_col, args.key_col]],
                                        args.user_col, args.item_col, args.rating_col, args.key_col,
                                        num_items, batch_size=args.train_batch_size,
                                        num_keys=len(keyPhrase),
                                        negative_sampling_size=args.negative_sampling_size)

    progress.section("Train")
    model = explanable_models[args.model](num_users=num_users,
                                          num_items=num_items,
                                          text_dim=len(keyPhrase),
                                          embed_dim=args.rank,
                                          num_layers=1,
                                          batch_size=args.train_batch_size,
                                          negative_sampler=negative_sampler,
                                          lamb=args.lamb,
                                          learning_rate=args.alpha)

    model.train_model(df_train, epoch=args.epoch)
    """
    progress.section("Predict")
    prediction, explanation = elementwisepredictor(model, df_train, args.user_col,
                                                   args.item_col, args.topk,
                                                   batch_size=args.predict_batch_size,
                                                   explain=True, key_names=keyPhrase,
                                                   topk_key=args.topk_key)
    """
    # metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
    #
    # R_valid = to_sparse_matrix(df_valid, num_users, num_items, args.user_col,
    #                            args.item_col, args.rating_col)
    #
    # result = evaluate(prediction, R_valid, metric_names, [args.topk])
    #
    # print("-- General Performance")
    # for metric in result.keys():
    #     print("{0}:{1}".format(metric, result[metric]))
    #
    # df_valid_explanation = predict_explanation(model, df_valid, args.user_col,
    #                                            args.item_col, topk_key=args.topk_key)
    #
    # explanation_result = evaluate_explanation(df_valid_explanation, df_valid,
    #                                           ['Recall', 'Precision'], [args.topk_key])
    #
    # print("-- Explanation Performance")
    # for metric in explanation_result.keys():
    #     print("{0}:{1}".format(metric, explanation_result[metric]))

    random_users = np.random.choice(num_users, 100)
    falling_rank_result = []
    for user in tqdm(random_users):
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
    parser.add_argument('-b', dest='rating_col', default="Binary")
    parser.add_argument('-d', dest='path', default="data/beer/")
    parser.add_argument('-e', dest='epoch', type=check_int_positive, default=1)
    parser.add_argument('-i', dest='item_col', default="ItemIndex")
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=10)
    parser.add_argument('-key-col', dest='key_col', default="keyVector")
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=1.0)
    parser.add_argument('-m', dest='model', default="NCF")
    parser.add_argument('-negative-sampling-size', dest='negative_sampling_size', type=check_int_positive, default=1)
    parser.add_argument('-p', dest='phrase', default="Phrases")
    parser.add_argument('-predict-batch-size', dest='predict_batch_size', type=check_int_positive, default=128)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=200)
    parser.add_argument('-topk-key', dest='topk_key', type=check_int_positive, default=10)
    parser.add_argument('-train-batch-size', dest='train_batch_size', type=check_int_positive, default=128)
    parser.add_argument('-u', dest='user_col', default="UserIndex")
    args = parser.parse_args()

    main(args)

