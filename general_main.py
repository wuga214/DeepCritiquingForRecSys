from evaluation.general_performance import evaluate, evaluate_explanation
from prediction.predictor import predict_elementwise, predict_explanation
from utils.argcheck import check_float_positive, check_int_positive
from utils.modelnames import models
from utils.progress import WorkSplitter
from utils.reformat import to_sparse_matrix
from utils.sampler import Negative_Sampler

import argparse
import ast
import pandas as pd


def main(args):
    # Progress bar
    progress = WorkSplitter()

    progress.section("Parameter Setting")
    print("Data Directory: {}".format(args.data_dir))
    print("Algorithm: {}".format(args.model))
    print("Learning Rate: {}".format(args.learning_rate))
    print("Epoch: {}".format(args.epoch))
    print("Number of Top Items Evaluated in Recommendation: {}".format(args.topk))
    print("Lambda: {}".format(args.lamb))
    print("Rank: {}".format(args.rank))
    print("Train Batch Size: {}".format(args.train_batch_size))
    print("Predict Batch Size: {}".format(args.predict_batch_size))
    print("Negative Sampling Size: {}".format(args.negative_sampling_size))
    print("Number of Keyphrases Evaluated in Explanation: {}".format(args.topk_keyphrase))
    print("Enable Validation: {}".format(args.enable_validation))

    progress.section("Load Data")
    num_users = pd.read_csv(args.data_dir + args.user_col + '.csv')[args.user_col].nunique()
    num_items = pd.read_csv(args.data_dir + args.item_col + '.csv')[args.item_col].nunique()
    print("Dataset U-I Dimensions: ({}, {})".format(num_users, num_items))

    df_train = pd.read_csv(args.data_dir + args.train_set)
    df_train = df_train[df_train[args.rating_col] == 1]
    df_train[args.keyphrase_vector_col] = df_train[args.keyphrase_vector_col].apply(ast.literal_eval)

    if args.enable_validation:
        df_valid = pd.read_csv(args.data_dir + args.valid_set)
    else:
        df_valid = pd.read_csv(args.data_dir + args.test_set)

    keyphrase_names = pd.read_csv(args.data_dir + args.keyphrase_set)[args.keyphrase_col].values
    num_keyphrases = len(keyphrase_names)

    progress.section("Initialize Negative Sampler")
    negative_sampler = Negative_Sampler(df_train[[args.user_col,
                                                  args.item_col,
                                                  args.keyphrase_vector_col]],
                                        args.user_col,
                                        args.item_col,
                                        args.rating_col,
                                        args.keyphrase_vector_col,
                                        num_items,
                                        batch_size=args.train_batch_size,
                                        num_keyphrases=num_keyphrases,
                                        negative_sampling_size=args.negative_sampling_size)

    progress.section("Train")
    model = models[args.model](num_users=num_users,
                               num_items=num_items,
                               text_dim=num_keyphrases,
                               embed_dim=args.rank,
                               num_layers=1,
                               negative_sampler=negative_sampler,
                               lamb=args.lamb,
                               learning_rate=args.learning_rate)

    model.train_model(df_train,
                      user_col=args.user_col,
                      item_col=args.item_col,
                      rating_col=args.rating_col,
                      epoch=args.epoch)

    progress.section("Predict")
    prediction, explanation = predict_elementwise(model,
                                                  df_train,
                                                  args.user_col,
                                                  args.item_col,
                                                  args.topk,
                                                  batch_size=args.predict_batch_size,
                                                  enable_explanation=True,
                                                  keyphrase_names=keyphrase_names,
                                                  topk_keyphrase=args.topk_keyphrase)

    metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

    R_valid = to_sparse_matrix(df_valid,
                               num_users,
                               num_items,
                               args.user_col,
                               args.item_col,
                               args.rating_col)

    result = evaluate(prediction, R_valid, metric_names, [args.topk])

    print("-- General Performance")
    for metric in result.keys():
        print("{}:{}".format(metric, result[metric]))

    df_valid_explanation = predict_explanation(model,
                                               df_valid,
                                               args.user_col,
                                               args.item_col,
                                               topk_keyphrase=args.topk_keyphrase)

    explanation_result = evaluate_explanation(df_valid_explanation,
                                              df_valid,
                                              ['Recall', 'Precision'],
                                              [args.topk_keyphrase],
                                              args.user_col,
                                              args.item_col,
                                              args.rating_col,
                                              args.keyphrase_vector_col)

    print("-- Explanation Performance")
    for metric in explanation_result.keys():
        print("{}:{}".format(metric, explanation_result[metric]))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Deep Language-based Critiquing")

    parser.add_argument('--data_dir', dest='data_dir', default="data/beer/",
                        help='Directory path to the dataset. (default: %(default)s)')

    parser.add_argument('--disable_validation', dest='enable_validation',
                        action='store_false',
                        help='Boolean flag indicating if validation is disabled.')

    parser.add_argument('--epoch', dest='epoch', default=1,
                        type=check_int_positive,
                        help='The number of epochs used in training models. (default: %(default)s)')

    parser.add_argument('--item_col', dest='item_col', default="ItemIndex",
                        help='Item column name in the dataset. (default: %(default)s)')

    parser.add_argument('--keyphrase', dest='keyphrase_set', default="KeyPhrases.csv",
                        help='Keyphrase set csv file. (default: %(default)s)')

    parser.add_argument('--keyphrase_col', dest='keyphrase_col', default="Phrases",
                        help='Keyphrase column name in the dataset. (default: %(default)s)')

    parser.add_argument('--keyphrase_vector_col', dest='keyphrase_vector_col', default="keyVector",
                        help='Keyphrase vector column name in the dataset. (default: %(default)s)')

    parser.add_argument('--lambda', dest='lamb', default=1.0,
                        type=check_float_positive,
                        help='Regularizer strength used in models. (default: %(default)s)')

    parser.add_argument('--learning_rate', dest='learning_rate', default=0.0001,
                        type=check_float_positive,
                        help='Learning rate used in training models. (default: %(default)s)')

    parser.add_argument('--model', dest='model', default="NCF",
                        help='Model currently using. (default: %(default)s)')

    parser.add_argument('--negative_sampling_size', dest='negative_sampling_size', default=5,
                        type=check_int_positive,
                        help='The number of negative sampling. (default: %(default)s)')

    parser.add_argument('--predict_batch_size', dest='predict_batch_size', default=128,
                        type=check_int_positive,
                        help='Batch size used in prediction. (default: %(default)s)')

    parser.add_argument('--rank', dest='rank', default=200,
                        type=check_int_positive,
                        help='Latent dimension. (default: %(default)s)')

    parser.add_argument('--rating_col', dest='rating_col', default="Binary",
                        help='Rating column name in the dataset. (default: %(default)s)')

    parser.add_argument('--test', dest='test_set', default="Test.csv",
                        help='Test set csv file. (default: %(default)s)')

    parser.add_argument('--topk', dest='topk', default=10,
                        type=check_int_positive,
                        help='The number of items being recommended at top. (default: %(default)s)')

    parser.add_argument('--topk_keyphrase', dest='topk_keyphrase', default=10,
                        type=check_int_positive,
                        help='The number of keyphrases being recommended at top. (default: %(default)s)')

    parser.add_argument('--train', dest='train_set', default="Train.csv",
                        help='Train set csv file. (default: %(default)s)')

    parser.add_argument('--train_batch_size', dest='train_batch_size', default=128,
                        type=check_int_positive,
                        help='Batch size used in training. (default: %(default)s)')

    parser.add_argument('--user_col', dest='user_col', default="UserIndex",
                        help='User column name in the dataset. (default: %(default)s)')

    parser.add_argument('--valid', dest='valid_set', default="Valid.csv",
                        help='Valid set csv file. (default: %(default)s)')

    args = parser.parse_args()

    main(args)
