from evaluation.general_performance import evaluate, evaluate_explanation
from prediction.predictor import predict_elementwise, predict_explanation
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
from utils.progress import WorkSplitter
from utils.reformat import to_sparse_matrix
from utils.sampler import Negative_Sampler

import pandas as pd
import tensorflow as tf


def hyper_parameter_tuning(num_users, num_items, user_col, item_col, rating_col, keyphrase_vector_col, df_train, df_valid, keyphrase_names, params, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    try:
        df = load_dataframe_csv(table_path, save_path)
    except:
        df = pd.DataFrame(columns=['model', 'rank', 'num_layers', 'train_batch_size', 'predict_batch_size',
                                   'lambda', 'topK', 'learning_rate', 'epoch', 'negative_sampling_size'])

    for algorithm in params['models']:

        for rank in params['rank']:

            for num_layers in params['num_layers']:

                for train_batch_size in params['train_batch_size']:

                    for predict_batch_size in params['predict_batch_size']:

                        for lamb in params['lambda']:

                            for learning_rate in params['learning_rate']:

                                for epoch in params['epoch']:

                                    for negative_sampling_size in params['negative_sampling_size']:

                                        if ((df['model'] == algorithm) &
                                            (df['rank'] == rank) &
                                            (df['num_layers'] == num_layers) &
                                            (df['train_batch_size'] == train_batch_size) &
                                            (df['predict_batch_size'] == predict_batch_size) &
                                            (df['lambda'] == lamb) &
                                            (df['learning_rate'] == learning_rate) &
                                            (df['epoch'] == epoch) &
                                            (df['negative_sampling_size'] == negative_sampling_size)).any():
                                            continue

                                        format = "model: {0}, rank: {1}, num_layers: {2}, " \
                                                 "train_batch_size: {3}, predict_batch_size: {4}, " \
                                                 "lambda: {5}, learning_rate: {6}, epoch: {7}, " \
                                                 "negative_sampling_size: {8}"
                                        progress.section(format.format(algorithm, rank, num_layers, train_batch_size,
                                                                       predict_batch_size, lamb, learning_rate, epoch,
                                                                       negative_sampling_size))

                                        progress.subsection("Initializing Negative Sampler")

                                        negative_sampler = Negative_Sampler(df_train[[user_col,
                                                                                      item_col,
                                                                                      keyphrase_vector_col]],
                                                                            user_col,
                                                                            item_col,
                                                                            rating_col,
                                                                            keyphrase_vector_col,
                                                                            num_items=num_items,
                                                                            batch_size=train_batch_size,
                                                                            num_keyphrases=len(keyphrase_names),
                                                                            negative_sampling_size=negative_sampling_size)

                                        model = params['models'][algorithm](num_users=num_users,
                                                                            num_items=num_items,
                                                                            text_dim=len(keyphrase_names),
                                                                            embed_dim=rank,
                                                                            num_layers=num_layers,
                                                                            negative_sampler=negative_sampler,
                                                                            lamb=lamb,
                                                                            learning_rate=learning_rate)

                                        progress.subsection("Training")

                                        model.train_model(df_train,
                                                          user_col,
                                                          item_col,
                                                          rating_col,
                                                          epoch=epoch)

                                        progress.subsection("Prediction")

                                        prediction, explanation = predict_elementwise(model,
                                                                                      df_train,
                                                                                      user_col,
                                                                                      item_col,
                                                                                      params['topK'][-1],
                                                                                      batch_size=predict_batch_size,
                                                                                      enable_explanation=True,
                                                                                      keyphrase_names=keyphrase_names)

                                        progress.subsection("Evaluation")

                                        R_valid = to_sparse_matrix(df_valid,
                                                                   num_users,
                                                                   num_items,
                                                                   user_col,
                                                                   item_col,
                                                                   rating_col)

                                        result = evaluate(prediction, R_valid, params['metric'], params['topK'])

                                        result_dict = {'model': algorithm,
                                                       'rank': rank,
                                                       'num_layers': num_layers,
                                                       'train_batch_size': train_batch_size,
                                                       'predict_batch_size': predict_batch_size,
                                                       'lambda': lamb,
                                                       'learning_rate': learning_rate,
                                                       'epoch': epoch,
                                                       'negative_sampling_size': negative_sampling_size}

                                        for name in result.keys():
                                            result_dict[name] = [round(result[name][0], 4),
                                                                 round(result[name][1], 4)]

                                        df = df.append(result_dict, ignore_index=True)

                                        model.sess.close()
                                        tf.reset_default_graph()

                                        save_dataframe_csv(df, table_path, save_path)


def explanation_parameter_tuning(num_users, num_items, user_col, item_col, rating_col, keyphrase_vector_col, df_train, df_valid, keyphrase_names, params, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    try:
        df = load_dataframe_csv(table_path, save_path)
    except:
        df = pd.DataFrame(columns=['model', 'rank', 'num_layers', 'train_batch_size', 'predict_batch_size',
                                   'lambda', 'topK', 'learning_rate', 'epoch', 'negative_sampling_size'])

    for algorithm in params['models']:

        for rank in params['rank']:

            for num_layers in params['num_layers']:

                for train_batch_size in params['train_batch_size']:

                    for predict_batch_size in params['predict_batch_size']:

                        for lamb in params['lambda']:

                            for learning_rate in params['learning_rate']:

                                for epoch in params['epoch']:

                                    for negative_sampling_size in params['negative_sampling_size']:

                                        if ((df['model'] == algorithm) &
                                            (df['rank'] == rank) &
                                            (df['num_layers'] == num_layers) &
                                            (df['train_batch_size'] == train_batch_size) &
                                            (df['predict_batch_size'] == predict_batch_size) &
                                            (df['lambda'] == lamb) &
                                            (df['learning_rate'] == learning_rate) &
                                            (df['epoch'] == epoch) &
                                            (df['negative_sampling_size'] == negative_sampling_size)).any():
                                            continue

                                        format = "model: {0}, rank: {1}, num_layers: {2}, " \
                                                 "train_batch_size: {3}, predict_batch_size: {4}, " \
                                                 "lambda: {5}, learning_rate: {6}, epoch: {7}, " \
                                                 "negative_sampling_size: {8}"
                                        progress.section(format.format(algorithm, rank, num_layers, train_batch_size,
                                                                       predict_batch_size, lamb, learning_rate, epoch,
                                                                       negative_sampling_size))

                                        progress.subsection("Initializing Negative Sampler")

                                        negative_sampler = Negative_Sampler(df_train[[user_col,
                                                                                      item_col,
                                                                                      keyphrase_vector_col]],
                                                                            user_col,
                                                                            item_col,
                                                                            rating_col,
                                                                            keyphrase_vector_col,
                                                                            num_items=num_items,
                                                                            batch_size=train_batch_size,
                                                                            num_keyphrases=len(keyphrase_names),
                                                                            negative_sampling_size=negative_sampling_size)

                                        model = params['models'][algorithm](num_users=num_users,
                                                                            num_items=num_items,
                                                                            text_dim=len(keyphrase_names),
                                                                            embed_dim=rank,
                                                                            num_layers=num_layers,
                                                                            negative_sampler=negative_sampler,
                                                                            lamb=lamb,
                                                                            learning_rate=learning_rate)

                                        progress.subsection("Training")

                                        model.train_model(df_train,
                                                          user_col,
                                                          item_col,
                                                          rating_col,
                                                          epoch=epoch)

                                        progress.subsection("Prediction")

                                        df_valid_explanation = predict_explanation(model,
                                                                                   df_valid,
                                                                                   user_col,
                                                                                   item_col,
                                                                                   topk_keyphrase=params['topK'][-1])

                                        progress.subsection("Evaluation")

                                        explanation_result = evaluate_explanation(df_valid_explanation,
                                                                                  df_valid,
                                                                                  params['metric'],
                                                                                  params['topK'],
                                                                                  user_col,
                                                                                  item_col,
                                                                                  rating_col,
                                                                                  keyphrase_vector_col)

                                        result_dict = {'model': algorithm,
                                                       'rank': rank,
                                                       'num_layers': num_layers,
                                                       'train_batch_size': train_batch_size,
                                                       'predict_batch_size': predict_batch_size,
                                                       'lambda': lamb,
                                                       'learning_rate': learning_rate,
                                                       'epoch': epoch,
                                                       'negative_sampling_size': negative_sampling_size}

                                        for name in explanation_result.keys():
                                            result_dict[name] = [round(explanation_result[name][0], 4),
                                                                 round(explanation_result[name][1], 4)]

                                        df = df.append(result_dict, ignore_index=True)

                                        model.sess.close()
                                        tf.reset_default_graph()

                                        save_dataframe_csv(df, table_path, save_path)

