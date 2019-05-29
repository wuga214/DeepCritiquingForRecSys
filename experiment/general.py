from evaluation.general_performance import evaluate
from prediction.predictor import predict_elementwise
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml, find_best_hyperparameters
from utils.modelnames import models
from utils.progress import WorkSplitter
from utils.reformat import to_sparse_matrix
from utils.sampler import Negative_Sampler

import pandas as pd
import tensorflow as tf


def general(num_users, num_items, user_col, item_col, rating_col, keyphrase_vector_col, df_train, df_test, keyphrase_names, params, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = find_best_hyperparameters(table_path + params['tuning_result_path'], 'NDCG')

    try:
        output_df = load_dataframe_csv(table_path, save_path)
    except:
        output_df = pd.DataFrame(columns=['model', 'rank', 'num_layers', 'train_batch_size', 'predict_batch_size',
                                          'lambda', 'topK', 'learning_rate', 'epoch', 'negative_sampling_size'])

    for index, row in df.iterrows():

        algorithm = row['model']
        rank = row['rank']
        num_layers = row['num_layers']
        train_batch_size = row['train_batch_size']
        predict_batch_size = row['predict_batch_size']
        lamb = row['lambda']
        learning_rate = row['learning_rate']
        epoch = 300
        negative_sampling_size = row['negative_sampling_size']

        row['topK'] = [5, 10, 15, 20, 50]
        row['metric'] = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        format = "model: {0}, rank: {1}, num_layers: {2}, train_batch_size: {3}, " \
                 "predict_batch_size: {4}, lambda: {5}, learning_rate: {6}, epoch: {7}, negative_sampling_size: {8}"
        progress.section(
            format.format(algorithm, rank, num_layers, train_batch_size, predict_batch_size, lamb, learning_rate, epoch,
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

        model = models[algorithm](num_users=num_users,
                                  num_items=num_items,
                                  text_dim=len(keyphrase_names),
                                  embed_dim=rank,
                                  num_layers=num_layers,
                                  negative_sampler=negative_sampler,
                                  lamb=lamb,
                                  learning_rate=learning_rate)

        progress.subsection("Training")

        pretrained_path = load_yaml('config/global.yml', key='path')['pretrained']
        # try:
        #     model.load_model(pretrained_path+params['tuning_result_path'], row['model'])
        # except:
        model.train_model(df_train,
                          user_col,
                          item_col,
                          rating_col,
                          epoch=epoch)
        # model.save_model(pretrained_path+params['tuning_result_path'], row['model'])

        progress.subsection("Prediction")

        prediction, explanation = predict_elementwise(model,
                                                      df_train,
                                                      user_col,
                                                      item_col,
                                                      row['topK'][-1],
                                                      batch_size=row['predict_batch_size'],
                                                      enable_explanation=False,
                                                      keyphrase_names=keyphrase_names)

        R_test = to_sparse_matrix(df_test,
                                  num_users,
                                  num_items,
                                  user_col,
                                  item_col,
                                  rating_col)

        result = evaluate(prediction, R_test, row['metric'], row['topK'])

        # Note Finished yet
        result_dict = {'model': row['model'],
                       'rank': row['rank'],
                       'num_layers': row['num_layers'],
                       'train_batch_size': row['train_batch_size'],
                       'predict_batch_size': row['predict_batch_size'],
                       'lambda': row['lambda'],
                       'topK': row['topK'][-1],
                       'learning_rate': row['learning_rate'],
                       'epoch': epoch,
                       'negative_sampling_size': row['negative_sampling_size'],
                       }

        for name in result.keys():
            result_dict[name] = round(result[name][0], 4)
        output_df = output_df.append(result_dict, ignore_index=True)

        model.sess.close()
        tf.reset_default_graph()

        save_dataframe_csv(output_df, table_path, save_path)

    return output_df
