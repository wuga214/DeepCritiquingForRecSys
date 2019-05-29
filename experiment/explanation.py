from evaluation.general_performance import evaluate_explanation
from prediction.predictor import predict_explanation
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
from utils.modelnames import explanable_models
from utils.progress import WorkSplitter
from utils.sampler import Negative_Sampler

import pandas as pd
import tensorflow as tf


def explain(num_users, num_items, user_col, item_col, rating_col, keyphrase_vector_col, df_train, df_valid, keyphrase_names, params, load_path, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = load_dataframe_csv(table_path, load_path)

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
        epoch = row['epoch']
        negative_sampling_size = row['negative_sampling_size']

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

        model = explanable_models[algorithm](num_users=num_users,
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
            result_dict[name] = [round(explanation_result[name][0], 4), round(explanation_result[name][1], 4)]

        output_df = output_df.append(result_dict, ignore_index=True)

        try:
            model.sess.close()
            tf.reset_default_graph()
        except:
            pass

        save_dataframe_csv(output_df, table_path, save_path)
