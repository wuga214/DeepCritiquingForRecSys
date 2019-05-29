from evaluation.critiquing_performance import latent_density_evaluation
from utils.io import save_dataframe_csv, load_yaml
from utils.modelnames import critiquing_models
from utils.progress import WorkSplitter
from utils.sampler import Negative_Sampler

import pandas as pd
import tensorflow as tf


def latent_density_estimation(num_users, num_items, user_col, item_col, rating_col, keyphrase_vector_col, df_train, keyphrase_names, params, num_users_sampled, load_path, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = pd.read_csv(table_path + load_path)

    dfs = []

    for index, row in df.iterrows():

        if row['model'] not in critiquing_models:
            continue

        algorithm = row['model']
        rank = row['rank']
        num_layers = row['num_layers']
        train_batch_size = row['train_batch_size']
        predict_batch_size = row['predict_batch_size']
        lamb = row['lambda']
        learning_rate = row['learning_rate']
        epoch = 200
        negative_sampling_size = 1

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

        model = critiquing_models[algorithm](num_users=num_users,
                                             num_items=num_items,
                                             text_dim=len(keyphrase_names),
                                             embed_dim=rank,
                                             num_layers=num_layers,
                                             negative_sampler=negative_sampler,
                                             lamb=lamb,
                                             learning_rate=learning_rate)

        pretrained_path = load_yaml('config/global.yml', key='path')['pretrained']
        try:
            model.load_model(pretrained_path+params['model_saved_path'], row['model'])
        except:
            model.train_model(df_train,
                              user_col,
                              item_col,
                              rating_col,
                              epoch=epoch)
            model.save_model(pretrained_path+params['model_saved_path'], row['model'])

        df_result = latent_density_evaluation(model, algorithm, num_users, num_items, num_users_sampled)

        dfs.append(df_result)

        model.sess.close()
        tf.reset_default_graph()

    df_output = pd.concat(dfs)
    save_dataframe_csv(df_output, table_path, name=save_path+'_Latent.csv')

