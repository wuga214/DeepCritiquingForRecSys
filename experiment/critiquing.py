from metrics.general_performance import evaluate, evaluate_explanation
from predicts.topk import elementwisepredictor, predict_explanation
from utils.modelnames import critiquing_models
from providers.sampler import Negative_Sampler
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml, find_best_hyperparameters
from utils.progress import WorkSplitter
from utils.reformat import to_sparse_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from critique.critique import critique_keyphrase
from metrics.critiquing_performance import falling_rank
from plots.rec_plots import show_critiquing


def critiquing(num_users, num_items, df_train, keyPhrase, params, num_critique, save_path, figure_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = find_best_hyperparameters(table_path + params['problem'], 'NDCG')

    dfs = []

    for index, row in df.iterrows():

        if row['model'] not in critiquing_models:
            continue

        algorithm = row['model']
        rank = row['rank']
        num_layers = row['num_layers']
        train_batch_size = row['train_batch_size']
        predict_batch_size = row['predict_batch_size']
        lam = row['lambda']
        learning_rate = row['learning_rate']
        epoch = 50
        negative_sampling_size = 1

        format = "model: {0}, rank: {1}, num_layers: {2}, train_batch_size: {3}, " \
                 "predict_batch_size: {4}, lambda: {5}, learning_rate: {6}, epoch: {7}, negative_sampling_size: {8}"
        progress.section(
            format.format(algorithm, rank, num_layers, train_batch_size, predict_batch_size, lam, learning_rate, epoch,
                          negative_sampling_size))

        progress.subsection("Initializing Negative Sampler")

        negative_sampler = Negative_Sampler(df_train[['UserIndex', 'ItemIndex', 'keyVector']],
                                            'UserIndex', 'ItemIndex', 'Binary', 'keyVector',
                                            num_items=num_items, batch_size=train_batch_size,
                                            num_keys=len(keyPhrase),
                                            negative_sampling_size=negative_sampling_size)

        model = critiquing_models[algorithm](num_users=num_users,
                                             num_items=num_items,
                                             text_dim=len(keyPhrase),
                                             embed_dim=rank,
                                             num_layers=num_layers,
                                             batch_size=train_batch_size,
                                             negative_sampler=negative_sampler,
                                             lamb=lam,
                                             learning_rate=learning_rate)

        model.train_model(df_train, epoch=epoch)

        affected_falling_rank_result = []
        inaffected_falling_rank_result = []

        for i in range(3):
            random_users = np.random.choice(num_users, num_critique)
            for user in tqdm(random_users):
                r_b, r_f, k = critique_keyphrase(model, user, num_items, topk_key=10, topk_item=500)
                affected_falling_rank_result.append(falling_rank(r_b.tolist(), r_f.tolist(), k))
                not_k = np.array(r_b.tolist()[:500])
                not_k = not_k[~np.in1d(not_k, k)]
                inaffected_falling_rank_result.append(falling_rank(r_b.tolist(), r_f.tolist(), not_k))

        df_affected = pd.DataFrame.from_dict({'Falling Rank': affected_falling_rank_result})
        df_affected['type'] = 'Affected'
        df_inaffected = pd.DataFrame.from_dict({'Falling Rank': inaffected_falling_rank_result})
        df_inaffected['type'] = 'Unaffected'
        df_box = pd.concat([df_inaffected, df_affected])

        df_box['model'] = algorithm

        dfs.append(df_box)

        model.sess.close()
        tf.reset_default_graph()

    df_output = pd.concat(dfs)
    save_dataframe_csv(df_output, table_path, name=save_path)

    show_critiquing(df_output, name=figure_path, x='model', y='Falling Rank', hue='type', save=True)