from metrics.general_performance import evaluate
from predicts.topk import elementwisepredictor
from tqdm import tqdm
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
from utils.progress import WorkSplitter
from utils.reformat import to_sparse_matrix

import inspect
import numpy as np
import pandas as pd
import tensorflow as tf


def hyper_parameter_tuning(df_data, df_train, df_valid, keyPhrase, params, save_path, gpu_on=True):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']

    try:
        df = load_dataframe_csv(table_path, save_path)
    except:
        df = pd.DataFrame(columns=['model', 'rank', 'num_layers', 'train_batch_size', 'predict_batch_size', 'lambda', 'topK', 'learning_rate', 'epoch'])

    for algorithm in params['models']:

        for rank in params['rank']:

            for num_layers in params['num_layers']:

                for train_batch_size in params['train_batch_size']:

                    for predict_batch_size in params['predict_batch_size']:

                        for lam in params['lambda']:

                            for learning_rate in params['learning_rate']:

                                for epoch in params['epoch']:

                                    if ((df['model'] == algorithm) &
                                        (df['rank'] == rank) &
                                        (df['num_layers'] == num_layers) &
                                        (df['train_batch_size'] == train_batch_size) &
                                        (df['predict_batch_size'] == predict_batch_size) &
                                        (df['lambda'] == lam) &
                                        (df['learning_rate'] == learning_rate) &
                                        (df['epoch'] == epoch)).any():
                                        continue

                                    format = "model: {0}, rank: {1}, num_layers: {2}, train_batch_size: {3}, predict_batch_size: {4}, lambda: {5}, learning_rate: {6}, epoch: {7}"
                                    progress.section(format.format(algorithm, rank, num_layers, train_batch_size, predict_batch_size, lam, learning_rate, epoch))

                                    model = params['models'][algorithm](num_users=df_data['UserIndex'].nunique(),
                                                                        num_items=df_data['ItemIndex'].nunique(),
                                                                        text_dim=len(keyPhrase),
                                                                        embed_dim=rank,
                                                                        num_layers=num_layers,
                                                                        batch_size=train_batch_size,
                                                                        lamb=lam,
                                                                        learning_rate=learning_rate)

                                    model.train_model(df_train, epoch=epoch)

                                    progress.subsection("Prediction")

                                    prediction, explanation = elementwisepredictor(model, df_train, 'UserIndex', 'ItemIndex',
                                                                                params['topK'][-1], batch_size=predict_batch_size, explain=True, key_names=keyPhrase)

                                    progress.subsection("Evaluation")

                                    R_valid = to_sparse_matrix(df_valid, df_data['UserIndex'].nunique(), df_data['ItemIndex'].nunique(), 'UserIndex', 'ItemIndex', 'Binary')

                                    result = evaluate(prediction, R_valid, params['metric'], params['topK'])

                                    result_dict = {'model': algorithm, 'rank': rank, 'num_layers': num_layers,
                                                   'train_batch_size': train_batch_size, 'predict_batch_size': predict_batch_size,
                                                   'lambda': lam, 'learning_rate': learning_rate, 'epoch': epoch}

                                    for name in result.keys():
                                        result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]

                                    df = df.append(result_dict, ignore_index=True)

                                    model.sess.close()
                                    tf.reset_default_graph()

                                    save_dataframe_csv(df, table_path, save_path)

