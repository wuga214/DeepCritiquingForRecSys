from metrics.general_performance import evaluate, evaluate_explanation
from predicts.topk import elementwisepredictor, predict_explanation
from utils.modelnames import models
from providers.sampler import Negative_Sampler
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml, find_best_hyperparameters
from utils.progress import WorkSplitter
from utils.reformat import to_sparse_matrix
import pandas as pd
import tensorflow as tf


def excute(num_users, num_items, df_train, df_test, keyPhrase, params, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    df = find_best_hyperparameters(table_path + params['problem'], 'NDCG')

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
        lam = row['lambda']
        learning_rate = row['learning_rate']
        epoch = 300
        negative_sampling_size = row['negative_sampling_size']

        row['topK'] = [5, 10, 15, 20, 50]
        row['metric'] = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

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

        model = models[algorithm](num_users=num_users,
                                  num_items=num_items,
                                  text_dim=len(keyPhrase),
                                  embed_dim=rank,
                                  num_layers=num_layers,
                                  batch_size=train_batch_size,
                                  negative_sampler=negative_sampler,
                                  lamb=lam,
                                  learning_rate=learning_rate)

        pretrained_path = load_yaml('config/global.yml', key='path')['pretrained']
        # try:
        #     model.load_model(pretrained_path+params['problem'], row['model'])
        # except:
        model.train_model(df_train, epoch=epoch)
        # model.save_model(pretrained_path+params['problem'], row['model'])

        prediction, explanation = elementwisepredictor(model, df_train,
                                                       'UserIndex', 'ItemIndex',
                                                       row['topK'][-1],
                                                       batch_size=row['predict_batch_size'],
                                                       explain=False,
                                                       key_names=keyPhrase)

        R_test = to_sparse_matrix(df_test, num_users, num_items,
                                  'UserIndex', 'ItemIndex', 'Binary')

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
                       'negative_sampling_size': row['negative_sampling_size'],
                       }

        for name in result.keys():
            result_dict[name] = round(result[name][0], 4)
        output_df =output_df.append(result_dict, ignore_index=True)

        model.sess.close()
        tf.reset_default_graph()

        save_dataframe_csv(output_df, table_path, save_path)

    return output_df