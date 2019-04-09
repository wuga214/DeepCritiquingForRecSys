from metrics.general_performance import evaluate
from predicts.topk import elementwisepredictor
from providers.sampler import Negative_Sampler
from utils.io import save_dataframe_csv
from utils.modelnames import models
from utils.progress import WorkSplitter
from utils.reformat import to_sparse_matrix

import json
import pandas as pd
import tensorflow as tf


def converge(num_users, num_items, df_train, df_test, keyPhrase, df, table_path, file_name, epochs=10, gpu_on=True):
    progress = WorkSplitter()

    valid_models = models.keys()

    results = pd.DataFrame(columns=['model', 'rank', 'num_layers', 'train_batch_size', 'predict_batch_size',
                                    'lambda', 'topK', 'learning_rate', 'epoch', 'negative_sampling_size', 'optimizer'])

    for run in range(3):

        for idx, row in df.iterrows():
            row = row.to_dict()
            if row['model'] not in valid_models:
                continue

            progress.section(json.dumps(row))

            row['metric'] = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
            row['topK'] = [10]

            if 'optimizer' not in row.keys():
                row['optimizer'] = 'Adam'

            negative_sampler = Negative_Sampler(df_train[['UserIndex', 'ItemIndex', 'keyVector']],
                                                'UserIndex', 'ItemIndex', 'Binary', 'keyVector',
                                                num_items=num_items, batch_size=row['train_batch_size'],
                                                num_keys=len(keyPhrase),
                                                negative_sampling_size=row['negative_sampling_size'])

            model = models[row['model']](num_users=num_users,
                                         num_items=num_items,
                                         text_dim=len(keyPhrase),
                                         embed_dim=row['rank'],
                                         num_layers=row['num_layers'],
                                         batch_size=row['train_batch_size'],
                                         negative_sampler=negative_sampler,
                                         lamb=row['lambda'],
                                         learning_rate=row['learning_rate'])

            batches = negative_sampler.get_batches()

            epoch_batch = 5

            for i in range(epochs//epoch_batch):

                model.train_model(df_train, epoch=epoch_batch, batches=batches)

                prediction, explanation = elementwisepredictor(model, df_train,
                                                               'UserIndex', 'ItemIndex',
                                                               row['topK'][0],
                                                               batch_size=row['predict_batch_size'],
                                                               explain=True,
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
                               'topK': row['topK'][0],
                               'learning_rate': row['learning_rate'],
                               'epoch': (i+1)*epoch_batch,
                               'negative_sampling_size': row['negative_sampling_size'],
                               'optimizer': row['optimizer']}

                for name in result.keys():
                    result_dict[name] = round(result[name][0], 4)
                results = results.append(result_dict, ignore_index=True)
                print("result is \n {}".format(results))

            model.sess.close()
            tf.reset_default_graph()

            save_dataframe_csv(results, table_path, file_name)

    return results

