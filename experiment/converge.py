from metrics.general_performance import evaluate
from predicts.topk import elementwisepredictor
from utils.io import save_dataframe_csv
from utils.modelnames import models
from utils.progress import WorkSplitter
from utils.reformat import to_sparse_matrix

import json
import pandas as pd
import tensorflow as tf


def converge(df_data, df_train, df_test, keyPhrase, df, table_path, file_name, epochs=10, gpu_on=True):
    progress = WorkSplitter()

    valid_models = models.keys()

    results = pd.DataFrame(columns=['model', 'rank', 'lambda', 'epoch', 'optimizer'])

    for run in range(3):

        for idx, row in df.iterrows():
            row = row.to_dict()
            if row['model'] not in valid_models:
                continue

            progress.section(json.dumps(row))

            row['metric'] = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
            row['topK'] = [10]

            if 'optimizer' not in row.keys():
                row['optimizer'] = 'RMSProp'

            model = models[row['model']](num_users=df_data['UserIndex'].nunique(),
                                         num_items=df_data['ItemIndex'].nunique(),
                                         text_dim=len(keyPhrase),
                                         embed_dim=row['rank'],
                                         num_layers=row['num_layers'],
                                         batch_size=row['train_batch_size'],
                                         lamb=row['lambda'],
                                         learning_rate=row['learning_rate'])

            batches = model.get_batches(df_train, batch_size=row['train_batch_size'],
                                        user_col='UserIndex', item_col='ItemIndex',
                                        rating_col='Binary', key_col='keyVector',
                                        num_keys=len(keyPhrase))

            epoch_batch = 5

            for i in range(epochs//epoch_batch):

                model.train_model(df_train, epoch=epoch_batch, batches=batches)

                prediction, explanation = elementwisepredictor(model, df_train,
                                                               'UserIndex', 'ItemIndex',
                                                               row['topK'][0],
                                                               batch_size=row['predict_batch_size'],
                                                               explain=True,
                                                               key_names=keyPhrase)

                R_test = to_sparse_matrix(df_test, df_data['UserIndex'].nunique(),
                                          df_data['ItemIndex'].nunique(),
                                          'UserIndex', 'ItemIndex', 'Binary')

                result = evaluate(prediction, R_test, row['metric'], row['topK'])

                # Note Finished yet
                result_dict = {'model': row['model'],
                               'rank': row['rank'],
                               'lambda': row['lambda'],
                               'optimizer': row['optimizer'],
                               'epoch': (i+1)*epoch_batch}

                for name in result.keys():
                    result_dict[name] = round(result[name][0], 4)
                results = results.append(result_dict, ignore_index=True)
                print("result is \n {}".format(results))

            model.sess.close()
            tf.reset_default_graph()

            save_dataframe_csv(results, table_path, file_name)

    return results

