from tqdm import tqdm

import numpy as np
import pandas as pd


def predict_elementwise(model, df_train, user_col, item_col, topk,
                        batch_size=1024, enable_explanation=False,
                        keyphrase_names=None, topk_keyphrase=10):

    predictions = []
    explanation = []

    num_users = model.num_users
    num_items = model.num_items

    for i in tqdm(range(num_users)):
        input_batch = []
        output_batch = []
        rated_items = df_train[df_train[user_col] == i][item_col].values
        for j in range(num_items):
            if j in rated_items:
                continue
            input_batch.append([i, j])
            if (j + 1) % batch_size == 0 or (j + 1) == num_items:
                inputs = np.array(input_batch)
                output_batch.append(np.concatenate([inputs] + model.predict(inputs), axis=1))
                input_batch = []

        prediction = np.concatenate(output_batch, axis=0)
        prediction = prediction[prediction[:, 2].argsort()[::-1][:topk]]
        candidates = prediction[:, 1].astype(int)
        predictions.append(candidates)

        if enable_explanation:
            for j in range(topk):
                candidate_keyphrase_indicies = np.argsort(prediction[j, 3:])[::-1][:topk_keyphrase]
                candidate_keyphrases = keyphrase_names[candidate_keyphrase_indicies]
                explanation.append({user_col: i, item_col: candidates[j],
                                    'ExplanIndex': candidate_keyphrase_indicies,
                                    'Explanation': candidate_keyphrases})

    return np.array(predictions), pd.DataFrame(explanation)


def predict_explanation(model, df_valid, user_col, item_col, topk_keyphrase=10):

    explanation = []

    _, explanation_scores = model.predict(df_valid[[user_col, item_col]].values)

    for explanation_score in tqdm(explanation_scores):
        explanation.append(np.argsort(explanation_score)[::-1][:topk_keyphrase])

    return pd.DataFrame.from_dict({user_col: df_valid[user_col].values,
                                   item_col: df_valid[item_col].values,
                                   'ExplanIndex': explanation})
