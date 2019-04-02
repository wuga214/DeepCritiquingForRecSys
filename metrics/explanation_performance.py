from tqdm import tqdm

import numpy as np
import pandas as pd


def predict_explanation(model, valid, user_col, item_col, topk_key=10):

    explanation = []

    _, explanation_scores = model.predict(valid[[user_col, item_col]].values)

    for explanation_score in tqdm(explanation_scores):
        explanation.append(np.argsort(explanation_score)[::-1][:topk_key])

    return pd.DataFrame.from_dict({user_col: valid[user_col].values,
                                   item_col: valid[item_col].values,
                                   'ExplanIndex': explanation})

