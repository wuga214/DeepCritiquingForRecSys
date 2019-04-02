import numpy as np


def critique_keyphrase(model, user_index, num_items, keyphrase_index, topk_key=10):
    inputs = np.array([[user_index, item_index] for item_index in range(num_items)])
    rating, explanation = model.predict(inputs)

    affected_items = []
    for i, explanation_score in enumerate(explanation):
        explanation_rank = np.argsort(explanation_score)[::-1][:topk_key]
        if keyphrase_index in explanation_rank:
            affected_items.append(i)

    explanation[:, keyphrase_index] = 0

    modified_rating, modified_explanation = model.refine_predict(inputs, explanation)

    return np.argsort(rating.flatten())[::-1], np.argsort(modified_rating.flatten())[::-1], affected_items