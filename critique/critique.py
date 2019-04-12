import numpy as np


def demo_critique_keyphrase(model, user_index, num_items, keyphrase_index, topk_key=10):
    inputs = np.array([[user_index, item_index] for item_index in range(num_items)])
    rating, explanation = model.predict(inputs)

    explanation_rank_list = np.empty((0, topk_key))
    for explanation_score in explanation:
        explanation_rank = np.argsort(explanation_score)[::-1][:topk_key]
        explanation_rank_list = np.append(explanation_rank_list, [explanation_rank], axis=0)

    affected_items = np.where(explanation_rank_list == keyphrase_index)[0]

    explanation[:, keyphrase_index] = 0

    modified_rating, modified_explanation = model.refine_predict(inputs, explanation)

    return np.argsort(rating.flatten())[::-1], np.argsort(modified_rating.flatten())[::-1], affected_items


def critique_keyphrase(model, user_index, num_items, topk_key=10, topk_item=500):
    inputs = np.array([[user_index, item_index] for item_index in range(num_items)])
    rating, explanation = model.predict(inputs)

    rating_order = np.argsort(rating.flatten())[::-1]
    top_items = rating_order[:topk_item]

    explanation_rank_list = np.argsort(-explanation[top_items], axis=1)[:, :topk_key]

    unique_keyphrase = np.unique(explanation_rank_list)
    keyphrase_index = int(np.random.choice(unique_keyphrase, 1)[0])

    affected_items = np.where(explanation_rank_list == keyphrase_index)[0]

    explanation[:, keyphrase_index] = 0

    modified_rating, modified_explanation = model.refine_predict(inputs, explanation)

    return rating_order, np.argsort(modified_rating.flatten())[::-1], affected_items

