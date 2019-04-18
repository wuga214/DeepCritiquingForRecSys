import numpy as np
from sklearn.decomposition import PCA


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


def critique_keyphrase(model, user_index, num_items, topk_key=10):
    inputs = np.array([[user_index, item_index] for item_index in range(num_items)])
    rating, explanation = model.predict(inputs)

    explanation_rank_list = np.argsort(-explanation, axis=1)[:, :topk_key]

    unique_keyphrase = np.unique(explanation_rank_list)
    keyphrase_index = int(np.random.choice(unique_keyphrase, 1)[0])

    affected_items = np.where(explanation_rank_list == keyphrase_index)[0]

    explanation[:, keyphrase_index] = np.min(explanation, axis=1)

    modified_rating, modified_explanation = model.refine_predict(inputs, explanation)

    return np.argsort(rating.flatten())[::-1], np.argsort(modified_rating.flatten())[::-1], affected_items


def latent_density(model, user_index, num_items, topk_key=10):
    inputs = np.array([[user_index, item_index] for item_index in range(num_items)])
    rating, explanation = model.predict(inputs)

    explanation_rank_list = np.argsort(-explanation, axis=1)[:, :topk_key]

    unique_keyphrase = np.unique(explanation_rank_list)
    keyphrase_index = int(np.random.choice(unique_keyphrase, 1)[0])

    explanation[:, keyphrase_index] = np.min(explanation, axis=1)

    mean, modified_mean = model.density_shifting_estimate(inputs, explanation)

    init_magnitude = np.mean(np.abs(mean), axis=1)
    critiqued_magnitude = np.mean(np.abs(modified_mean), axis=1)

    pca = PCA(n_components=2)
    mean_2d = pca.fit_transform(mean)
    modified_mean = pca.transform(modified_mean)

    return mean_2d, modified_mean, init_magnitude, critiqued_magnitude

