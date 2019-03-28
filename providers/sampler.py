import ast
import numpy as np
import scipy.sparse as sparse


def get_negative_sample(df, num_item, user_col, item_col, size_per_user, num_keys):
    m = df[user_col].unique()

    users = []
    items = []

    for i in m:
        sampled_items = np.random.choice(num_item, size_per_user, replace=False)
        # observed_items = df[df[user_col] == i][item_col].tolist() #.as_matrix().flatten()
        # sampled_items = sampled_items[~np.isin(sampled_items, observed_items)]
        users += [i] * len(sampled_items)
        items += sampled_items.tolist()

    ratings = [0] * len(users)
    keys = sparse.csr_matrix((len(users), num_keys))

    return [np.array(users), np.array(items), np.array(ratings), keys]


def sparsify_keys(key_vector, num_keys):
    indecs = []
    for i in range(len(key_vector)):
        for j in key_vector[i]:
            indecs.append([i, j])

    indecs = np.array(indecs)
    return sparse.csr_matrix((np.ones(len(indecs)), (indecs[:, 0], indecs[:, 1])),
                             shape=(len(key_vector), num_keys))


def get_arrays(df, user_col, item_col, rating_col, key_col, num_keys):
    users = df[user_col].values
    items = df[item_col].values
    ratings = df[rating_col].values
    keys = sparsify_keys(df[key_col].apply(ast.literal_eval).values.tolist(), num_keys)
    return [users, items, ratings, keys]


def concate_data(positive, negative, permutation=True):
    users = np.concatenate([positive[0], negative[0]])
    items = np.concatenate([positive[1], negative[1]])
    ratings = np.concatenate([positive[2], negative[2]])
    keys = sparse.vstack([positive[3], negative[3]])
    if permutation:
        index = np.random.permutation(len(users))
        users = users[index]
        items = items[index]
        ratings = ratings[index]
        keys = keys[index]
    return [np.array(users), np.array(items), np.array(ratings), keys]

