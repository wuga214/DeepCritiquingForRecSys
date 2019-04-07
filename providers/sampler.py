import ast
import numpy as np
import random
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


def get_batches(df, batch_size, user_col, item_col, rating_col, key_col, num_items, num_keys):

    remaining_size = len(df)

    if batch_size > 4096:
        df = df.iloc[np.random.permutation(len(df))]

    batch_index = 0
    batches = []
    while remaining_size > 0:
        if remaining_size < batch_size:
            df_batch = df[ batch_index *batch_size:]
            positive_data = get_arrays(df_batch, user_col, item_col, rating_col, key_col, num_keys)
            negative_data = get_negative_sample(df_batch, num_items, user_col, item_col, 10, num_keys)
            train_array = concate_data(positive_data, negative_data)
            batches.append(train_array)
        else:
            df_batch = df[ batch_index *batch_size:( batch_index +1 ) *batch_size]
            positive_data = get_arrays(df_batch, user_col, item_col, rating_col, key_col, num_keys)
            negative_data = get_negative_sample(df_batch, num_items, user_col, item_col, 10, num_keys)
            train_array = concate_data(positive_data, negative_data)
            batches.append(train_array)
        batch_index += 1
        remaining_size -= batch_size
    random.shuffle(batches)
    return batches
