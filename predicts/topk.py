import numpy as np
from tqdm import tqdm


def elementwisepredictor(model, train, user_col, item_col, topk, batch_size=1000):

    prediction = []

    num_user = model.num_users
    num_item = model.num_items

    for i in tqdm(range(num_user)):

        input_batch = []
        output_batch = []
        rated_item = train[train[user_col] == i][item_col].as_matrix()
        for j in range(num_item):
            if j in rated_item:
                continue
            input_batch.append([i, j])
            if (j + 1) % 1000 == 0 or (j + 1) == num_item:
                inputs = np.array(input_batch)
                output_batch.append(np.concatenate([inputs] + model.predict(inputs), axis=1))
                input_batch = []

        user_output = np.concatenate(output_batch, axis=0)

        prediction.append(user_output[user_output[:, 2].argsort()[::-1][:topk]])

    return np.array(prediction)




