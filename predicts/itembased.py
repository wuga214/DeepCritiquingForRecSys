import numpy as np
from tqdm import tqdm


def topk_predict(model, Rvalid, topk, n, m, k):

    output_tensor = -np.ones((m, topk, k+1))

    # Rtrain_index = []
    # for i in range(m):
    #     Rtrain_index.append(Rtrain[i].nonzero())

    for i in tqdm(range(n)):
        results = model.predict(Rvalid[i:i + 1])
        minimum_index = np.argmin(output_tensor[:, :, 1], axis=1)
        for j in range(m):
            if output_tensor[j, minimum_index[j], 1] < results[0, j, 0]:
                output_tensor[j, minimum_index[j], 1:] = results[0, j]
                output_tensor[j, minimum_index[j], 0] = i

    index = np.flip(np.argsort(output_tensor[:, :, 1], axis=1), axis=1)
    output_tensor = np.delete(output_tensor, 1, 2)

    for i in tqdm(range(m)):
        output_tensor[i] = output_tensor[i][index[i]]

    return output_tensor