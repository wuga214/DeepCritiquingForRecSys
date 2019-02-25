import sparse
import numpy as np
from tqdm import tqdm


def split_seed_randomly(rating_tensor, ratio=[0.8, 0.1, 0.1],
                        remove_empty=True, split_seed=8292, sampling=False, percentage=0.1):
    '''
    Split based on a deterministic seed randomly
    '''

    if sampling:
        m, n, k = rating_tensor.shape
        index = np.random.choice(m, int(m * percentage))
        rating_tensor = rating_tensor[index]

    if remove_empty:
        # Remove empty columns. record original item index
        nonzero_index = np.unique(rating_tensor.nonzero()[1])
        rating_tensor = rating_tensor[:, nonzero_index,:]

        # Remove empty rows. record original user index
        nonzero_rows = np.unique(rating_tensor.nonzero()[0])
        rating_tensor = rating_tensor[nonzero_rows]

    # Note: This just gets the highest userId and doesn't account for non-contiguous users.
    user_num, item_num, k = rating_tensor.shape

    # work until here!!!

    rtrain = sparse.DOK((user_num, item_num, k))
    rvalid = sparse.DOK((user_num, item_num, k))
    rtest = sparse.DOK((user_num, item_num, k))

    # Set the random seed for splitting
    np.random.seed(split_seed)
    permuteIndices = np.random.permutation(rating_tensor.nnz)

    num_test = int(rating_tensor.nnz * ratio[2])
    num_valid = int(rating_tensor.nnz * (ratio[1] + ratio[2]))

    valid_offset = rating_tensor.nnz - num_valid

    test_offset = rating_tensor.nnz - num_test

    train_index = permuteIndices[:valid_offset]
    valid_index = permuteIndices[valid_offset:test_offset]
    test_index = permuteIndices[test_offset:]


    nonzeros = rating_tensor.nonzero()

    for i in tqdm(train_index):
        rtrain[nonzeros[0][i], nonzeros[1][i], nonzeros[2][i]] = rating_tensor[nonzeros[0][i],
                                                                               nonzeros[1][i],
                                                                               nonzeros[2][i]]
    rtrain = sparse.COO(rtrain)

    for i in tqdm(valid_index):
        rvalid[nonzeros[0][i], nonzeros[1][i], nonzeros[2][i]] = rating_tensor[nonzeros[0][i],
                                                                               nonzeros[1][i],
                                                                               nonzeros[2][i]]
    rvalid = sparse.COO(rvalid)

    for i in tqdm(test_index):
        rtest[nonzeros[0][i], nonzeros[1][i], nonzeros[2][i]] = rating_tensor[nonzeros[0][i],
                                                                              nonzeros[1][i],
                                                                              nonzeros[2][i]]
    rtest = sparse.COO(rtest)

    return rtrain, rvalid, rtest