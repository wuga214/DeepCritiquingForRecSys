import scipy.sparse as sparse
import numpy as np

def to_sparse_matrix(df, num_user, num_item, user_col, item_col, rating_col, threshold):

    dok = df[[user_col, item_col, rating_col]].copy()
    dok = dok.as_matrix()
    dok = dok[dok[:, 2] > 0]
    shape = [num_user, num_item]

    return sparse.csr_matrix((dok[:, 2].astype(np.float32), (dok[:, 0], dok[:, 1])), shape=shape)


