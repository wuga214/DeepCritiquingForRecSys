import scipy.sparse as sparse
import numpy as np
from sklearn.utils.extmath import randomized_svd


def to_sparse_matrix(df, num_user, num_item, user_col, item_col, rating_col):

    dok = df[[user_col, item_col, rating_col]].copy()
    dok = dok.as_matrix()
    dok = dok[dok[:, 2] > 0]
    shape = [num_user, num_item]

    return sparse.csr_matrix((dok[:, 2].astype(np.float32), (dok[:, 0], dok[:, 1])), shape=shape)


def to_laplacian(R, rank):
    W = R.dot(R.T)
    D = np.squeeze(np.asarray(W.sum(axis=1)))
    sqrtD = 1./np.sqrt(D) # inverse expression
    sqrtD= sparse.spdiags(sqrtD, 0, len(sqrtD), len(sqrtD))
    normL = sparse.identity(len(D)) - (sqrtD.dot(W)).dot(sqrtD)

    P, sigma, _ = randomized_svd(normL,
                                 n_components=rank,
                                 n_iter=4,
                                 random_state=1)

    return P*sigma


def to_svd(R, rank):

    P, sigma, QT = randomized_svd(R,
                                  n_components=rank,
                                  n_iter=4,
                                  random_state=1)

    return P*np.sqrt(sigma), QT.T*np.sqrt(sigma)


def get_phrase_set(df, user_col, explanation_col, set_col):
    grouped = df.groupby(user_col).agg(list).reset_index()
    grouped[set_col] = grouped[explanation_col].apply(lambda x:
                                                      set([item for sublist in x for item in sublist]))
    return grouped