from sklearn.utils.extmath import randomized_svd

import numpy as np
import scipy.sparse as sparse


def to_sparse_matrix(df, num_user, num_item, user_col, item_col, rating_col):

    dok = df[[user_col, item_col, rating_col]].copy()
    dok = dok.values
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


def to_svd(R, rank, standard=True):

    P, sigma, QT = randomized_svd(R,
                                  n_components=rank,
                                  n_iter=7,
                                  random_state=1)

    if standard:
        return standarize(P*np.sqrt(sigma)), standarize(QT.T*np.sqrt(sigma))
    else:
        return P*np.sqrt(sigma), QT.T*np.sqrt(sigma)


def standarize(array):
    return (array - np.mean(array, axis=0))/np.std(array, axis=0)
