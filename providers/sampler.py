import numpy as np
import scipy.sparse as sparse

def negative_data(df, size_per_user):

    m = df['UserID'].nunique()
    n = df['ItemID'].nunique()

    negative_data = []
    for i in range(m):
        items = np.random.choice(n, size_per_user, replace=False)
        observed_items = df[df['UserID'] == i]['ItemID'].as_matrix().flatten()
        items = items[np.invert(np.isin(items, observed_items))]
        negative_data.append(sparse.csr_matrix((np.full(len(items), i), (range(len(items)), np.zeros(len(items)))),
                                               shape=(len(items), len(df.columns)))
                             + sparse.csr_matrix((items, (range(len(items)), np.ones(len(items)))),
                                                 shape=(len(items), len(df.columns))))

    return sparse.vstack(negative_data)