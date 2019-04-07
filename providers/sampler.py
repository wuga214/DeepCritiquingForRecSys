import numpy as np
import pandas as pd
import random
import scipy.sparse as sparse


class Negative_Sampler(object):
    def __init__(self, df, user_col, item_col, rating_col, key_col, num_items, batch_size, num_keys, negative_sampling_size=10):
        self.df = df
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.key_col = key_col
        self.num_items = num_items
        self.batch_size = batch_size
        self.num_keys = num_keys
        self.negative_sampling_size = negative_sampling_size
        self.prepare_positive_sampling()
        self.prepare_negative_sampling()

    def concate_data(self, permutation=True):
        self.users = np.concatenate([self.pos_users, self.neg_users])
        self.items = np.concatenate([self.pos_items, self.neg_items])
        self.ratings = np.concatenate([self.pos_ratings, self.neg_ratings])
        self.keys = sparse.vstack([self.pos_keys, self.neg_keys])
        if permutation:
            index = np.random.permutation(len(self.users))
            self.users = self.users[index]
            self.items = self.items[index]
            self.ratings = self.ratings[index]
            self.keys = self.keys[index]

    def sparsify_keys(self):
        df_keys = self.df[[self.key_col]].assign(row_index=np.arange(len(self.df)))
        series_key = df_keys.set_index(['row_index'])[self.key_col].apply(pd.Series).stack().reset_index(level=1, drop=True)

        row = series_key.index.values
        col = series_key.values.astype(int)

        return sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(self.df), self.num_keys))

    def prepare_positive_sampling(self):
        self.pos_users = self.df[self.user_col].values
        self.pos_items = self.df[self.item_col].values
        self.pos_ratings = np.ones(len(self.pos_users))
        self.pos_keys = self.sparsify_keys()

    def prepare_negative_sampling(self):
        self.df_user = self.df[[self.user_col, self.item_col]].groupby(self.user_col)[self.item_col].apply(list).to_frame().reset_index()
        self.df_user['Num_Pos'] = self.df_user[self.item_col].str.len()
        self.df_user['Unobserved'] = self.df_user[self.item_col].apply(lambda observed_items: np.setdiff1d(np.arange(self.num_items), observed_items))
        self.df_user['Num_Unobserved'] = self.df_user['Unobserved'].str.len()
        self.df_user['Num_Neg'] = self.df_user['Num_Pos'] * self.negative_sampling_size
        self.df_user.loc[self.df_user.Num_Neg > self.df_user.Num_Unobserved, 'Num_Neg'] = self.df_user.Num_Unobserved

    def sample_negative(self):
        self.df_user['Sampled_Items'] = self.df_user.apply(lambda row: np.random.choice(row['Unobserved'], row['Num_Neg'], replace=False), axis=1)
        series_user = self.df_user.set_index([self.user_col])['Sampled_Items'].apply(pd.Series).stack().reset_index(level=1, drop=True)

        self.neg_users = series_user.index.values
        self.neg_items = series_user.values.astype(int)
        self.neg_ratings = np.zeros(len(self.neg_users))
        self.neg_keys = sparse.csr_matrix((len(self.neg_users), self.num_keys))

    def get_batches(self):
        self.sample_negative()
        self.concate_data()

        remaining_size = len(self.users)

        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < self.batch_size:
                batches.append([self.users[batch_index*self.batch_size:],
                                self.items[batch_index*self.batch_size:],
                                self.ratings[batch_index*self.batch_size:],
                                self.keys[batch_index*self.batch_size:]])
            else:
                batches.append([self.users[batch_index*self.batch_size:(batch_index+1)*self.batch_size],
                                self.items[batch_index*self.batch_size:(batch_index+1)*self.batch_size],
                                self.ratings[batch_index*self.batch_size:(batch_index+1)*self.batch_size],
                                self.keys[batch_index*self.batch_size:(batch_index+1)*self.batch_size]])
            batch_index += 1
            remaining_size -= self.batch_size
        random.shuffle(batches)
        return batches

