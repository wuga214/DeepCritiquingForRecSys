import numpy as np
import pandas as pd


class UserPop(object):
    def __init__(self,
                 num_users,
                 num_items,
                 text_dim,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.text_dim = text_dim

    def train_model(self, df, user_col, item_col, rating_col, epoch, keyphrase_vector_col='keyVector', **unused):

        df_user_pop = df[[user_col, keyphrase_vector_col]].groupby(user_col)[keyphrase_vector_col].apply(sum).to_frame().reset_index()

        df_user = pd.DataFrame({user_col: range(self.num_users)})
        df_user_pop = pd.merge(df_user, df_user_pop, how='left', on=user_col)

        df_user_pop[keyphrase_vector_col] = df_user_pop[keyphrase_vector_col].apply(lambda x: self.udf(x))
        df_user_pop['Unobserved'] = (df_user_pop[keyphrase_vector_col]
                                     .apply(lambda observed_items: np.setdiff1d(np.arange(self.text_dim),
                                                                                observed_items)))
        df_user_pop['predict'] = df_user_pop.apply(lambda x: np.concatenate((x[keyphrase_vector_col],x['Unobserved'])), axis=1)
        self.user_pop = df_user_pop.drop(columns=[keyphrase_vector_col, 'Unobserved']).values

    def predict(self, inputs):
        user_index = inputs[:, 0]
        phrase_prediction = np.array(self.user_pop[user_index][:, 1].tolist())
        return None, phrase_prediction

    @staticmethod
    def udf(keyVector):
        u, count = np.unique(keyVector, return_counts=True)
        count_sort_ind = np.argsort(-count)
        return u[count_sort_ind].astype(int)
