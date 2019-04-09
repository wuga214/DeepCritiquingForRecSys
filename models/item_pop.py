import numpy as np
import pandas as pd

class ItemPop(object):
    def __init__(self,
                 num_users,
                 num_items,
                 text_dim,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.text_dim = text_dim

    def train_model(self, df, item_col='ItemIndex', key_col='keyVector', **unused):

        df_item_pop = df[[item_col, key_col]].groupby(item_col)[key_col].apply(sum).to_frame().reset_index()

        df_item = pd.DataFrame({item_col: range(self.num_items)})
        df_item_pop = pd.merge(df_item, df_item_pop, how='left', on=item_col)

        df_item_pop.loc[df_item_pop[key_col].isnull(), key_col] = df_item_pop.loc[df_item_pop[key_col].isnull(),
                                                                                  key_col].apply(lambda x: [])

        df_item_pop[key_col] = df_item_pop[key_col].apply(lambda x: self.udf(x))
        df_item_pop['Unobserved'] = (df_item_pop[key_col]
                                     .apply(lambda observed_items: np.setdiff1d(np.arange(self.text_dim),
                                                                                observed_items)))
        df_item_pop['predict'] = df_item_pop.apply(lambda x: np.concatenate((x[key_col],x['Unobserved'])), axis=1)
        self.item_pop = df_item_pop.drop(columns=[key_col, 'Unobserved']).values

    def predict(self, inputs):
        item_index = inputs[:, 1]
        phrase_prediction = np.array(self.item_pop[item_index][:, 1].tolist())
        return None, phrase_prediction

    @staticmethod
    def udf(keyVector):
        u, count = np.unique(keyVector, return_counts=True)
        count_sort_ind = np.argsort(-count)
        return u[count_sort_ind].astype(int)