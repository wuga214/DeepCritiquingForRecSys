import numpy as np
from sklearn.model_selection import train_test_split


def random_split(df, ratio=0.2):
    train_and_valid, test = train_test_split(df, test_size=ratio)
    train, valid = train_test_split(train_and_valid, test_size=ratio)
    return train, valid, test


def leave_one_out_split(df, user_col, ratio):
    grouped = df.groupby(user_col, as_index=False)
    valid = grouped.apply(lambda x: x.sample(frac=ratio, random_state=8292))
    train = df.loc[~df.index.isin([x[1] for x in valid.index])]
    return train, valid

