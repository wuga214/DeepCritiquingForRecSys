import numpy as np
from sklearn.model_selection import train_test_split


def random_split(df, ratio=0.2):
    train_and_valid, test = train_test_split(df, test_size=ratio)
    train, valid = train_test_split(train_and_valid, test_size=ratio)
    return train, valid, test
