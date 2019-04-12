import numpy as np


def falling_rank(rank_before, rank_after, item_affected):
    rank_before = np.array([rank_before.index(x) for x in item_affected])
    rank_after = np.array([rank_after.index(x) for x in item_affected])
    return np.sum((rank_after-rank_before) > 0)/float(len(rank_before)) - np.sum((rank_before-rank_after) > 0)/float(len(rank_before))

