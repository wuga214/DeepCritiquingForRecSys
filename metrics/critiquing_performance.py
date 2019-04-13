import numpy as np

def falling_rank(rank_before, rank_after, item_affected):
    sum_rank_before = sum([rank_before.index(x) for x in item_affected])
    sum_rank_after = sum([rank_after.index(x) for x in item_affected])
    return sum_rank_after - sum_rank_before

# def falling_rank(rank_before, rank_after, item_affected):
#     rank_before = np.array([rank_before.index(x) for x in item_affected])
#     rank_after = np.array([rank_after.index(x) for x in item_affected])
#     return np.sum((rank_after-rank_before) > 0)/float(len(rank_before)) - np.sum((rank_before-rank_after) > 0)/float(len(rank_before))