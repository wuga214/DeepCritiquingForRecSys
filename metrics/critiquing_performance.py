

def falling_rank(rank_before, rank_after, item_affected):
    sum_rank_before = sum([rank_before.index(x) for x in item_affected])
    sum_rank_after = sum([rank_after.index(x) for x in item_affected])
    return sum_rank_after - sum_rank_before

