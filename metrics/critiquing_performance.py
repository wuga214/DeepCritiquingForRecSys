import numpy as np
import pandas as pd
from tqdm import tqdm
from critique.critique import critique_keyphrase
from metrics.general_performance import ndcg, precisionk, recallk, average_precisionk


def falling_rank(rank_before, rank_after, item_affected):
    if len(item_affected) == 0:
        return 0.
    sum_rank_before = sum([rank_before.index(x) for x in item_affected])
    sum_rank_after = sum([rank_after.index(x) for x in item_affected])
    return (sum_rank_after - sum_rank_before)/float(len(item_affected))


def critiquing_evaluation(model, model_name, num_users, num_items, num_critique, topk):

    affected_falling_rank_result = []
    inaffected_falling_rank_result = []

    column_names = ['model'] + ["MAP@{0}".format(k) for k in topk]

    map_results = [[] for k in topk]

    for i in range(5):
        random_users = np.random.choice(num_users, num_critique)
        for user in tqdm(random_users):
            r_b, r_f, affected_items = critique_keyphrase(model, user, num_items, topk_key=10)
            if len(affected_items) != 0:
                affected_falling_rank_result.append(falling_rank(r_b.tolist(), r_f.tolist(), affected_items))
            not_affected_items = np.array(range(num_items))
            not_affected_items = not_affected_items[~np.in1d(not_affected_items, affected_items)]
            if len(not_affected_items) != 0:
                inaffected_falling_rank_result.append(falling_rank(r_b.tolist(), r_f.tolist(), not_affected_items))

            for i, k in enumerate(topk):
                map_results[i].append(average_precisionk(r_b[:k], np.isin(r_b[:k], affected_items))
                                      - average_precisionk(r_f[:k], np.isin(r_f[:k], affected_items)))


    map_results_dict = dict()
    map_results_dict['model'] = model_name
    for i, k in enumerate(topk):
        map_results_dict['MAP@{0}'.format(k)] = map_results[i]
    df_map = pd.DataFrame(map_results_dict)

    df_affected = pd.DataFrame.from_dict({'Falling Rank': affected_falling_rank_result})
    df_affected['type'] = 'Affected'
    df_inaffected = pd.DataFrame.from_dict({'Falling Rank': inaffected_falling_rank_result})
    df_inaffected['type'] = 'Unaffected'
    df_fr = pd.concat([df_inaffected, df_affected])

    df_fr['model'] = model_name

    return df_fr, df_map