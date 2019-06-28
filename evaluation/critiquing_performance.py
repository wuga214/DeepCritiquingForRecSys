from evaluation.general_performance import average_precisionk
from tqdm import tqdm
from utils.critique import critique_keyphrase, latent_density

import numpy as np
import pandas as pd


def critiquing_evaluation(model, model_name, num_users, num_items, num_users_sampled, topk):
    fmap_results = [[] for _ in topk]
    for iteration in range(5):
        sampled_users = np.random.choice(num_users, num_users_sampled)
        for user in tqdm(sampled_users):
            top_items_before_critique, top_items_after_critique, affected_items = critique_keyphrase(model, user,
                                                                                                     num_items,
                                                                                                     topk_keyphrase=10)

            all_items = np.array(range(num_items))
            unaffected_items = all_items[~np.in1d(all_items, affected_items)]

            for i, k in enumerate(topk):
                fmap_results[i].append(average_precisionk(top_items_before_critique[:k],
                                                          np.isin(top_items_before_critique[:k], affected_items))
                                       - average_precisionk(top_items_after_critique[:k],
                                                            np.isin(top_items_after_critique[:k], affected_items)))

    fmap_results_dict = dict()
    fmap_results_dict['model'] = model_name
    for i, k in enumerate(topk):
        fmap_results_dict['F-MAP@{0}'.format(k)] = fmap_results[i]
    df_fmap = pd.DataFrame(fmap_results_dict)

    return df_fmap


def latent_density_evaluation(model, model_name, num_users, num_items, num_users_sampled):
    df_list = []

    for iteration in range(1):
        sampled_users = np.random.choice(num_users, num_users_sampled)
        for user in tqdm(sampled_users):
            mean, modified_mean, init_mag, critiqued_mag = latent_density(model, user, num_items, topk_keyphrase=10)
            initial_dict = dict()
            initial_dict['model'] = model_name
            initial_dict['stage'] = "Initial"
            modified_dict = dict()
            modified_dict['model'] = model_name
            modified_dict['stage'] = "Modified"

            initial_dict['UserIndex'] = user
            initial_dict["X"] = mean[:, 0].tolist()
            initial_dict["Y"] = mean[:, 1].tolist()
            initial_dict['Magnitude'] = init_mag.tolist()
            modified_dict['UserIndex'] = user
            modified_dict["X"] = modified_mean[:, 0].tolist()
            modified_dict["Y"] = modified_mean[:, 1].tolist()
            modified_dict["" \
                          ""] = critiqued_mag.tolist()

            df_list.append(pd.DataFrame(initial_dict))
            df_list.append(pd.DataFrame(modified_dict))

    df = pd.concat(df_list)

    return df
