import json
import pandas as pd
from tqdm import tqdm
from collections import Counter


def getsubset(path, user_col, item_col, review_col, user_threshold, item_threshold, review_length_threshold):

    user_count = Counter()
    item_count = Counter()
    with open(path) as f:
        for line in tqdm(f):
            line_content = json.loads(line)
            if len(line_content[review_col]) > review_length_threshold:
                user_count[line_content[user_col]] += 1
                item_count[line_content[item_col]] += 1

    user_candidates = {x for x in user_count if (user_count[x] >= user_threshold[0])
                       and (user_count[x] <= user_threshold[1])}
    item_candidates = {x for x in item_count if (item_count[x] >= item_threshold[0])
                       and (item_count[x] <= item_threshold[1])}

    data = []
    with open(path) as f:
        for line in tqdm(f):
            line_content = json.loads(line)
            if (line_content[user_col] in user_candidates
                and line_content[item_col] in item_candidates):
                data.append(line_content)

    return pd.DataFrame(data)