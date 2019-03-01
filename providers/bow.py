import re
import sparse
import numpy as np
import pandas as pd
from collections import Counter

import nltk
#nltk.download('averaged_perceptron_tagger')
from nltk.tag import PerceptronTagger
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tqdm import tqdm


stopwords = set(stopwords.words('english'))

grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """


def get_bow_dataframe(df, user_col, item_col, review_col, rating_col, k, implicit=False):

    tagger = PerceptronTagger()
    chunker = nltk.RegexpParser(grammar)
    pos_tag = tagger.tag

    columns_to_drop = df.columns

    df['UserID'] = df[user_col].astype('category').cat.rename_categories(range(0, df[user_col].nunique()))
    df['ItemID'] = df[item_col].astype('category').cat.rename_categories(range(0, df[item_col].nunique()))

    item_map = df[[item_col, 'ItemID']]

    # one_hot = pd.get_dummies(df[rating_col], prefix='rating')
    #
    # df = df.join(one_hot)

    reviews = df[review_col].values

    # Top-k frequent terms
    counter = Counter()
    for i, review in tqdm(enumerate(reviews)):
        counter.update(flatten([word
                                for word
                                in get_terms(chunker.parse(pos_tag(re.findall(r'\w+', review))))
                                ]))
    topk = counter.most_common(k)

    freqReview = []
    for i in tqdm(range(len(reviews))):
        tempCounter = Counter(flatten([word
                                       for word
                                       in get_terms(chunker.parse(pos_tag(re.findall(r'\w+', reviews[i]))))]))
        topkinReview = [1 if tempCounter[word] > 0 else 0 for (word, wordCount) in topk]
        freqReview.append(topkinReview)


    if implicit:
        df['Value'] = (df[rating_col] > 3)*1
    else:
        df['Value'] = df[rating_col]

    df = df.drop(columns_to_drop, axis=1)

    # Prepare freqReviewDf
    freqReviewDf = pd.DataFrame(freqReview)
    dfName = []
    for c in topk:
        dfName.append(c[0])
    freqReviewDf.columns = dfName

    df = df.join(freqReviewDf)

    #tensor = tensorfy(df, 'UserID', 'ItemID', len(df.columns)-2)

    return df, topk, item_map


def tensorfy(df, user_id, item_id, k):
    m = df[user_id].nunique()
    n = df[item_id].nunique()
    tensor = sparse.DOK((m, n, k), dtype=np.float32)

    df_numpy = df.as_matrix().astype(int)

    for row in df_numpy:
        tensor[row[0], row[1]] = row[2:]

    return sparse.COO(tensor)


# generator, generate leaves one by one
def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' or t.label() == 'JJ' or t.label() == 'RB'):
        yield subtree.leaves()


# stemming, lematizing, lower case...
def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    return word


# stop-words and length control
def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40 and word.lower() not in stopwords)
    return accepted


# generator, create item once a time
def get_terms(tree):
    for leaf in leaves(tree):
        term = [normalise(w) for w, t in leaf if acceptable_word(w)]
        # Phrase only
        if len(term) > 1:
            yield term


# Flatten phrase lists to get tokens for analysis
def flatten(npTokenList):
    finalList =[]
    for phrase in npTokenList:
        token = ''
        for word in phrase:
            token += word + ' '
        finalList.append(token.rstrip())
    return finalList
