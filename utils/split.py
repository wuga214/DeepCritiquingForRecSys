def leave_one_out_split(df, user_col, ratio, random_state=None):
    grouped = df.groupby(user_col, as_index=False)
    valid = grouped.apply(lambda x: x.sample(frac=ratio, random_state=random_state))
    train = df.loc[~df.index.isin([x[1] for x in valid.index])]
    return train, valid

