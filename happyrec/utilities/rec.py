# coding=utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..configs.settings import *
from ..configs.constants import *


def group_user_history(uids, iids) -> dict:
    user_dict = {}
    for uid, iid in zip(uids, iids):
        if uid not in user_dict:
            user_dict[uid] = []
        user_dict[uid].append(iid)
    return user_dict


def group_user_history_df(df, label_filter=lambda x: x > 0) -> dict:
    dfs = [df] if type(df) is pd.DataFrame else df
    dfs = [df[df[LABEL].apply(label_filter)] if LABEL in df else df for df in dfs]
    df = pd.concat([df[[UID, IID]] for df in dfs], ignore_index=True)
    return group_user_history(uids=df[UID].values, iids=df[IID].values)


def sample_iids(sample_n, uids, item_num, exclude_iids=None, replace=False, verbose=True):
    if verbose:
        uids = tqdm(uids, dynamic_ncols=True, mininterval=1, leave=False, desc='sample_iids')

    global_exclude = None if type(exclude_iids) is dict \
        else set([]) if exclude_iids is None else set(exclude_iids)
    if global_exclude is None:
        exclude_iids = {k: set(exclude_iids[k]) for k in exclude_iids}

    iid_buffer = np.random.randint(low=0, high=item_num, size=sample_n * len(uids))
    buffer_idx = 0
    result = []
    for uid in uids:
        exclude = global_exclude if global_exclude is not None \
            else exclude_iids[uid] if uid in exclude_iids else set([])
        if not replace and item_num - len(exclude) < sample_n:
            uid_result = [i for i in range(item_num) if i not in exclude]
            while len(uid_result) != sample_n:
                uid_result.append(0)
            result.append(uid_result)
            continue
        uid_result = []
        tmp_set = set([])
        while len(uid_result) < sample_n:
            if len(iid_buffer) <= buffer_idx:
                iid_buffer = np.random.randint(low=0, high=item_num, size=sample_n * len(uids))
                buffer_idx = 0
            iid = iid_buffer[buffer_idx]
            buffer_idx += 1
            if iid not in exclude and (replace or iid not in tmp_set):
                uid_result.append(iid)
                tmp_set.add(iid)
        result.append(uid_result)
    return np.array(result)
