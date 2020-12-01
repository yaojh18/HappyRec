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

# def sample_iids(sample_n, index, iids, exclude_iids=None, iids_p=None, replace=False, verbose=True):
#     '''
#     TODO: need test whether exist bugs
#     :param sample_n:
#     :param index:
#     :param iids:
#     :param exclude_iids:
#     :param iids_p:
#     :param replace:
#     :param verbose:
#     :return:
#     '''
#     if verbose:
#         index = tqdm(index, dynamic_ncols=True, mininterval=1, leave=False, desc='sample_iids')
#
#     # if dict then None else a array
#     global_candidate = None if type(iids) is dict \
#         else np.arange(iids) if type(iids) is int or type(iids) is np.int else np.array(iids)
#     # if dict then None else a set
#     global_exclude = None if type(exclude_iids) is dict \
#         else set([]) if exclude_iids is None else set(exclude_iids)
#     # if dict then None else a array
#     global_p = None if type(iids_p) is dict else iids_p
#     if global_candidate is not None and global_p is not None:
#         assert len(global_p) == len(global_candidate)
#     if global_candidate is not None and global_exclude is not None:
#         if len(global_exclude) > 0:
#             if global_p is not None:
#                 global_p = np.array(
#                     [global_p[i] for i in range(len(global_p)) if global_candidate[i] not in global_exclude])
#             global_candidate = np.array([i for i in global_candidate if i not in global_exclude])
#             global_exclude = set([])
#         if not replace and len(global_candidate) <= sample_n:
#             sup_n = sample_n - len(global_candidate)
#             return np.array([np.concatenate([global_candidate, np.zeros(sup_n, dtype=int)])] * len(index))
#
#     result = []
#     for idx in index:
#         exclude = global_exclude if global_exclude is not None \
#             else set(exclude_iids[idx]) if idx in exclude_iids else set([])
#         candidate = global_candidate if global_candidate is not None else iids[idx]
#         if type(iids_p) is dict:
#             p = np.array(iids_p[idx]) if idx in iids_p else None
#             assert len(p) == len(candidate) or p is None
#         elif type(iids_p) is None:
#             p = None
#         elif global_candidate is None:
#             p = np.array([global_p[i] if i not in exclude else 0 for i in candidate])
#             exclude = set([])
#         else:
#             p = global_p
#         if len(exclude) <= 0:
#             result.append(np.random.choice(a=candidate, size=sample_n, p=p, replace=replace))
#         elif len(candidate) - len(exclude) < 2 * sample_n:
#             if p is not None:
#                 p = np.array([p[i] for i in range(len(p)) if candidate[i] not in exclude])
#             candidate = np.array([iid for iid in candidate if iid not in exclude])
#             if not replace and len(candidate) <= sample_n:
#                 sup_n = sample_n - len(candidate)
#                 result.append(np.concatenate([candidate, np.zeros(sup_n, dtype=int)]))
#             else:
#                 result.append(np.random.choice(a=candidate, size=sample_n, p=p, replace=replace))
#         else:
#             idx_result, tmp_set = [], set([])
#             iid_buffer, buffer_idx = [], 1
#             while len(idx_result) < sample_n:
#                 if len(iid_buffer) <= buffer_idx:
#                     iid_buffer = np.random.choice(a=candidate, size=len(candidate), p=p, replace=replace)
#                     buffer_idx = 0
#                 iid = iid_buffer[buffer_idx]
#                 buffer_idx += 1
#                 if iid not in exclude and (replace or iid not in tmp_set):
#                     idx_result.append(iid)
#                     tmp_set.add(iid)
#             result.append(idx_result)
#     return np.array(result)
