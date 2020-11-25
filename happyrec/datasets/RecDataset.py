# coding=utf-8

import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from ..datasets.Dataset import Dataset
from ..configs.constants import *
from ..utilities.formatter import pad_array


class RecDataset(Dataset):
    def __init__(self, model, phase: int, buffer_ds: int = 0,
                 *args, **kwargs):
        self.model = model
        self.phase = phase
        self.buffer_ds = 0 if model.train_sample_n > 0 and phase == TRAIN_PHASE else buffer_ds
        Dataset.__init__(self, model=model, phase=phase, buffer_ds=self.buffer_ds, *args, **kwargs)

    def get_interaction(self, index_dict: dict, index: int) -> dict:
        for c in [LABEL, UID, IID, TIME]:
            if c in self.data:
                index_dict[c] = [self.data[c][index]]
        return index_dict

    def sample_iids(self, index_dict: dict, sample_n: int, label: int = 0) -> dict:
        iids = np.random.choice(self.reader.item_num, size=sample_n, replace=False)
        index_dict = self.index_extend_key(index_dict, key=IID, data=iids)
        index_dict = self.index_extend_key(index_dict, key=LABEL, data=np.array([label] * len(iids), dtype=int))
        return index_dict

    def extend_eval_iids(self, index_dict: dict, index: int, sample_n: int) -> dict:
        iids = self.data[EVAL_IIDS][index][:sample_n]
        iids = pad_array(iids, max_len=sample_n, v=0)
        index_dict = self.index_extend_key(index_dict, key=IID, data=iids)
        labels = np.zeros(len(iids), dtype=int) if EVAL_LABELS not in self.data \
            else self.data[EVAL_LABELS][index][:sample_n]
        labels = pad_array(labels, max_len=sample_n, v=0)
        index_dict = self.index_extend_key(index_dict, key=LABEL, data=labels)
        return index_dict

    def eval_all_iids(self, index_dict: dict, index: int):
        iids = np.arange(self.reader.item_num)
        labels = np.zeros(len(iids), dtype=int)
        if LABEL in index_dict:
            labels[index_dict[IID]] = index_dict[LABEL]
        if EVAL_LABELS in self.data:
            labels[self.data[EVAL_IIDS][index]] = self.data[EVAL_LABELS][index]
        index_dict[IID] = iids
        index_dict[LABEL] = labels
        return index_dict

    # def get_user_features(self, index_dict: dict, float_f: bool = True, int_f: bool = True, cat_f: bool = True):
    #     uids = index_dict[UID]
    #     u_fs, u_fvs = [], []
    #     for k in self.reader.user_features:
    #         lo, hi = self.reader.user_features[k]
    #         f_data = self.reader.user_data[k]
    #         if cat_f and k.endswith(CAT_F):
    #             u_fs.append(f_data[uids] + lo)
    #             if float_f or int_f:
    #                 u_fvs.append(np.ones_like(uids))
    #         if (float_f and k.endswith(FLOAT_F)) or (int_f and k.endswith(INT_F)):
    #             u_fs.append(np.zeros_like(uids) + lo)
    #             u_fvs.append(f_data[uids])
    #     if len(u_fs) > 0:
    #         index_dict[USER_F] = np.stack(u_fs, axis=-1)
    #         if float_f or int_f:
    #             index_dict[USER_FV] = np.stack(u_fvs, axis=-1)
    #     return index_dict
    #
    # def get_item_features(self, index_dict: dict, float_f: bool = True, int_f: bool = True, cat_f: bool = True):
    #     iids = index_dict[IID]
    #     i_fs, i_fvs = [], []
    #     for k in self.reader.item_features:
    #         lo, hi = self.reader.item_features[k]
    #         f_data = self.reader.item_data[k]
    #         if cat_f and k.endswith(CAT_F):
    #             i_fs.append(f_data[iids] + lo)
    #             if float_f or int_f:
    #                 i_fvs.append(np.ones_like(iids))
    #         if (float_f and k.endswith(FLOAT_F)) or (int_f and k.endswith(INT_F)):
    #             i_fs.append(np.zeros_like(iids) + lo)
    #             i_fvs.append(f_data[iids])
    #     if len(i_fs) > 0:
    #         index_dict[ITEM_F] = np.stack(i_fs, axis=-1)
    #         if float_f or int_f:
    #             index_dict[ITEM_FV] = np.stack(i_fvs, axis=-1)
    #     return index_dict
    #
    # def get_ctxt_features(self, index_dict: dict, index: int,
    #                       float_f: bool = True, int_f: bool = True, cat_f: bool = True):
    #     c_fs, c_fvs = [], []
    #     for k in self.reader.ctxt_features:
    #         lo, hi = self.reader.ctxt_features[k]
    #         f_data = self.data[k]
    #         if cat_f and k.endswith(CAT_F):
    #             c_fs.append(f_data[index] + lo)
    #             if float_f or int_f:
    #                 c_fvs.append(1)
    #         if (float_f and k.endswith(FLOAT_F)) or (int_f and k.endswith(INT_F)):
    #             c_fs.append(lo)
    #             c_fvs.append(f_data[index])
    #     if len(c_fs):
    #         index_dict[CTXT_F] = np.array(c_fs)
    #         if float_f or int_f:
    #             index_dict[CTXT_FV] = np.array(c_fvs)
    #     return index_dict
