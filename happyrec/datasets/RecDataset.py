# coding=utf-8

import torch
import numpy as np
from tqdm import tqdm

from ..datasets.Dataset import Dataset
from ..configs.constants import *
from ..utilities.formatter import pad_array
from ..utilities.rec import sample_iids


class RecDataset(Dataset):
    def __init__(self, model, phase: int, buffer_ds: int = 0,
                 *args, **kwargs):
        self.model = model
        self.phase = phase
        self.buffer_ds = 0 if model.train_sample_n > 0 and phase == TRAIN_PHASE else buffer_ds
        self.buffer_train_iids = None
        Dataset.__init__(self, model=model, phase=phase, buffer_ds=self.buffer_ds, *args, **kwargs)

    def get_interaction(self, index_dict: dict, index: int) -> dict:
        for c in [LABEL, UID, IID, TIME]:
            if c in self.data:
                index_dict[c] = [self.data[c][index]]
        return index_dict

    def sample_train_iids(self, sample_n: int):
        uids = self.data[UID] if UID in self.data else None
        if uids is not None:
            tran_pos_his = {uid: self.reader.user_data[TRAIN_POS_HIS][uid] for uid in uids}
            iids = sample_iids(sample_n=sample_n, uids=uids, item_num=self.reader.item_num,
                               exclude_iids=tran_pos_his, replace=False, verbose=True)
        else:
            iids = np.random.choice(self.reader.item_num, size=(len(self), sample_n), replace=False)
        self.buffer_train_iids = iids
        return iids

    def extend_train_iids(self, index_dict: dict, index: int, label: int = 0) -> dict:
        iids = self.buffer_train_iids[index]
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
        if UID in index_dict:
            tran_pos_his = self.reader.user_data[TRAIN_POS_HIS][index_dict[UID][0]]
            iids[tran_pos_his] = 0
            if LABEL in index_dict:
                iids[index_dict[IID]] = index_dict[IID]
            if EVAL_LABELS in self.data:
                iids[self.data[EVAL_IIDS][index]] = self.data[EVAL_IIDS][index]
        if LABEL in index_dict:
            labels[index_dict[IID]] = index_dict[LABEL]
        if EVAL_LABELS in self.data:
            labels[self.data[EVAL_IIDS][index]] = self.data[EVAL_LABELS][index]
        index_dict[IID] = iids
        index_dict[LABEL] = labels
        return index_dict
