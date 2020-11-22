# coding=utf-8

import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from ..datasets.Dataset import Dataset
from ..configs.constants import *


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
        index_dict = self.index_extend_key(index_dict, key=IID, data=iids)
        labels = np.zeros(len(iids), dtype=int) if EVAL_LABELS not in self.data \
            else self.data[EVAL_LABELS][index][:sample_n]
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
