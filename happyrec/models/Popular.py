# coding=utf-8
import torch
import numpy as np
from collections import defaultdict

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel


class Popular(RecModel):
    def __init__(self, train_sample_n: int = 0, es_patience: int = 0, *args, **kwargs):
        super(Popular, self).__init__(train_sample_n=0, es_patience=0, *args, **kwargs)

    def read_data(self, dataset_dir: str = None, reader=None, formatters: dict = None):
        reader = super().read_data(dataset_dir=dataset_dir, reader=reader, formatters=formatters)
        reader.prepare_item_pos_his()

    def on_train_epoch_start(self) -> None:
        return

    def dataset_get_item(self, dataset, index: int) -> dict:
        if dataset.buffer_ds > 0: return dataset.index_buffer[index]
        index_dict = super().dataset_get_item(dataset=dataset, index=index)
        reader = dataset.reader
        index_dict[PREDICTION] = np.array(
            [reader.item_data[HIS_POS_TRAIN][iid] for iid in index_dict[IID]])
        return index_dict

    def init_modules(self, *args, **kwargs) -> None:
        return

    def forward(self, batch, *args, **kwargs):
        return batch

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        return

    def training_step(self, batch, batch_idx, *args, **kwargs):
        out_dict = self.forward(batch)
        if self.train_metrics is not None:
            self.train_metrics.update(out_dict)
        out_dict[LOSS] = torch.tensor(0)
        return out_dict
