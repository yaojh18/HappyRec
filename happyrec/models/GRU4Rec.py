# coding=utf-8
import torch
import logging
from argparse import ArgumentParser
import numpy as np

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel
from ..modules.rnn import GRU


class GRU4Rec(RecModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_his', type=int, default=-1,
                            help='Max history length. All his if max_his < 0')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of GRU layers.')
        return RecModel.add_model_specific_args(parser)

    def __init__(self, max_his: int = -1, num_layers: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_his = max_his
        self.num_layers = num_layers

    def read_data(self, dataset_dir: str = None, reader=None, formatters: dict = None, *args, **kwargs):
        reader = super().read_data(dataset_dir=dataset_dir, reader=reader, formatters=formatters)
        if self.max_his != 0:
            reader.prepare_user_pos_his()
        return reader

    def dataset_get_item(self, dataset, index: int) -> dict:
        if dataset.buffer_ds > 0: return dataset.index_buffer[index]
        index_dict = super().dataset_get_item(dataset=dataset, index=index)
        hi = dataset.data[HIS_POS_IDX][index]
        lo = 0 if self.max_his < 0 else max(0, hi - self.max_his)
        index_dict[HIS_POS_SEQ] = dataset.reader.user_data[HIS_POS_SEQ][index_dict[UID][0]][lo:hi]
        return index_dict

    def dataset_collate_batch(self, dataset, batch: list) -> dict:
        uhis = [b.pop(HIS_POS_SEQ) for b in batch]
        uhis = dataset.collate_padding(uhis)
        result_dict = super().dataset_collate_batch(dataset, batch)
        result_dict[HIS_POS_SEQ] = uhis
        return result_dict

    def init_modules(self, *args, **kwargs) -> None:
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.vec_size)
        self.gru = GRU(vec_size=self.vec_size, hidden_size=self.vec_size,
                       num_layers=self.num_layers, dropout=self.dropout)
        self.apply(self.init_weights)
        return

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        u_his = batch[HIS_POS_SEQ]  # B * l
        i_vectors = self.iid_embeddings(i_ids)  # B * S * v
        u_his_vectors = self.iid_embeddings(u_his)  # B * l * v
        output, hidden = self.gru(seq_vectors=u_his_vectors, valid=u_his.gt(0).byte())
        u_vectors = hidden[0].unsqueeze(dim=1)  # B * 1 * v
        prediction = (u_vectors * i_vectors).sum(dim=-1)  # B * S
        batch[PREDICTION] = prediction
        return batch
