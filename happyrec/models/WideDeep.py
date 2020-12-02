# coding=utf-8
import torch
import logging
from argparse import ArgumentParser
import numpy as np

from ..data_readers.DataReader import DataReader
from ..data_readers.RecReader import RecReader
from ..datasets.Dataset import Dataset
from ..datasets.RecDataset import RecDataset
from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel
from ..utilities.formatter import split_seq
from ..modules.loss import BPRRankLoss
from ..metrics.MetricList import MetricsList
from ..metrics.RankMetricList import RankMetricsList
from ..metrics.metrics import METRICS_SMALLER

USER_MH = 'user_mh'
USER_NM = 'user_nm'
ITEM_MH = 'item_mh'
ITEM_NM = 'item_nm'
CTXT_MH = 'ctxt_mh'
CTXT_NM = 'ctxt_nm'


class WideDeep(RecModel):
    @staticmethod
    def parse_model_args(parent_parser):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        parser.add_argument('--layers', type=str, default='[64]',
                            help='Hidden layers in the deep part.')
        return RecModel.parse_model_args(parser)

    def __init__(self, multihot_f_num: int = None, multihot_f_dim: int = None, numeric_f_num: int = None,
                 layers: str = '[64]', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihot_f_num = multihot_f_num
        self.multihot_f_dim = multihot_f_dim
        self.numeric_f_num = numeric_f_num
        self.layers = eval(layers) if type(layers) is str else layers

    def read_data(self, dataset_dir: str = None, reader=None, formatters: dict = None, *args, **kwargs):
        reader = super().read_data(dataset_dir=dataset_dir, reader=reader, formatters=formatters)
        reader.prepare_user_features(include_uid=True, multihot_features=USER_MH, numeric_features=USER_NM)
        reader.prepare_item_features(include_iid=True, multihot_features=ITEM_MH, numeric_features=ITEM_NM)
        reader.prepare_ctxt_features(include_time=False, multihot_features=CTXT_MH, numeric_features=CTXT_NM)
        self.multihot_f_num = reader.user_multihot_f_num + reader.item_multihot_f_num + reader.ctxt_multihot_f_num
        self.multihot_f_dim = reader.user_multihot_f_dim + reader.item_multihot_f_dim + reader.ctxt_multihot_f_dim
        self.numeric_f_num = reader.user_numeric_f_num + reader.item_numeric_f_num + reader.ctxt_numeric_f_num
        return reader

    def read_formatters(self, formatters=None) -> dict:
        current = {
            '^({}|{}|{}).*({}|{}|{})$'.format(USER_F, ITEM_F, CTXT_F, INT_F, FLOAT_F, CAT_F): None
        }
        if formatters is None:
            return super().read_formatters(current)
        return super().read_formatters({**current, **formatters})

    def dataset_get_item(self, dataset, index: int) -> dict:
        if dataset.buffer_ds > 0: return dataset.index_buffer[index]
        index_dict = super().dataset_get_item(dataset=dataset, index=index)
        reader = dataset.reader
        if USER_MH in reader.user_data:
            index_dict[USER_MH] = reader.user_data[USER_MH][index_dict[UID]]
        if USER_NM in reader.user_data:
            index_dict[USER_NM] = reader.user_data[USER_NM][index_dict[UID]]
        if ITEM_MH in reader.item_data:
            index_dict[ITEM_MH] = reader.item_data[ITEM_MH][index_dict[IID]] + reader.user_multihot_f_dim
        if ITEM_NM in reader.item_data:
            index_dict[ITEM_NM] = reader.item_data[ITEM_NM][index_dict[IID]]
        if CTXT_MH in dataset.data:
            index_dict[CTXT_MH] = dataset.data[CTXT_MH][index] + reader.user_multihot_f_dim + reader.item_multihot_f_dim
        if CTXT_NM in dataset.data:
            index_dict[CTXT_NM] = dataset.data[CTXT_NM][index]
        return index_dict

    # def dataset_collate_batch(self, dataset, batch: list) -> dict:
    #     result = {}
    #     for c in [LABEL, IID]:
    #         if c in batch[0]:
    #             result[c] = dataset.collate_padding([b[c] for b in batch], padding=0)
    #     for c in [UID, TIME, USER_MH, USER_NM, ITEM_MH, ITEM_NM, CTXT_MH, CTXT_NM]:
    #         if c in batch[0]:
    #             result[c] = dataset.collate_stack([b[c] for b in batch])
    #     return result

    def init_modules(self, *args, **kwargs) -> None:
        self.feature_embeddings = torch.nn.Embedding(self.multihot_f_dim, self.vec_size)
        self.multihot_bias = torch.nn.Embedding(self.multihot_f_dim, 1)
        self.numeric_bias = torch.nn.Parameter(torch.tensor([0.1] * self.numeric_f_num), requires_grad=True)
        pre_size = self.multihot_f_num * self.vec_size + self.numeric_f_num
        self.deep_layers = torch.nn.ModuleList()
        for size in self.layers:
            self.deep_layers.append(torch.nn.Linear(pre_size, size))
            # self.deep_layers.append(torch.nn.BatchNorm1d(size))
            self.deep_layers.append(torch.nn.ReLU())
            self.deep_layers.append(torch.nn.Dropout(self.dropout))
            pre_size = size
        self.deep_layers.append(torch.nn.Linear(pre_size, 1))
        self.apply(self.init_weights)
        return

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        sample_n = i_ids.size(-1)  # =S

        wide_prediction = 0  # B * S
        deep_vectors, wide_numeric = [], []
        if USER_MH in batch:
            user_mh_vectors = self.feature_embeddings(batch[USER_MH])  # B * 1 * uf * v
            user_mh_vectors = user_mh_vectors.flatten(start_dim=-2).expand(-1, sample_n, -1)  # B * S * (uf*v)
            deep_vectors.append(user_mh_vectors)
            user_mh_bias = self.multihot_bias(batch[USER_MH]).squeeze(dim=-1)  # B * 1 * uf
            wide_prediction += user_mh_bias.sum(dim=-1)  # B * S
        if USER_NM in batch:
            wide_numeric.append(batch[USER_NM].expand(-1, sample_n, -1))  # B * S * unm
        if ITEM_MH in batch:
            item_mh_vectors = self.feature_embeddings(batch[ITEM_MH])  # B * S * if * v
            item_mh_vectors = item_mh_vectors.flatten(start_dim=-2)  # B * S * (if*v)
            deep_vectors.append(item_mh_vectors)
            item_mh_bias = self.multihot_bias(batch[ITEM_MH]).squeeze(dim=-1)  # B * S * if
            wide_prediction = wide_prediction + item_mh_bias.sum(dim=-1)  # B * S
        if ITEM_NM in batch:
            wide_numeric.append(batch[ITEM_NM])  # B * S * inm
        if CTXT_MH in batch:
            ctxt_mh_vectors = self.feature_embeddings(batch[CTXT_MH])  # B * cf * v
            ctxt_mh_vectors = ctxt_mh_vectors.flatten(start_dim=-2). \
                unsqueeze(dim=1).expand(-1, sample_n, -1)  # B * S * (cf*v)
            deep_vectors.append(ctxt_mh_vectors)
            ctxt_mh_bias = self.multihot_bias(batch[CTXT_MH]).squeeze(dim=-1)  # B * cf
            wide_prediction += ctxt_mh_bias.sum(dim=-1, keepdim=True)  # B * S
        if CTXT_NM in batch:
            wide_numeric.append(batch[CTXT_NM].unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * S * cnm
        if self.numeric_f_num > 0:
            wide_numeric = torch.cat(wide_numeric, dim=-1)  # B * S * nm
            deep_vectors.append(wide_numeric)
            wide_numeric = wide_numeric * self.numeric_bias  # B * S * nm
            wide_prediction += wide_numeric.sum(dim=-1)  # B * S
        deep_vectors = torch.cat(deep_vectors, dim=-1).flatten(start_dim=0, end_dim=1)  # (B*S) * fv
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # (B*S) * 1
        deep_prediction = deep_vectors.squeeze(dim=-1).view_as(i_ids)  # B * S
        prediction = wide_prediction + deep_prediction  # B * S
        out_dict = {PREDICTION: prediction}
        return out_dict
