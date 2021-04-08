# coding=utf-8
import torch
import logging
from argparse import ArgumentParser
import numpy as np

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel

'''
@inproceedings{DBLP:conf/recsys/Cheng0HSCAACCIA16,
  author    = {Heng{-}Tze Cheng and
               Levent Koc and
               Jeremiah Harmsen and
               Tal Shaked and
               Tushar Chandra and
               Hrishi Aradhye and
               Glen Anderson and
               Greg Corrado and
               Wei Chai and
               Mustafa Ispir and
               Rohan Anil and
               Zakaria Haque and
               Lichan Hong and
               Vihan Jain and
               Xiaobing Liu and
               Hemal Shah},
  title     = {Wide {\&} Deep Learning for Recommender Systems},
  booktitle = {DLRS@RecSys},
  pages     = {7--10},
  publisher = {{ACM}},
  year      = {2016}
}
'''

USER_MH = 'user_mh'
USER_NU = 'user_nm'
ITEM_MH = 'item_mh'
ITEM_NU = 'item_nm'
CTXT_MH = 'ctxt_mh'
CTXT_NU = 'ctxt_nm'


class WideDeep(RecModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--layers', type=str, default='[64]',
                            help='Hidden layers in the deep part.')
        return RecModel.add_model_specific_args(parser)

    def __init__(self, multihot_f_num: int = None, multihot_f_dim: int = None, numeric_f_num: int = None,
                 layers: str = '[64]', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihot_f_num = multihot_f_num
        self.multihot_f_dim = multihot_f_dim
        self.numeric_f_num = numeric_f_num
        self.layers = eval(layers) if type(layers) is str else layers

    def read_data(self, dataset_dir: str = None, reader=None, formatters: dict = None, *args, **kwargs):
        reader = super().read_data(dataset_dir=dataset_dir, reader=reader, formatters=formatters)
        reader.prepare_user_features(include_uid=True, multihot_features=USER_MH, numeric_features=USER_NU)
        reader.prepare_item_features(include_iid=True, multihot_features=ITEM_MH, numeric_features=ITEM_NU)
        reader.prepare_ctxt_features(include_time=False, multihot_features=CTXT_MH, numeric_features=CTXT_NU)
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
        if USER_NU in reader.user_data:
            index_dict[USER_NU] = reader.user_data[USER_NU][index_dict[UID]]
        if ITEM_MH in reader.item_data:
            index_dict[ITEM_MH] = reader.item_data[ITEM_MH][index_dict[IID]] + reader.user_multihot_f_dim
        if ITEM_NU in reader.item_data:
            index_dict[ITEM_NU] = reader.item_data[ITEM_NU][index_dict[IID]]
        if CTXT_MH in dataset.data:
            index_dict[CTXT_MH] = dataset.data[CTXT_MH][index] + reader.user_multihot_f_dim + reader.item_multihot_f_dim
        if CTXT_NU in dataset.data:
            index_dict[CTXT_NU] = dataset.data[CTXT_NU][index]
        return index_dict

    def init_modules(self, *args, **kwargs) -> None:
        self.multihot_embeddings = torch.nn.Embedding(self.multihot_f_dim, self.vec_size)
        self.multihot_bias = torch.nn.Embedding(self.multihot_f_dim, 1)
        self.numeric_bias = torch.nn.Parameter(torch.tensor([0.01] * self.numeric_f_num), requires_grad=True)
        self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        pre_size = self.multihot_f_num * self.vec_size + self.numeric_f_num
        self.deep_layers = torch.nn.ModuleList()
        for size in self.layers:
            self.deep_layers.append(torch.nn.Linear(pre_size, size))
            # self.deep_layers.append(torch.nn.BatchNorm1d(size))
            self.deep_layers.append(torch.nn.ReLU())
            self.deep_layers.append(torch.nn.Dropout(self.dropout))
            pre_size = size
        self.deep_layers.append(torch.nn.Linear(pre_size, 1))
        self.init_weights()
        return

    def concat_features(self, batch):
        i_ids = batch[IID]  # B * S
        sample_n = i_ids.size(-1)  # =S
        mh_features, nu_features = [], []
        if USER_MH in batch:
            mh_features.append(batch[USER_MH].expand(-1, sample_n, -1))  # B * S * uf
        if USER_NU in batch:
            nu_features.append(batch[USER_NU].expand(-1, sample_n, -1))  # B * S * unu
        if ITEM_MH in batch:
            mh_features.append(batch[ITEM_MH])  # B * S * if
        if ITEM_NU in batch:
            nu_features.append(batch[ITEM_NU])  # B * S * inu
        if CTXT_MH in batch:
            mh_features.append(batch[CTXT_MH].unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * S * cf
        if CTXT_NU in batch:
            nu_features.append(batch[CTXT_NU].unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * S * cnu
        mh_features = torch.cat(mh_features, dim=-1) if len(mh_features) > 0 else None  # B * S * mh
        nu_features = torch.cat(nu_features, dim=-1) if len(nu_features) > 0 else None  # B * S * nu
        return mh_features, nu_features

    def forward(self, batch, *args, **kwargs):
        mh_features, nu_features = self.concat_features(batch=batch)  # B * S * mh, B * S * nu
        deep_vectors = self.multihot_embeddings(mh_features).flatten(start_dim=-2)  # B * S * (mh*v)
        wide_prediction = self.overall_bias + self.multihot_bias(mh_features).squeeze(dim=-1).sum(dim=-1)  # B * S
        if nu_features is not None:
            deep_vectors = torch.cat([deep_vectors, nu_features], dim=-1)  # B * S * (mh*v+nu)
            wide_prediction = wide_prediction + (nu_features * self.numeric_bias).sum(dim=-1)  # B * S
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # B * S * 1
        deep_prediction = deep_vectors.squeeze(dim=-1)  # B * S
        prediction = wide_prediction + deep_prediction  # B * S
        batch[PREDICTION] = prediction
        return batch
