# coding=utf-8
from argparse import ArgumentParser

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel

USER_MH = 'user_mh'
USER_NU = 'user_nm'
ITEM_MH = 'item_mh'
ITEM_NU = 'item_nm'
CTXT_MH = 'ctxt_mh'
CTXT_NU = 'ctxt_nm'


class DeepModel(RecModel):
    """
    不要作为模型使用，本模型作为公共基类，主要为了简化代码
    使用前请务必初始化self.first_layer_size
    """
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
                 first_layer_size: int = None, layers: str = '[64]', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihot_f_num = multihot_f_num
        self.multihot_f_dim = multihot_f_dim
        self.numeric_f_num = numeric_f_num
        self.first_layer_size = first_layer_size
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
        self.multihot_bias = torch.nn.Embedding(self.multihot_f_dim, 1)
        self.numeric_bias = torch.nn.Parameter(torch.tensor([0.1] * self.numeric_f_num), requires_grad=True)
        self.deep_layers = torch.nn.ModuleList()
        if self.first_layer_size:
            pre_size = self.first_layer_size
            for size in self.layers:
                self.deep_layers.append(torch.nn.Linear(pre_size, size))
                # self.deep_layers.append(torch.nn.BatchNorm1d(size))
                self.deep_layers.append(torch.nn.ReLU())
                self.deep_layers.append(torch.nn.Dropout(self.dropout))
                pre_size = size
            self.deep_layers.append(torch.nn.Linear(pre_size, 1))
        self.apply(self.init_weights)
        self.init_weights(self.numeric_bias)
        return