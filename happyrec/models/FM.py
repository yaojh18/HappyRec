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


class FM(RecModel):
    """
    不要作为模型使用，本模型作为公共基类，主要为了简化代码
    使用前请务必初始化self.first_layer_size
    """

    def __init__(self, multihot_f_num: int = None, multihot_f_dim: int = None, numeric_f_num: int = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihot_f_num = multihot_f_num
        self.multihot_f_dim = multihot_f_dim
        self.numeric_f_num = numeric_f_num

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
        self.multihot_embeddings = torch.nn.Embedding(self.multihot_f_dim, self.vec_size)
        self.numeric_embeddings = torch.nn.Parameter(torch.Tensor(self.numeric_f_num, self.vec_size),
                                                     requires_grad=True)
        self.apply(self.init_weights)
        self.init_weights(self.numeric_embeddings)
        self.init_weights(self.numeric_bias)

    def get_embeddings(self, batch):
        i_ids = batch[IID]  # B * S
        sample_n = i_ids.size(-1)  # =S
        vectors = []
        numeric_vectors = []
        bias = []
        if USER_MH in batch:
            user_mh_vectors = self.multihot_embeddings(batch[USER_MH].long())  # B * 1 * uf * v
            vectors.append(user_mh_vectors.expand(-1, sample_n, -1, -1))  # B * S * uf * v
            bias.append(self.multihot_bias(batch[USER_MH].long()).squeeze(dim=-1).expand(-1, sample_n, -1))  # B * 1 * uf
        if USER_NU in batch:
            numeric_vectors.append(batch[USER_NU].expand(-1, sample_n, -1))  # B * S * unm
        if ITEM_MH in batch:
            item_mh_vectors = self.multihot_embeddings(batch[ITEM_MH].long())  # B * S * if * v
            vectors.append(item_mh_vectors)  # B * S * if * v
            bias.append(self.multihot_bias(batch[ITEM_MH].long()).squeeze(dim=-1))  # B * S * i
        if ITEM_NU in batch:
            numeric_vectors.append(batch[ITEM_NU])  # B * S * inm
        if CTXT_MH in batch:
            ctxt_mh_vectors = self.multihot_embeddings(batch[CTXT_MH].long())  # B * cf * v
            vectors.append(ctxt_mh_vectors.unsqueeze(dim=1).expand(-1, sample_n, -1, -1))  # B * S * cf * v
            bias.append(self.multihot_bias(batch[CTXT_MH].long()).squeeze(dim=-1).unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * cf
        if CTXT_NU in batch:
            numeric_vectors.append(batch[CTXT_NU].unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * S * cnm
        if self.numeric_f_num > 0:
            numeric_vectors = torch.cat(numeric_vectors, dim=-1)  # B * S * nm
            bias.append(numeric_vectors * self.numeric_bias)  # B * S * nm
            numeric_vectors = numeric_vectors.unsqueeze(dim=-1) * self.numeric_embeddings  # B * S * nm * v
            vectors.append(numeric_vectors)
        vectors = torch.cat(vectors, dim=-2)  # B * S * f * v
        bias = torch.cat(bias, dim=-1)
        return vectors, bias