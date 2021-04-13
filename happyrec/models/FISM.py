# coding=utf-8
import torch

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel
from argparse import ArgumentParser

'''
@inproceedings{DBLP:conf/kdd/KabburNK13,
  author    = {Santosh Kabbur and
               Xia Ning and
               George Karypis},
  title     = {{FISM:} factored item similarity models for top-N recommender systems},
  booktitle = {{KDD}},
  pages     = {659--667},
  publisher = {{ACM}},
  year      = {2013}
}
'''


class FISM(RecModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='The negative exponent alpha of (num+ - 1).')
        return RecModel.add_model_specific_args(parser)

    def __init__(self, alpha: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def read_data(self, dataset_dir: str = None, reader=None, formatters: dict = None, *args, **kwargs):
        reader = super().read_data(dataset_dir=dataset_dir, reader=reader, formatters=formatters)
        reader.prepare_user_pos_his()
        return reader

    def dataset_get_item(self, dataset, index: int) -> dict:
        if dataset.buffer_ds > 0: return dataset.index_buffer[index]
        index_dict = super().dataset_get_item(dataset=dataset, index=index)
        uid = index_dict[UID][0]
        if dataset.phase == TRAIN_PHASE:
            hi = dataset.reader.user_data[HIS_POS_TRAIN][uid]
        else:
            hi = dataset.data[HIS_POS_IDX][index]
        index_dict[HIS_POS_SEQ] = dataset.reader.user_data[HIS_POS_SEQ][uid][0:hi]
        return index_dict

    def dataset_collate_batch(self, dataset, batch: list) -> dict:
        uhis = [b.pop(HIS_POS_SEQ) for b in batch]
        uhis = dataset.collate_padding(uhis)
        result_dict = super().dataset_collate_batch(dataset, batch)
        result_dict[HIS_POS_SEQ] = uhis
        return result_dict

    def init_modules(self, *args, **kwargs) -> None:
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.vec_size)
        self.iid_embeddings_his = torch.nn.Embedding(self.item_num, self.vec_size)
        self.init_weights()
        return

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        u_ids = batch[UID]  # B * 1
        u_his = batch[HIS_POS_SEQ]  # B * l
        # item representation
        i_vectors = self.iid_embeddings(i_ids)  # B * S * v

        # user all history
        u_his_vectors = self.iid_embeddings_his(u_his)  # B * l * v
        u_his_valid = u_his.gt(0).float()  # B * l
        u_his_vectors = u_his_vectors * u_his_valid.unsqueeze(-1)  # B * l * v
        u_vectors = u_his_vectors.sum(dim=1, keepdim=True)  # B * 1 * v
        u_his_num = u_his_valid.sum(dim=-1, keepdim=True)  # B * 1

        # remove current pos inter when training
        if batch[PHASE] == TRAIN_PHASE:
            contain = i_ids.unsqueeze(dim=-1).eq(u_his.unsqueeze(dim=1)).float()  # B * S * l
            contain = contain.sum(dim=-1).gt(0).float() * i_ids.gt(0).float()  # B * S
            i_vectors_his = self.iid_embeddings_his(i_ids)  # B * S * v
            u_vectors = u_vectors - i_vectors_his * contain.unsqueeze(dim=-1)  # B * S * v
            u_his_num = u_his_num - contain  # B * S

        # prediction
        prediction = (u_vectors * i_vectors).sum(dim=-1)  # B * S
        u_bias = self.user_bias(u_ids).squeeze(dim=-1)  # B * 1
        i_bias = self.item_bias(i_ids).squeeze(dim=-1)  # B * S
        u_his_num = u_his_num + u_his_num.le(0).float()  # B * 1/S
        prediction = u_bias + i_bias + prediction * u_his_num.pow(-self.alpha)  # B * S
        batch[PREDICTION] = prediction
        return batch
