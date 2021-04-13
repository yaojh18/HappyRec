# coding=utf-8
import torch
from argparse import ArgumentParser

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel

'''
@inproceedings{DBLP:conf/www/HeLZNHC17,
  author    = {Xiangnan He and
               Lizi Liao and
               Hanwang Zhang and
               Liqiang Nie and
               Xia Hu and
               Tat{-}Seng Chua},
  title     = {Neural Collaborative Filtering},
  booktitle = {{WWW}},
  pages     = {173--182},
  publisher = {{ACM}},
  year      = {2017}
}
'''


class NeuMF(RecModel):
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
                            help='Hidden layers in MLP.')
        return RecModel.add_model_specific_args(parser)

    def __init__(self, layers: str = '[64]', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = eval(layers) if type(layers) is str else layers

    def init_modules(self, *args, **kwargs) -> None:
        self.mf_uid_embeddings = torch.nn.Embedding(self.user_num, self.vec_size)
        self.mf_iid_embeddings = torch.nn.Embedding(self.item_num, self.vec_size)
        self.mlp_uid_embeddings = torch.nn.Embedding(self.user_num, self.vec_size)
        self.mlp_iid_embeddings = torch.nn.Embedding(self.item_num, self.vec_size)

        pre_size = 2 * self.vec_size
        self.mlp_layers = torch.nn.ModuleList()
        for size in self.layers:
            self.mlp_layers.append(torch.nn.Linear(pre_size, size))
            self.mlp_layers.append(torch.nn.ReLU())
            self.mlp_layers.append(torch.nn.Dropout(self.dropout))
            pre_size = size
        pre_size += self.vec_size
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(pre_size, 1),
        )
        self.init_weights()
        return

    def forward(self, batch, *args, **kwargs):
        u_ids = batch[UID]  # B * 1
        i_ids = batch[IID]  # B * S
        mf_u_vectors = self.mf_uid_embeddings(u_ids)  # B * 1 * v
        mf_i_vectors = self.mf_iid_embeddings(i_ids)  # B * S * v
        mlp_u_vectors = self.mlp_uid_embeddings(u_ids)  # B * 1 * v
        mlp_i_vectors = self.mlp_iid_embeddings(i_ids)  # B * S * v
        gmf_vectors = mf_u_vectors * mf_i_vectors  # B * S * v
        mlp_vectors = torch.cat((mlp_u_vectors.expand_as(mlp_i_vectors), mlp_i_vectors), dim=-1)  # B * S * 2v
        for layer in self.mlp_layers:
            mlp_vectors = layer(mlp_vectors)  # B * S * x
        neu_mf_vectors = torch.cat((gmf_vectors, mlp_vectors), dim=-1)  # B * S * (v + x)
        prediction = self.output_layer(neu_mf_vectors).squeeze()  # B * S
        batch[PREDICTION] = prediction
        return batch
