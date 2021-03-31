# coding=utf-8
import torch

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel
from argparse import ArgumentParser, Namespace

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


class NCF(RecModel):
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
        self.mlp_uid_embeddings = torch.nn.Embedding(self.user_num, self.vec_size)
        self.mlp_iid_embeddings = torch.nn.Embedding(self.item_num, self.vec_size)

        pre_size = 2 * self.vec_size
        self.mlp_layers = torch.nn.ModuleList()
        for size in self.layers:
            self.mlp_layers.append(torch.nn.Linear(pre_size, size))
            self.mlp_layers.append(torch.nn.ReLU())
            self.mlp_layers.append(torch.nn.Dropout(self.dropout))
            pre_size = size
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(pre_size, 1),
        )
        self.apply(self.init_weights)
        return

    def forward(self, batch, *args, **kwargs):
        u_ids = batch[UID]  # B * 1
        i_ids = batch[IID]  # B * S
        mlp_u_vectors = self.mlp_uid_embeddings(u_ids)  # B * 1 * v
        mlp_i_vectors = self.mlp_iid_embeddings(i_ids)  # B * S * v
        mlp_vectors = torch.cat((mlp_u_vectors.repeat(1, i_ids.size(1), 1), mlp_i_vectors), dim=-1)  # B * S * 2v
        for layer in self.mlp_layers:
            mlp_vectors = layer(mlp_vectors)  # B * S * x
        prediction = self.output_layer(mlp_vectors).squeeze()  # B * S
        batch[PREDICTION] = prediction
        return batch
