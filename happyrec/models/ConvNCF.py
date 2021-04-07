# coding=utf-8
import torch

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel
from argparse import ArgumentParser

'''
@inproceedings{DBLP:conf/ijcai/0001DWTTC18,
  author    = {Xiangnan He and
               Xiaoyu Du and
               Xiang Wang and
               Feng Tian and
               Jinhui Tang and
               Tat{-}Seng Chua},
  title     = {Outer Product-based Neural Collaborative Filtering},
  booktitle = {{IJCAI}},
  pages     = {2227--2233},
  publisher = {ijcai.org},
  year      = {2018}
}
'''


class ConvNCF(RecModel):
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--out_channel', type=int, default=32,
                            help='Output channels after CNN.')
        parser.add_argument('--conv_dim', type=int, default=32,
                            help='Output matrix size after CNN.')
        return RecModel.add_model_specific_args(parser)

    def __init__(self, out_channel: int = 32, conv_dim: int = 32, *args, **kwargs):
        self.out_channel = out_channel
        self.conv_dim = conv_dim
        super().__init__(*args, **kwargs)

    def init_modules(self, *args, **kwargs) -> None:
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.vec_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.vec_size)
        self.cnn_layers = torch.nn.ModuleList()
        self.cnn_layers.append(
            torch.nn.Conv2d(in_channels=1, out_channels=self.out_channel, kernel_size=(2, 2), stride=(2, 2)))
        self.cnn_layers.append(torch.nn.ReLU())
        pre_size = self.vec_size // 2
        while pre_size > self.conv_dim:
            self.cnn_layers.append(
                torch.nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel,
                                kernel_size=(2, 2), stride=(2, 2)))
            self.cnn_layers.append(torch.nn.ReLU())
            pre_size //= 2
        self.weight_vector = torch.nn.Linear(self.out_channel * (self.conv_dim ** 2), 1)
        self.init_weights()
        return

    def forward(self, batch, *args, **kwargs):
        u_ids = batch[UID]  # B * 1
        i_ids = batch[IID]  # B * S
        u_vectors = self.uid_embeddings(u_ids)  # B * 1 * v
        i_vectors = self.iid_embeddings(i_ids)  # B * S * v
        interaction_map = torch.matmul(u_vectors.unsqueeze(3), i_vectors.unsqueeze(2))  # B * S * v * v
        output_vector = interaction_map.unsqueeze(2).flatten(start_dim=0, end_dim=1)  # (B * S) * 1 * v * v
        for layer in self.cnn_layers:
            output_vector = layer(output_vector)  # (B * S) * o * x * x
        output_vector = output_vector.flatten(start_dim=1, end_dim=3)  # (B * S) * (o * e * e)
        prediction = self.weight_vector(output_vector).squeeze().view_as(i_ids)  # B * S
        batch[PREDICTION] = prediction
        return batch
