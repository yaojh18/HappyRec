# coding=utf-8
import torch
from argparse import ArgumentParser

from ..models.WideDeep import *
from ..configs.constants import *

'''
@inproceedings{DBLP:conf/sigir/0001C17,
  author    = {Xiangnan He and
               Tat{-}Seng Chua},
  title     = {Neural Factorization Machines for Sparse Predictive Analytics},
  booktitle = {{SIGIR}},
  pages     = {355--364},
  publisher = {{ACM}},
  year      = {2017}
}
'''


class NFM(WideDeep):
    def init_modules(self, *args, **kwargs) -> None:
        self.multihot_embeddings = torch.nn.Embedding(self.multihot_f_dim, self.vec_size)
        self.multihot_bias = torch.nn.Embedding(self.multihot_f_dim, 1)
        self.numeric_embeddings = torch.nn.Embedding(self.numeric_f_num, self.vec_size)
        self.numeric_bias = torch.nn.Parameter(torch.tensor([0.01] * self.numeric_f_num), requires_grad=True)
        self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        pre_size = self.vec_size
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

    def forward(self, batch, *args, **kwargs):
        mh_features, nu_features = self.concat_features(batch=batch)  # B * S * mh, B * S * nu
        vectors = self.multihot_embeddings(mh_features)  # B * S * mh * v
        prediction = self.overall_bias + self.multihot_bias(mh_features).squeeze(dim=-1).sum(dim=-1)  # B * S
        if nu_features is not None:
            nu_vectors = self.numeric_embeddings.weight[None, None, :] * nu_features.unsqueeze(dim=-1)  # B * S * nm * v
            nu_bias = self.numeric_bias * nu_features  # B * S * nu
            vectors = torch.cat([vectors, nu_vectors], dim=-2)  # B * S * (mh+nu) * v
            prediction = prediction + nu_bias.sum(dim=-1)  # B * S
        deep_vectors = 0.5 * (vectors.sum(dim=-2).pow(2) - vectors.pow(2).sum(dim=-2))  # B * S * v
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # B * S * 1
        prediction = prediction + deep_vectors.squeeze(dim=-1)  # B * S
        batch[PREDICTION] = prediction
        return batch
