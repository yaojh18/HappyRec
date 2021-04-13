# coding=utf-8
import torch

from ..configs.constants import *
from ..configs.settings import *
from ..models.RecModel import RecModel

'''
@article{DBLP:journals/computer/KorenBV09,
  author    = {Yehuda Koren and
               Robert M. Bell and
               Chris Volinsky},
  title     = {Matrix Factorization Techniques for Recommender Systems},
  journal   = {Computer},
  volume    = {42},
  number    = {8},
  pages     = {30--37},
  year      = {2009}
}
'''


class BiasedMF(RecModel):
    def init_modules(self, *args, **kwargs) -> None:
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.vec_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.vec_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.init_weights()
        return

    def forward(self, batch, *args, **kwargs):
        u_ids = batch[UID]  # B * 1
        i_ids = batch[IID]  # B * S
        cf_u_vectors = self.uid_embeddings(u_ids)  # B * 1 * v
        cf_i_vectors = self.iid_embeddings(i_ids)  # B * S * v
        u_bias = self.user_bias(u_ids).squeeze(dim=-1)  # B * 1
        i_bias = self.item_bias(i_ids).squeeze(dim=-1)  # B * S
        bias = u_bias + i_bias + self.global_bias  # B * S
        prediction = bias + (cf_u_vectors * cf_i_vectors).sum(dim=-1)  # B * S
        batch[PREDICTION] = prediction
        return batch
