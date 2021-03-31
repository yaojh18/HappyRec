from ..models.WideDeep import *
from ..configs.constants import *
import torch
from argparse import ArgumentParser

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
        self.numeric_bias = torch.nn.Embedding(self.numeric_f_num, 1)
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
        i_ids = batch[IID]  # B * S
        sample_n = i_ids.size(-1)  # =S

        vectors, numeric_features = [], []
        prediction = self.overall_bias  # 1
        if USER_MH in batch:
            user_mh_vectors = self.multihot_embeddings(batch[USER_MH])  # B * 1 * uf * v
            vectors.append(user_mh_vectors.expand(-1, sample_n, -1, -1))  # B * S * uf * v
            user_mh_bias = self.multihot_bias(batch[USER_MH]).squeeze(dim=-1)  # B * 1 * uf
            prediction = prediction + user_mh_bias.sum(dim=-1)  # B * 1
        if USER_NU in batch:
            numeric_features.append(batch[USER_NU].expand(-1, sample_n, -1))  # B * S * unm
        if ITEM_MH in batch:
            item_mh_vectors = self.multihot_embeddings(batch[ITEM_MH])  # B * S * if * v
            vectors.append(item_mh_vectors)  # B * S * if * v
            item_mh_bias = self.multihot_bias(batch[ITEM_MH]).squeeze(dim=-1)  # B * S * if
            prediction = prediction + item_mh_bias.sum(dim=-1)  # B * S
        if ITEM_NU in batch:
            numeric_features.append(batch[ITEM_NU])  # B * S * inm
        if CTXT_MH in batch:
            ctxt_mh_vectors = self.multihot_embeddings(batch[CTXT_MH])  # B * cf * v
            vectors.append(ctxt_mh_vectors.unsqueeze(dim=1).expand(-1, sample_n, -1, -1))  # B * S * cf * v
            ctxt_mh_bias = self.multihot_bias(batch[CTXT_MH]).squeeze(dim=-1)  # B * cf
            prediction = prediction + ctxt_mh_bias.sum(dim=-1, keepdim=True)  # B * S
        if CTXT_NU in batch:
            numeric_features.append(batch[CTXT_NU].unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * S * cnm
        if self.numeric_f_num > 0:
            numeric_features = torch.cat(numeric_features, dim=-1)  # B * S * nm
            numeric_bias = self.numeric_bias.weight[None, None, :] * numeric_features  # B * S * nm
            prediction = prediction + numeric_bias.sum(dim=-1)  # B * S
            numeric_vectors = self.numeric_embeddings.weight[None, None, :] * \
                              numeric_features.unsqueeze(dim=-1)  # B * S * nm * v
            vectors.append(numeric_vectors)  # B * S * nm * v
        vectors = torch.cat(vectors, dim=-2)  # B * S * f * v
        deep_vectors = 0.5 * (vectors.sum(dim=-2).pow(2) - vectors.pow(2).sum(dim=-2))  # B * S * v
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # B * S * 1
        prediction = prediction + deep_vectors.squeeze(dim=-1)  # B * S
        batch[PREDICTION] = prediction
        return batch
