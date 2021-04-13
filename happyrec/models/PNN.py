from .DeepModel import DeepModel
from ..configs.constants import *
import torch
from argparse import ArgumentParser

USER_MH = 'user_mh'
USER_NU = 'user_nm'
ITEM_MH = 'item_mh'
ITEM_NU = 'item_nm'
CTXT_MH = 'ctxt_mh'
CTXT_NU = 'ctxt_nm'


class PNN(DeepModel):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--first_layer_size', type=int, default=64,
                            help='First Layer dimension in the deep part.')
        return DeepModel.add_model_specific_args(parser)

    def __init__(self, first_layer_size : int = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_layer_size = first_layer_size
        self.eval_batch_size = 1

    def init_modules(self, *args, **kwargs) -> None:
        self.multihot_embeddings = torch.nn.Embedding(self.multihot_f_dim, self.vec_size)
        self.numeric_embeddings = torch.nn.Parameter(torch.Tensor(self.numeric_f_num, self.vec_size),
                                                     requires_grad=True)
        self.z_weight = torch.nn.Parameter(torch.Tensor(self.first_layer_size, self.multihot_f_num + self.numeric_f_num,
                                                        self.vec_size),
                                           requires_grad=True)
        self.deep_layers = torch.nn.ModuleList()
        if self.first_layer_size:
            pre_size = self.first_layer_size * 2
            for size in self.layers:
                self.deep_layers.append(torch.nn.Linear(pre_size, size))
                # self.deep_layers.append(torch.nn.BatchNorm1d(size))
                self.deep_layers.append(torch.nn.ReLU())
                self.deep_layers.append(torch.nn.Dropout(self.dropout))
                pre_size = size
            self.deep_layers.append(torch.nn.Linear(pre_size, 1))
        self.apply(self.init_weights)
        self.init_weights(self.numeric_embeddings)
        self.init_weights(self.z_weight)
        return

    def calutate_lp(self, vectors):
        pass

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        labels = batch[LABEL]
        sample_n = i_ids.size(-1)  # S

        multihot_vectors = []
        numeric_vectors = []
        vectors = []
        if USER_MH in batch:
            user_mh_vectors = self.multihot_embeddings(batch[USER_MH].long())  # B * 1 * uf * v
            multihot_vectors.append(user_mh_vectors.expand(-1, sample_n, -1, -1))
        if USER_NU in batch:
            numeric_vectors.append(batch[USER_NU].expand(-1, sample_n, -1))  # B * S * unm
        if ITEM_MH in batch:
            item_mh_vectors = self.multihot_embeddings(batch[ITEM_MH].long())  # B * S * if * v
            multihot_vectors.append(item_mh_vectors)
        if ITEM_NU in batch:
            numeric_vectors.append(batch[ITEM_NU])  # B * S * inm
        if CTXT_MH in batch:
            ctxt_mh_vectors = self.multihot_embeddings(batch[CTXT_MH].long())  # B * cf * v
            multihot_vectors.append(ctxt_mh_vectors.unsqueeze(dim=1).expand(-1, sample_n, -1, -1))  # B * S * cf * v
        if CTXT_NU in batch:
            numeric_vectors.append(batch[CTXT_NU].unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * S * cnm
        if self.multihot_f_num > 0:
            multihot_vectors = torch.cat(multihot_vectors, dim=-2)  # B * S * mn * v
            vectors.append(multihot_vectors)
        if self.numeric_f_num > 0:
            numeric_vectors = torch.cat(numeric_vectors, dim=-1)  # B * S * nm
            numeric_vectors = numeric_vectors.unsqueeze(dim=-1) * self.numeric_embeddings  # B * S * nm * v
            vectors.append(numeric_vectors)
        vectors = torch.cat(vectors, dim=-2)  # B * S * n * v
        lz = (vectors.unsqueeze(dim=-3) * self.z_weight).sum(dim=-1).sum(dim=-1)  # B * S * fls
        lp = self.calutate_lp(vectors)  # B * S * fls
        deep_vectors = torch.cat([lz, lp], dim=-1).flatten(start_dim=0, end_dim=1)  # (B*S) * fls
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # (B*S) * 1
        prediction = deep_vectors.squeeze(dim=-1).view_as(i_ids)  # B * S
        out_dict = {PREDICTION: prediction, LABEL: labels}
        return out_dict


class IPNN(PNN):

    def init_modules(self, *args, **kwargs) -> None:
        super().init_modules(*args, **kwargs)
        self.p_weight = torch.nn.Parameter(torch.Tensor(self.first_layer_size, self.multihot_f_num + self.numeric_f_num), requires_grad=True)
        self.init_weights(self.p_weight)
        return

    def calutate_lp(self, vectors):
        lp = self.p_weight.matmul(vectors).norm(dim=-1, p=2)
        # fls * n matmul B * S * n * v = B * S * fls * v
        return lp

class OPNN(PNN):

    def init_modules(self, *args, **kwargs) -> None:
        super().init_modules(*args, **kwargs)
        self.p_weight = torch.nn.Parameter(torch.Tensor(self.first_layer_size, self.vec_size, self.vec_size),
                                           requires_grad=True)
        self.init_weights(self.p_weight)
        return

    def calutate_lp(self, vectors):
        sum_f = vectors.sum(dim=-2)   # B * S * v
        a = sum_f.unsqueeze(dim=-1)   # B * S * v * 1
        b = sum_f.unsqueeze(dim=-2)   # B * S * 1 * v
        lp = ((a * b).unsqueeze(dim=-3) * self.p_weight).sum(dim=-1).sum(dim=-1)   # B * S * lys
        return lp