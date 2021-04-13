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


class NFM(DeepModel):

    def __init__(self, *args, **kwargs) -> None:
        # NFM,"l2=1e-05,dropout=0.3,layers='[128, 64]',eval_batch_size=16",loss= 5445.5254,ndcg@10= 0.0813,ndcg@5= 0.0562  ndcg@10= 0.0669  ndcg@20= 0.0846  ndcg@50= 0.1097  ndcg@100= 0.1357  hit@10= 0.1230  recall@10= 0.1230  recall@20= 0.1941  precision@10= 0.0123
        super().__init__(*args, **kwargs)
        self.l2 = 1e-05
        self.dropout = 0.3
        self.layers = '[128,64]'
        self.eval_batch_size = 16

    def init_modules(self, *args, **kwargs) -> None:
        self.multihot_embeddings = torch.nn.Embedding(self.multihot_f_dim, self.vec_size)
        self.numeric_embeddings = torch.nn.Parameter(torch.Tensor(self.numeric_f_num, self.vec_size),
                                                     requires_grad=True)
        self.overall_bias = torch.nn.Parameter(torch.tensor([0.1]))
        # init_weight是否对Parameter进行了初始化
        self.first_layer_size = self.vec_size
        self.init_weights(self.numeric_embeddings)
        super().init_modules(*args, **kwargs)
        return

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        labels = batch[LABEL]
        sample_n = i_ids.size(-1)  # S

        multihot_vectors = []
        numeric_vectors = []
        vectors = []
        prediction = self.overall_bias
        if USER_MH in batch:
            user_mh_vectors = self.multihot_embeddings(batch[USER_MH].long())  # B * 1 * uf * v
            user_mh_bias = self.multihot_bias(batch[USER_MH].long()).squeeze(dim=-1)  # B * 1 * uf
            multihot_vectors.append(user_mh_vectors.expand(-1, sample_n, -1, -1))
            prediction = prediction + user_mh_bias.expand(-1, sample_n, -1).sum(dim=-1)
        if USER_NU in batch:
            numeric_vectors.append(batch[USER_NU].expand(-1, sample_n, -1))  # B * S * unm
        if ITEM_MH in batch:
            item_mh_vectors = self.multihot_embeddings(batch[ITEM_MH].long())  # B * S * if * v
            item_mh_bias = self.multihot_bias(batch[ITEM_MH].long()).squeeze(dim=-1)  # B * S * i
            multihot_vectors.append(item_mh_vectors)
            prediction += item_mh_bias.sum(dim=-1)  # B * S
        if ITEM_NU in batch:
            numeric_vectors.append(batch[ITEM_NU])  # B * S * inm
        if CTXT_MH in batch:
            ctxt_mh_vectors = self.multihot_embeddings(batch[CTXT_MH].long())  # B * cf * v
            ctxt_mh_bias = self.multihot_bias(batch[CTXT_MH].long()).squeeze(dim=-1)  # B * cf
            multihot_vectors.append(ctxt_mh_vectors.unsqueeze(dim=1).expand(-1, sample_n, -1, -1))  # B * S * cf * v
            prediction += ctxt_mh_bias.sum(dim=-1, keepdim=True)  # B * S
        if CTXT_NU in batch:
            numeric_vectors.append(batch[CTXT_NU].unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * S * cnm
        if self.multihot_f_num > 0:
            multihot_vectors = torch.cat(multihot_vectors, dim=-2)  # B * S * mn * v
            vectors.append(multihot_vectors)
        if self.numeric_f_num > 0:
            numeric_vectors = torch.cat(numeric_vectors, dim=-1)  # B * S * nm
            numeric_bias = numeric_vectors * self.numeric_bias  # B * S * nm
            prediction += numeric_bias.sum(dim=-1)  # B * S
            numeric_vectors = numeric_vectors.unsqueeze(dim=-1) * self.numeric_embeddings  # B * S * nm * v
            vectors.append(numeric_vectors)
        vectors = torch.cat(vectors, dim=-2)
        bi_inter_layer = 0.5 * (vectors.sum(dim=-2).pow(2) - vectors.pow(2).sum(dim=-2))  # B * S * v
        deep_vectors = bi_inter_layer.flatten(start_dim=0, end_dim=1)  # (B*S) * v
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # (B*S) * 1
        prediction += deep_vectors.squeeze(dim=-1).view_as(i_ids)  # B * S
        out_dict = {PREDICTION: prediction, LABEL: labels}
        return out_dict