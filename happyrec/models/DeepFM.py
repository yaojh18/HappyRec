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


class DeepFM(DeepModel):

    def __init__(self, *args, **kwargs) -> None:
        # DeepFM,"l2=0.0001,layers='[64]',eval_batch_size=16",loss= 7735.3701,ndcg@10= 0.0799,ndcg@5= 0.0561  ndcg@10= 0.0683  ndcg@20= 0.0858  ndcg@50= 0.1175  ndcg@100= 0.1440  hit@10= 0.1262  recall@10= 0.1262  recall@20= 0.1962  precision@10= 0.0126
        super().__init__(*args, **kwargs)
        self.l2 = 0.0001
        self.eval_batch_size =16

    def init_modules(self, *args, **kwargs) -> None:
        self.multihot_embeddings = torch.nn.Embedding(self.multihot_f_dim, self.vec_size)
        self.numeric_embeddings = torch.nn.Parameter(torch.Tensor(self.numeric_f_num, self.vec_size), requires_grad=True)
        self.first_layer_size = self.multihot_f_num * self.vec_size + self.numeric_f_num
        self.init_weights(self.numeric_embeddings)
        super().init_modules(*args, **kwargs)

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        labels = batch[LABEL]
        sample_n = i_ids.size(-1)  # =S

        wide_prediction = 0  # B * S
        numeric_vectors = []
        vectors = []
        if USER_MH in batch:
            user_mh_vectors = self.multihot_embeddings(batch[USER_MH].long())  # B * 1 * uf * v
            vectors.append(user_mh_vectors.expand(-1, sample_n, -1, -1)) # B * S * uf * v
            user_mh_bias = self.multihot_bias(batch[USER_MH].long()).squeeze(dim=-1)  # B * 1 * uf
            wide_prediction += user_mh_bias.sum(dim=-1).expand(-1, sample_n) # B * S
        if USER_NU in batch:
            numeric_vectors.append(batch[USER_NU].expand(-1, sample_n, -1))  # B * S * unm
        if ITEM_MH in batch:
            item_mh_vectors = self.multihot_embeddings(batch[ITEM_MH].long())  # B * S * if * v
            vectors.append(item_mh_vectors)  # B * S * if * v
            item_mh_bias = self.multihot_bias(batch[ITEM_MH].long()).squeeze(dim=-1)  # B * S * i
            wide_prediction += item_mh_bias.sum(dim=-1)  # B * S
        if ITEM_NU in batch:
            numeric_vectors.append(batch[ITEM_NU])  # B * S * inm
        if CTXT_MH in batch:
            ctxt_mh_vectors = self.multihot_embeddings(batch[CTXT_MH].long())  # B * cf * v
            vectors.append(ctxt_mh_vectors.unsqueeze(dim=1).expand(-1, sample_n, -1, -1))  # B * S * cf * v
            ctxt_mh_bias = self.multihot_bias(batch[CTXT_MH].long()).squeeze(dim=-1)  # B * cf
            wide_prediction += ctxt_mh_bias.sum(dim=-1, keepdim=True) # B * S
        if CTXT_NU in batch:
            numeric_vectors.append(batch[CTXT_NU].unsqueeze(dim=1).expand(-1, sample_n, -1))  # B * S * cnm
        if self.numeric_f_num > 0:
            numeric_vectors = torch.cat(numeric_vectors, dim=-1)  # B * S * nm
            numeric_bias = numeric_vectors * self.numeric_bias  # B * S * nm
            numeric_vectors = numeric_vectors.unsqueeze(dim=-1) * self.numeric_embeddings  # B * S * nm * v
            vectors.append(numeric_vectors)
            wide_prediction += numeric_bias.sum(dim=-1)
        vectors = torch.cat(vectors, dim=-2) # B * S * f * v
        wide_prediction += 0.5 * (vectors.sum(dim=-2).pow(2) - vectors.pow(2).sum(dim=-2)).sum(dim=-1) # B * S
        deep_vectors = vectors.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2)   # (B * S) * (f * v)
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # (B*S) * 1
        deep_prediction = deep_vectors.squeeze(dim=-1).view_as(i_ids)  # B * S
        prediction = wide_prediction + deep_prediction  # B * S
        out_dict = {PREDICTION: prediction, LABEL: labels}
        return out_dict

