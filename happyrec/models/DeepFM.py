from .FM import FM
from ..configs.constants import *
import torch
from argparse import ArgumentParser

USER_MH = 'user_mh'
USER_NU = 'user_nm'
ITEM_MH = 'item_mh'
ITEM_NU = 'item_nm'
CTXT_MH = 'ctxt_mh'
CTXT_NU = 'ctxt_nm'


class DeepFM(FM):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--layers', type=str, default='[64]',
                            help='Hidden layers in the deep part.')
        return FM.add_model_specific_args(parser)

    def __init__(self, layers='[64]', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = eval(layers) if type(layers) is str else layers

        self.l2 = 0.0001
        self.eval_batch_size = 16
        self.layers = [64]
        # 以上为最佳参数， 不需要可以注释掉
        # DeepFM,"l2=0.0001,layers='[64]',eval_batch_size=16",loss= 7735.3701,ndcg@10= 0.0799,
        # ndcg@5= 0.0561
        # ndcg@10= 0.0683
        # ndcg@20= 0.0858
        # ndcg@50= 0.1175
        # ndcg@100= 0.1440
        # hit@10= 0.1262
        # recall@10= 0.1262
        # recall@20= 0.1962
        # precision@10= 0.0126

    def init_modules(self, *args, **kwargs) -> None:
        pre_size = self.multihot_f_num * self.vec_size + self.numeric_f_num
        self.deep_layers = torch.nn.ModuleList()
        for size in self.layers:
            self.deep_layers.append(torch.nn.Linear(pre_size, size))
            self.deep_layers.append(torch.nn.ReLU())
            self.deep_layers.append(torch.nn.Dropout(self.dropout))
            pre_size = size
        self.deep_layers.append(torch.nn.Linear(pre_size, 1))
        super().init_modules(*args, **kwargs)

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        labels = batch[LABEL]
        vectors, bias = self.get_embeddings(batch)

        wide_prediction = bias.sum(dim=-1) + 0.5 * (vectors.sum(dim=-2).pow(2) - vectors.pow(2).sum(dim=-2)).sum(dim=-1)  # B * S
        deep_vectors = vectors.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2)  # (B * S) * (f * v)
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # (B*S) * 1
        deep_prediction = deep_vectors.squeeze(dim=-1).view_as(i_ids)  # B * S
        prediction = wide_prediction + deep_prediction  # B * S

        out_dict = {PREDICTION: prediction, LABEL: labels}
        return out_dict