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


class NFM(FM):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--layers', type=str, default='[128, 64]',
                            help='Hidden layers in the deep part.')
        return FM.add_model_specific_args(parser)

    def __init__(self, layers='[64]', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = eval(layers) if type(layers) is str else layers

        self.l2 = 1e-05
        self.dropout = 0.3
        self.layers = [128, 64]
        self.eval_batch_size = 16
        # 以上为最佳参数， 不需要可以注释掉
        # NFM,"l2=1e-05,dropout=0.3,layers='[128, 64]',eval_batch_size=16",loss= 5445.5254,ndcg@10= 0.0813,
        # ndcg@5= 0.0562
        # ndcg@10= 0.0669
        # ndcg@20= 0.0846
        # ndcg@50= 0.1097
        # ndcg@100= 0.1357
        # hit@10= 0.1230
        # recall@10= 0.1230
        # recall@20= 0.1941
        # precision@10= 0.0123

    def init_modules(self, *args, **kwargs) -> None:
        self.overall_bias = torch.nn.Parameter(torch.tensor([0.1]))
        pre_size = self.vec_size
        self.deep_layers = torch.nn.ModuleList()
        for size in self.layers:
            self.deep_layers.append(torch.nn.Linear(pre_size, size))
            self.deep_layers.append(torch.nn.ReLU())
            self.deep_layers.append(torch.nn.Dropout(self.dropout))
            pre_size = size
        self.deep_layers.append(torch.nn.Linear(pre_size, 1))
        super().init_modules(*args, **kwargs)
        return

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        labels = batch[LABEL]
        vectors, bias = self.get_embeddings(batch)

        bi_inter_layer = 0.5 * (vectors.sum(dim=-2).pow(2) - vectors.pow(2).sum(dim=-2))  # B * S * v
        prediction = bias.sum(dim=-1)
        deep_vectors = bi_inter_layer.flatten(start_dim=0, end_dim=1)  # (B*S) * v
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # (B*S) * 1
        prediction += deep_vectors.squeeze(dim=-1).view_as(i_ids)  # B * S

        out_dict = {PREDICTION: prediction, LABEL: labels}
        return out_dict