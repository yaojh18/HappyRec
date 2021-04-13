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


class PNN(FM):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--first_layer_size', type=int, default=64,
                            help='First Layer dimension in the deep part.')
        parser.add_argument('--layers', type=str, default='[64]',
                            help='Hidden layers in the deep part.')
        return FM.add_model_specific_args(parser)

    def __init__(self, first_layer_size : int = 64, layers='[64]', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_layer_size = first_layer_size
        self.layers = eval(layers) if type(layers) is str else layers

    def init_modules(self, *args, **kwargs) -> None:
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
        self.init_weights(self.z_weight)
        super().init_modules(*args, **kwargs)
        return

    def calutate_lp(self, vectors):
        pass

    def forward(self, batch, *args, **kwargs):
        i_ids = batch[IID]  # B * S
        labels = batch[LABEL]
        vectors, _ = self.get_embeddings(batch)

        lz = (vectors.unsqueeze(dim=-3) * self.z_weight).sum(dim=-1).sum(dim=-1)  # B * S * fls
        lp = self.calutate_lp(vectors)  # B * S * fls
        deep_vectors = torch.cat([lz, lp], dim=-1).flatten(start_dim=0, end_dim=1)  # (B*S) * fls
        for layer in self.deep_layers:
            deep_vectors = layer(deep_vectors)  # (B*S) * 1
        prediction = deep_vectors.squeeze(dim=-1).view_as(i_ids)  # B * S

        out_dict = {PREDICTION: prediction, LABEL: labels}
        return out_dict


class IPNN(PNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval_batch_size = 4
        self.layers = [128, 64]
        self.l2 = 1e-05
        # 以上为最佳参数， 不需要可以注释掉
        # IPNN, "l2=1e-05,layers='[128, 64]',eval_batch_size=4", loss = 5779.3350, ndcg @ 10 = 0.0711,
        # ndcg @ 5 = 0.0497
        # ndcg @ 10 = 0.0649
        # ndcg @ 20 = 0.0881
        # ndcg @ 50 = 0.1165
        # ndcg @ 100 = 0.1419
        # hit @ 10 = 0.1262
        # recall @ 10 = 0.1262
        # recall @ 20 = 0.2185
        # precision @ 10 = 0.0126

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval_batch_size = 1
        self.layers = [64]
        self.l2 = 0
        # 以上为最佳参数， 不需要可以注释掉
        # OPNN,"l2=0,layers='[64]',eval_batch_size=1",loss= 9551.4297,ndcg@10= 0.0807,
        # ndcg@5= 0.0553
        # ndcg@10= 0.0704
        # ndcg@20= 0.0914
        # ndcg@50= 0.1196
        # ndcg@100= 0.1452
        # hit@10= 0.1357
        # recall@10= 0.1357
        # recall@20= 0.2206
        # precision@10= 0.0136

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