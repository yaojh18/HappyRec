from .FM import FM
from ..configs.constants import *
import torch
from argparse import ArgumentParser

class AFM(FM):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--attention_vec_size', type=int, default=64,
                            help='Attention Layer dimension.')
        return FM.add_model_specific_args(parser)


    def __init__(self, attention_vec_size=32, *args, **kwargs):
        # "l2=0.0001,dropout=0.3,attention_vec_size=32,eval_batch_size=1", loss = 12302.6895, ndcg @ 10 = 0.0663,
        # ndcg @ 5 = 0.0458
        # ndcg @ 10 = 0.0606
        # ndcg @ 20 = 0.0794
        # ndcg @ 50 = 0.1091
        # ndcg @ 100 = 0.1351
        # hit @ 10 = 0.1145
        # recall @ 10 = 0.1145
        # recall @ 20 = 0.1898
        # precision @ 10 = 0.0115, 130, 1:10: 11, 2021 - 04 - 14
        # 效果为什么这么差？？？
        super().__init__(*args, **kwargs)
        self.l2 = 0.0001
        self.dropout = 0.3
        self.eval_batch_size = 1
        self.attention_vec_size = attention_vec_size

    def init_modules(self, *args, **kwargs) -> None:
        self.p = torch.nn.Linear(self.vec_size, 1)
        self.w = torch.nn.Linear(self.vec_size, self.attention_vec_size)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(self.dropout)
        self.h = torch.nn.Linear(self.attention_vec_size, 1, bias=False)
        super().init_modules(*args, **kwargs)


    def forward(self, batch, *args, **kwargs):
        labels = batch[LABEL] # b s
        vectors, bias = self.get_embeddings(batch) #bsnv bsn1

        B, S, N, _ = vectors.shape
        bi_layer = vectors.unsqueeze(dim=-2) * vectors.unsqueeze(dim=-3)  # B * S * n * n * v
        a = self.w(bi_layer)  # B * S * n * n * v
        a = self.relu(a)  # B * S * n * n * v
        a = self.drop(a)  # B * S * n * n * v
        a = self.h(a)  # B * S * n * n * 1
        a = a.view((B, S, N * N)).softmax(dim=-1).view((B, S, N, N, 1))  # B * S * n * n * 1
        a = (a * bi_layer).sum(dim=-2).sum(dim=-2)  # B * S * v
        a = self.p(a.flatten(start_dim=0, end_dim=1)).view((B, S))  # B * S
        prediction = bias.sum(dim=-1) + a
        out_dict = {PREDICTION: prediction, LABEL: labels}
        return out_dict
