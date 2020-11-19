# coding=utf-8

import torch
import pytorch_lightning as pl


class HitRatio(pl.metrics.Metric):
    def __init__(self, topk: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.add_state(name='atk', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state(name='total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, prediction, label, ranked: bool = True) -> None:
        assert prediction.shape == label.shape
        if not ranked:
            prediction, indices = prediction.sort(dim=-1, descending=True)  # ? * K
            label = label.gather(dim=-1, index=indices)  # ? * K
        l = label[..., :self.topk] if self.topk > 0 else label
        l = l.sum(dim=-1).gt(0).float()  # ?
        self.atk += l.sum()
        self.total += l.numel()
        return

    def compute(self):
        return self.atk / self.total
