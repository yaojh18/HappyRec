# coding=utf-8
import torch
import pytorch_lightning as pl


class RMSE(pl.metrics.MeanSquaredError):
    def __init__(self, *args, **kwargs):
        super(RMSE, self).__init__(*args, **kwargs)

    def compute(self):
        return super().compute().sqrt()
