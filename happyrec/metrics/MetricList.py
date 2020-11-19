# coding=utf-8

import torch
import pytorch_lightning as pl

from ..configs.constants import *
from ..metrics.regression import RMSE


class MetricsList(torch.nn.Module):
    support_metrics = {
        'mse': pl.metrics.MeanSquaredError,
        'rmse': RMSE
    }

    def __init__(self, metrics, **kwargs):
        super().__init__()
        self.metrics_kwargs = kwargs
        if type(metrics) is str:
            metrics = self.parse_metrics_str(metrics)
        self.metrics_str = metrics
        self.metrics = torch.nn.ModuleDict()
        self.init_metrics()

    def parse_metrics_str(self, metrics_str: str):
        return metrics_str.strip().lower().split(';')

    def init_metrics(self):
        for metric in self.metrics_str:
            metric = metric.strip()
            if metric in self.support_metrics:
                if metric in self.metrics:
                    continue
                self.metrics[metric] = self.support_metrics[metric](**self.metrics_kwargs)

    def forward(self, *args, **kwargs):
        self.update(*args, **kwargs)
        if self.compute_on_step:
            return self.compute()

    def update(self, output: dict) -> None:
        prediction, label = output[PREDICTION], output[LABEL]
        for key in self.metrics:
            metric = self.metrics[key]
            metric.update(prediction, label)

    def compute(self):
        result = {}
        for metric in self.metrics_str:
            if metric not in result and metric in self.metrics:
                result[metric] = self.metrics[metric].compute()
        return result
