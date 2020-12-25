# coding=utf-8

import torch
import pytorch_lightning as pl

from ..configs.constants import *
from ..metrics.regression import RMSE
from ..metrics.metrics import *


class MetricsList(torch.nn.Module):
    support_metrics = {
        'mse': pl.metrics.MeanSquaredError,
        'rmse': RMSE
    }

    def __init__(self, metrics, **kwargs):
        super().__init__()
        self.metrics_kwargs = kwargs
        self.metrics_str = self.parse_metrics_str(metrics)
        self.metrics = torch.nn.ModuleDict()
        self.init_metrics()

    def parse_metrics_str(self, metrics_str: str):
        if type(metrics_str) is str:
            metrics_str = metrics_str.lower().strip().split(METRIC_SPLITTER)
        metrics = []
        for metric in metrics_str:
            metric = metric.strip()
            if metric == '' or metric not in self.support_metrics:
                continue
            metrics.append(metric)
        return metrics

    def init_metrics(self):
        for metric in self.metrics_str:
            metric = metric.strip()
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
        for metric in self.metrics:
            result[metric] = self.metrics[metric].compute()
        return result
