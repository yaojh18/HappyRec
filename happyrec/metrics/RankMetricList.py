# coding=utf-8

import torch
import pytorch_lightning as pl

from ..configs.constants import *
from ..metrics.regression import *
from ..metrics.ranking import *
from ..metrics.MetricList import MetricsList


class RankMetricsList(MetricsList):
    support_metrics = {
        'mse': pl.metrics.MeanSquaredError,
        'rmse': RMSE,
        'hit': HitRatio
    }

    def __init__(self, *args, **kwargs):
        self.require_rank = False
        super().__init__(*args, **kwargs)

    def parse_metrics_str(self, metrics_str: str):
        metrics_str = metrics_str.strip().split(';')
        metrics = []
        for metric in metrics_str:
            if '@' not in metric:
                metrics.append(metric)
            else:
                metric, topk = metric.split('@')
                topk = [k.strip() for k in topk.split(',')]
                for k in topk:
                    metrics.append(metric + '@' + k)
        return metrics

    def init_metrics(self):
        for metric in self.metrics_str:
            metric = metric.strip()
            if metric in self.metrics:
                continue
            if '@' not in metric and metric in self.support_metrics:
                self.metrics[metric] = self.support_metrics[metric](**self.metrics_kwargs)
            elif '@' in metric:
                rank_m, topk = metric.split('@')
                if rank_m in self.support_metrics:
                    self.metrics[metric] = self.support_metrics[rank_m](topk=int(topk), **self.metrics_kwargs)
                    self.require_rank = True

    def update(self, output: dict, ranked=False) -> None:
        prediction, label = output[PREDICTION], output[LABEL]
        if self.require_rank and not ranked:
            assert prediction.shape == label.shape
            prediction, indices = prediction.sort(dim=-1, descending=True)  # ? * K
            label = label.gather(dim=-1, index=indices)  # ? * K
        for key in self.metrics:
            metric = self.metrics[key]
            if '@' in key:
                metric.update(prediction, label, ranked=True)
            else:
                metric.update(prediction, label)
