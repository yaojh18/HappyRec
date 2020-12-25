# coding=utf-8

import torch
import pytorch_lightning as pl
from collections import defaultdict

from ..configs.constants import *
from ..metrics.regression import *
from ..metrics.ranking import *
from ..metrics.MetricList import MetricsList
from ..metrics.metrics import *


class RankMetricsList(MetricsList):
    support_metrics = {
        'mse': pl.metrics.MeanSquaredError,
        'rmse': RMSE,
        'hit': HitRatio,
        'precision': Precision,
        'recall': Recall,
        'ndcg': NDCG,
    }

    def __init__(self, *args, **kwargs):
        self.require_rank = False
        super().__init__(*args, **kwargs)

    def parse_metrics_str(self, metrics_str: str):
        if type(metrics_str) is str:
            metrics_str = metrics_str.lower().strip().split(METRIC_SPLITTER)
        metrics = []
        for metric in metrics_str:
            metric = metric.strip()
            if metric == '':
                continue
            if '@' not in metric and metric in self.support_metrics:
                metrics.append(metric)
            else:
                metric, topk = metric.split('@')
                if metric not in self.support_metrics:
                    continue
                topk = [k.strip() for k in topk.split(RANK_SPLITTER)]
                for k in topk:
                    metrics.append(metric + '@' + k)
        return metrics

    def init_metrics(self):
        metric_topks = defaultdict(list)
        for metric in self.metrics_str:
            metric = metric.strip()
            if '@' not in metric and metric not in self.metrics:
                self.metrics[metric] = self.support_metrics[metric](**self.metrics_kwargs)
            elif '@' in metric:
                rank_m, topk = metric.split('@')
                topk = int(topk)
                if topk not in metric_topks[rank_m]:
                    metric_topks[rank_m].append(topk)
                    self.require_rank = True
        for metric in metric_topks:
            self.metrics[metric] = self.support_metrics[metric](topks=metric_topks[metric], **self.metrics_kwargs)

    def update(self, output: dict, ranked=False) -> None:
        prediction, label = output[PREDICTION], output[LABEL]
        if self.require_rank and not ranked:
            prediction, label = RankMetric.sort_rank(prediction, label)
        for key in self.metrics:
            metric = self.metrics[key]
            if '@' in key:
                metric.update(prediction, label, ranked=True)
            else:
                metric.update(prediction, label)

    def compute(self):
        result = {}
        for metric in self.metrics:
            metric_result = self.metrics[metric].compute()
            if type(metric_result) is not dict:
                result[metric] = metric_result
            else:
                for topk in metric_result:
                    result["{}@{}".format(metric, topk)] = metric_result[topk]
        return result
