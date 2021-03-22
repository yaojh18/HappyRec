# coding=utf-8

import torch
import pytorch_lightning as pl


class RankMetric(pl.metrics.Metric):
    def __init__(self, topks: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(topks) is not list:
            topks = [topks]
        self.topks = topks
        for topk in topks:
            self.add_state(name='at{}'.format(topk), default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state(name='total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    @staticmethod
    def sort_rank(prediction, label):
        assert prediction.shape == label.shape
        # prediction, indices = prediction.sort(dim=-1, descending=True)  # ? * n
        prediction, indices = prediction.sort(dim=-1)  # ? * n
        prediction = prediction.flip(1)  # ? * n
        indices = indices.flip(1)  # ? * n
        label = label.gather(dim=-1, index=indices)  # ? * n
        return prediction, label

    def update(self, prediction, label, ranked: bool = False) -> None:
        if not ranked:
            prediction, label = self.sort_rank(prediction, label)  # ? * n
        metrics = self.metric_at_k(prediction, label)
        for topk in self.topks:
            metric = metrics[topk]
            exec("self.at{} += metric.sum()".format(topk))
        for topk in self.topks:
            self.total += metrics[topk].numel()
            break

    def compute(self):
        result = {}
        for topk in self.topks:
            result[topk] = eval("self.at{}".format(topk)) / self.total
        return result

    def metric_at_k(self, prediction, label) -> dict:
        pass


class HitRatio(RankMetric):
    def metric_at_k(self, prediction, label) -> dict:
        label = label.gt(0).float()  # ? * n
        result = {}
        for topk in self.topks:
            l = label[..., :topk] if topk > 0 else label  # ? * K
            hit = l.sum(dim=-1).gt(0).float()  # ?
            result[topk] = hit
        return result


class Recall(RankMetric):
    def metric_at_k(self, prediction, label) -> dict:
        label = label.gt(0).float()  # ? * n
        total = label.sum(dim=-1)  # ?
        result = {}
        for topk in self.topks:
            l = label[..., :topk] if topk > 0 else label  # ? * K
            l = l.sum(dim=-1)  # ?
            recall = l / total  # ?
            result[topk] = recall
        return result


class Precision(RankMetric):
    def metric_at_k(self, prediction, label) -> dict:
        label = label.gt(0).float()  # ? * n
        result = {}
        for topk in self.topks:
            l = label[..., :topk] if topk > 0 else label  # ? * K
            precision = l.mean(dim=-1)  # ?
            result[topk] = precision
        return result


class NDCG(RankMetric):
    def metric_at_k(self, prediction, label) -> dict:
        ideal_rank, _ = label.sort(dim=-1, descending=True)  # ? * n
        discounts = torch.log2(torch.arange(max(self.topks)).to(device=prediction.device) + 2.0)  # K
        result = {}
        for topk in self.topks:
            discount = discounts[:topk]
            l = label[..., :topk] if topk > 0 else label  # ? * K
            dcg = (l / discount).sum(dim=-1)  # ?
            ideal = ideal_rank[..., :topk] if topk > 0 else ideal_rank  # ? * K
            idcg = (ideal / discount).sum(dim=-1)  # ?
            ndcg = dcg / idcg  # ?
            result[topk] = ndcg
        return result
