# coding=utf-8
import torch
import logging
from argparse import ArgumentParser
import numpy as np

from ..configs.constants import *
from ..configs.settings import *
from ..models.Model import Model
from ..utilities.formatter import split_seq
from ..modules.loss import BPRRankLoss, SoftmaxRankLoss
from ..metrics.MetricList import MetricsList
from ..metrics.RankMetricList import RankMetricsList

'''
@inproceedings{DBLP:conf/uai/RendleFGS09,
  author    = {Steffen Rendle and
               Christoph Freudenthaler and
               Zeno Gantner and
               Lars Schmidt{-}Thieme},
  title     = {{BPR:} Bayesian Personalized Ranking from Implicit Feedback},
  booktitle = {{UAI}},
  pages     = {452--461},
  publisher = {{AUAI} Press},
  year      = {2009}
}
'''


class RecModel(Model):
    default_reader = 'RecReader'
    default_dataset = 'RecDataset'

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_sample_n', type=int, default=1,
                            help='Number of iids sampling during training.')
        parser.add_argument('--val_sample_n', type=int, default=-1,
                            help='Number of eval iids during validation, all<=-1, rating=0, ranking>0.')
        parser.add_argument('--test_sample_n', type=int, default=-1,
                            help='Number of eval iids during testing, all<=-1, rating=0, ranking>0.')
        parser.add_argument('--vec_size', type=int, default=64,
                            help='Vector size of user/item embeddings.')
        return Model.add_model_specific_args(parser)

    def __init__(self, train_sample_n: int = 1, val_sample_n: int = -1, test_sample_n: int = -1,
                 user_num: int = None, item_num: int = None, vec_size: int = 64,
                 *args, **kwargs):
        super(RecModel, self).__init__(*args, **kwargs)
        self.train_sample_n = train_sample_n
        self.val_sample_n = val_sample_n
        self.test_sample_n = test_sample_n
        self.user_num = user_num
        self.item_num = item_num
        self.vec_size = vec_size

    def read_data(self, dataset_dir: str = None, reader=None, formatters: dict = None):
        if formatters is None:
            formatters = self.read_formatters()
        reader = Model.read_data(self, dataset_dir=dataset_dir, reader=reader, formatters=formatters)
        reader.read_user(filename=USER_FILE, formatters=formatters)
        reader.read_item(filename=ITEM_FILE, formatters=formatters)
        if self.train_sample_n > 0 or self.val_sample_n != 0 or self.test_sample_n != 0:
            reader.prepare_user_pos_his(label_filter=lambda x: x > 0)
        if self.val_sample_n != 0:
            reader.read_val_iids(filename=VAL_IIDS_FILE, formatters=formatters, sample_n=self.val_sample_n)
        if self.test_sample_n != 0:
            reader.read_test_iids(filename=TEST_IIDS_FILE, formatters=formatters, sample_n=self.test_sample_n)
        if self.user_num is None:
            self.user_num = reader.user_num
        if self.item_num is None:
            self.item_num = reader.item_num
        return reader

    def read_formatters(self, formatters=None) -> dict:
        current = {
            '^' + LABEL + '$': None,
            '^' + UID + '$': None,
            '^' + IID + '$': None,
            '^' + TIME + '$': None,
            '^' + EVAL_IIDS + '$': lambda x: split_seq(x),
            '^' + EVAL_LABELS + '$': lambda x: split_seq(x)
        }
        if formatters is None:
            return current
        return {**current, **formatters}

    def on_train_epoch_start(self) -> None:
        if self.train_sample_n > 0:
            self.train_dataset.sample_train_iids(sample_n=self.train_sample_n)

    def dataset_get_item(self, dataset, index: int) -> dict:
        if dataset.buffer_ds > 0: return dataset.index_buffer[index]
        index_dict = dataset.get_interaction({}, index=index)
        if dataset.phase == TRAIN_PHASE and self.train_sample_n > 0:
            index_dict = dataset.extend_train_iids(index_dict, index=index)
        elif dataset.phase == VAL_PHASE and self.val_sample_n > 0:
            index_dict = dataset.extend_eval_iids(index_dict, index=index, sample_n=self.val_sample_n)
        elif dataset.phase == TEST_PHASE and self.test_sample_n > 0:
            index_dict = dataset.extend_eval_iids(index_dict, index=index, sample_n=self.test_sample_n)
        elif (dataset.phase == VAL_PHASE and self.val_sample_n < 0) or \
                (dataset.phase == TEST_PHASE and self.test_sample_n < 0):
            index_dict = dataset.eval_all_iids(index_dict, index=index)
        return index_dict

    def init_metrics(self, train_metrics=None, val_metrics=None, test_metrics=None, *args, **kwargs):
        if train_metrics is not None:
            if not isinstance(train_metrics, MetricsList):
                train_metrics = MetricsList(train_metrics) if self.train_sample_n <= 0 \
                    else RankMetricsList(train_metrics)
            self.train_metrics = train_metrics
        if val_metrics is not None:
            if not isinstance(val_metrics, MetricsList):
                val_metrics = MetricsList(val_metrics) if self.val_sample_n == 0 \
                    else RankMetricsList(val_metrics)
            self.val_metrics = val_metrics
        if test_metrics is not None:
            if not isinstance(test_metrics, MetricsList):
                test_metrics = MetricsList(test_metrics) if self.test_sample_n == 0 \
                    else RankMetricsList(test_metrics)
            self.test_metrics = test_metrics

    def init_modules(self, *args, **kwargs) -> None:
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.vec_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.vec_size)
        self.init_weights()
        return

    def forward(self, batch, *args, **kwargs):
        u_ids = batch[UID]  # B * 1
        i_ids = batch[IID]  # B * S
        cf_u_vectors = self.uid_embeddings(u_ids)  # B * 1 * v
        cf_i_vectors = self.iid_embeddings(i_ids)  # B * S * v
        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)  # B * S
        batch[PREDICTION] = prediction
        return batch

    def loss_func(self, out_dict, *args, **kwargs):
        prediction, label = out_dict[PREDICTION], out_dict[LABEL]
        if self.train_sample_n > 0:
            loss = BPRRankLoss()(prediction, label, neg_thresh=0, loss_sum=self.loss_sum)
            # loss = SoftmaxRankLoss()(prediction, label, neg_thresh=0, loss_sum=self.loss_sum)
        else:
            loss = torch.nn.MSELoss(reduction='sum' if self.loss_sum == 1 else 'mean')(prediction, label.float())
        return loss

    def training_step(self, batch, batch_idx, *args, **kwargs):
        out_dict = self.forward(batch)
        loss = self.loss_func(out_dict)
        if self.train_sample_n > 0:
            self.log('train_bprrank_loss', loss, on_step=True)
        else:
            self.log('train_mse_loss', loss, on_step=True)
        if self.train_metrics is not None:
            self.train_metrics.update(out_dict)
        out_dict[LOSS] = loss
        return out_dict
