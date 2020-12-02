# coding=utf-8
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser

from ..data_readers.DataReader import DataReader
from ..data_readers.RecReader import RecReader
from ..datasets.Dataset import Dataset
from ..datasets.RecDataset import RecDataset
from ..configs.constants import *
from ..configs.settings import *
from ..metrics.MetricList import MetricsList
from ..metrics.metrics import METRICS_SMALLER


class Model(pl.LightningModule):
    default_reader = 'DataReader'
    default_dataset = 'Dataset'

    @staticmethod
    def parse_model_args(parent_parser):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2_bias', type=int, default=0,
                            help='Whether add l2 regularizer on bias.')
        parser.add_argument('--l2', type=float, default=1e-6,
                            help='Weight of l2_regularize in pytorch optimizer.')
        parser.add_argument('--loss_sum', type=int, default=1,
                            help='Reduction of batch loss 1=sum, 0=mean')
        parser.add_argument('--buffer_ds', type=int, default=0,
                            help='Whether buffer dataset items or not.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128,
                            help='Batch size during testing.')
        parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of processors when get batches in DataLoader')
        parser.add_argument('--es_patience', type=int, default=20,
                            help='#epochs with no improvement after which training will be stopped (early stop).')
        parser.add_argument('--train_metrics', type=str, default='',
                            help='Calculate metrics on training')
        parser.add_argument('--val_metrics', type=str, default='',
                            help='Calculate metrics on validation')
        parser.add_argument('--test_metrics', type=str, default='rmse',
                            help='Calculate metrics on testing')
        return parser

    def __init__(self,
                 lr: float = 0.001, optimizer: str = 'Adam', dropout: float = 0.2,
                 l2: float = 1e-6, l2_bias: int = 0, loss_sum: int = 1,
                 buffer_ds: int = 0, batch_size: int = 128, eval_batch_size: int = 128, num_workers: int = 5,
                 es_patience: int = 20,
                 *args, **kwargs):
        super(Model, self).__init__()
        self.lr = lr
        self.optimizer_name = optimizer
        self.dropout = dropout
        self.l2_weight = l2
        self.l2_bias = l2_bias
        self.loss_sum = loss_sum
        self.buffer_ds = buffer_ds
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.es_patience = es_patience

        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.reader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def read_data(self, dataset_dir: str = None, reader=None, formatters: dict = None):
        if reader is None:
            reader = eval(self.default_reader)(dataset_dir=dataset_dir)
        self.reader = reader
        if formatters is None:
            formatters = self.read_formatters()
        reader.read_train(filename=TRAIN_FILE, formatters=formatters)
        reader.read_validation(filename=VAL_FILE, formatters=formatters)
        reader.read_test(filename=TEST_FILE, formatters=formatters)
        return reader

    def read_formatters(self, formatters: dict = None) -> dict:
        current = {
            '^' + LABEL + '$': None,
            INT_F + '$': None,
            FLOAT_F + '$': None,
        }
        if formatters is None:
            return current
        return {**current, **formatters}

    def get_dataset(self, phase):
        if self.reader is None:
            LOGGER.error('Model.Reader is None, Please read data first. (model.read_data(...))')
            return
        if phase == TRAIN_PHASE:
            if self.train_dataset is not None:
                return self.train_dataset
            self.train_dataset = eval(self.default_dataset)(
                data=self.reader.train_data, reader=self.reader,
                model=self, buffer_ds=self.buffer_ds, phase=TRAIN_PHASE)
            return self.train_dataset
        if phase == VAL_PHASE:
            if self.val_dataset is not None:
                return self.val_dataset
            self.val_dataset = eval(self.default_dataset)(
                data=self.reader.val_data, reader=self.reader, model=self, buffer_ds=self.buffer_ds, phase=VAL_PHASE)
            return self.val_dataset
        if phase == TEST_PHASE:
            if self.test_dataset is not None:
                return self.test_dataset
            self.test_dataset = eval(self.default_dataset)(
                data=self.reader.test_data, reader=self.reader, model=self, buffer_ds=self.buffer_ds, phase=TEST_PHASE)
            return self.test_dataset
        logging.error("ERROR: unknown phase {}".format(phase))
        return

    def dataset_length(self, dataset):
        if type(dataset.data) is dict:
            for key in dataset.data:
                return len(dataset.data[key])
        return len(dataset.data)

    def dataset_get_item(self, dataset, index: int) -> dict:
        if dataset.buffer_ds > 0: return dataset.index_buffer[index]
        index_dict = {}
        for c in dataset.data:
            index_dict[c] = dataset.data[c][index]
        return index_dict

    def dataset_collate_batch(self, dataset, batch: list) -> dict:
        result_dict = {}
        for c in batch[0]:
            result_dict[c] = dataset.collate_stack([b[c] for b in batch])
        return result_dict

    def dataset_get_dataloader(self, dataset: Dataset) -> torch.utils.data.DataLoader:
        if dataset.phase == TRAIN_PHASE:
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                collate_fn=dataset.collate_batch, pin_memory=True)
        if dataset.phase == VAL_PHASE or dataset.phase == TEST_PHASE:
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                collate_fn=dataset.collate_batch, pin_memory=True)
        logging.error("ERROR: unknown phase {}".format(dataset.phase))
        return None

    def init_metrics(self, train_metrics=None, val_metrics=None, test_metrics=None, *args, **kwargs):
        if train_metrics is not None:
            if not isinstance(train_metrics, MetricsList):
                train_metrics = MetricsList(train_metrics)
            self.train_metrics = train_metrics
        if val_metrics is not None:
            if not isinstance(val_metrics, MetricsList):
                val_metrics = MetricsList(val_metrics)
            self.val_metrics = val_metrics
        if test_metrics is not None:
            if not isinstance(test_metrics, MetricsList):
                test_metrics = MetricsList(test_metrics)
            self.test_metrics = test_metrics

    def configure_optimizers(self):
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        if self.l2_bias == 1:
            optimize_dict = [{'params': weight_p + bias_p, 'weight_decay': self.l2_weight}]
        else:
            optimize_dict = [{'params': weight_p, 'weight_decay': self.l2_weight},
                             {'params': bias_p, 'weight_decay': 0.0}]

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            LOGGER.info("Optimizer: GD")
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adagrad':
            LOGGER.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adam':
            LOGGER.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(optimize_dict, lr=self.lr)
        else:
            LOGGER.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        return optimizer

    def init_weights(self, m) -> None:
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        return

    def fit(self, train_data=None, val_data=None, trainer=None, **kwargs):
        default_para = {
            'gpus': 1,
            'weights_summary': None,
            'callbacks': [EarlyStopping(mode='max', patience=self.es_patience)]
        }
        default_para.update(kwargs)
        if trainer is None:
            trainer = self.trainer if self.trainer is not None else pl.Trainer(**default_para)
        if train_data is None:
            train_data = self.get_dataset(phase=TRAIN_PHASE)
        if val_data is None:
            val_data = self.get_dataset(phase=VAL_PHASE)
        if isinstance(train_data, Dataset):
            train_data = self.dataset_get_dataloader(train_data)
        if isinstance(val_data, Dataset):
            val_data = self.dataset_get_dataloader(val_data)
        return trainer.fit(model=self, train_dataloader=train_data, val_dataloaders=val_data)

    def test(self, test_data=None, trainer=None, **kwargs):
        if trainer is None:
            trainer = self.trainer
        if test_data is None:
            test_data = self.get_dataset(phase=TEST_PHASE)
        if isinstance(test_data, Dataset):
            test_data = self.dataset_get_dataloader(test_data)
        return trainer.test(model=self, test_dataloaders=test_data)

    def init_modules(self, *args, **kwargs) -> None:
        return

    def forward(self, batch, *args, **kwargs):
        return {}

    def training_step(self, batch, batch_idx, *args, **kwargs):
        return {}

    def training_epoch_end(self, outputs) -> None:
        if self.train_metrics is not None:
            metrics = self.train_metrics.compute()
            val_metrics = {}
            for key in metrics:
                val_metrics['train_' + key] = metrics[key]
            self.log_dict(metrics)

    def validation_step(self, batch, *args, **kwargs):
        out_dict = self.forward(batch)
        out_dict[LABEL] = batch[LABEL]
        if self.val_metrics is not None:
            self.val_metrics.update(out_dict)
        else:
            out_dict[LOSS] = self.loss_func(out_dict)
        return out_dict

    def validation_epoch_end(self, outputs):
        if self.val_metrics is not None:
            metrics = self.val_metrics.compute()
            val_metrics = {}
            for key in metrics:
                val_metrics['val_' + key] = metrics[key]
            self.log_dict(val_metrics)
            metrics_name = self.val_metrics.metrics_str[0]
            early_stop_on = metrics[metrics_name]
            if metrics_name in METRICS_SMALLER:
                early_stop_on = -early_stop_on
        else:
            loss = torch.stack([o[LOSS] for o in outputs])
            early_stop_on = -loss
        self.log('early_stop_on', early_stop_on)

    def test_step(self, batch, *args, **kwargs):
        out_dict = self.forward(batch)
        if LABEL in batch:
            out_dict[LABEL] = batch[LABEL]
            if self.test_metrics is not None:
                self.test_metrics.update(out_dict)
        return out_dict

    def test_epoch_end(self, outputs):
        if self.test_metrics is not None:
            metrics = self.test_metrics.compute()
            test_metrics = {}
            for key in metrics:
                test_metrics['test_' + key] = metrics[key]
            self.log_dict(test_metrics)
