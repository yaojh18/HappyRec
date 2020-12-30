# coding=utf-8
import pytorch_lightning as pl
import logging
import os
import hashlib

from ..metrics.metrics import METRICS_SMALLER
from ..configs.constants import *

DEFAULT_LOGGER = logging.getLogger("lightning")


def format_log_metrics_list(metrics_buf: list):
    metrics = metrics_buf[-1]
    log_str = ''
    for key in metrics:
        if key in METRICS_SMALLER:
            best = min([m[key] for m in metrics_buf])
            better = metrics[key] <= best
        else:
            best = max([m[key] for m in metrics_buf])
            better = metrics[key] >= best
        log_str += ' *' + key if better else ' ' + key
        log_str += '= {:.4f} '.format(metrics[key])
    return log_str


def logger_add_file_handler(file_path: str):
    for h in DEFAULT_LOGGER.handlers:
        if type(h) is logging.FileHandler:
            if h.baseFilename == os.path.abspath(file_path):
                return
    DEFAULT_LOGGER.addHandler(logging.FileHandler(filename=file_path))
    return


def hash_hparams(hparams):
    hash_code = hashlib.blake2b(digest_size=10, key=PROJECT_NAME.encode('utf-8'), person=PROJECT_NAME.encode('utf-8'))
    hash_code.update(str(hparams).encode('utf-8'))
    hash_code = hash_code.hexdigest()
    return hash_code


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, epoch, ckpt_name_metrics):
        return self.format_checkpoint_name(epoch, ckpt_name_metrics)
