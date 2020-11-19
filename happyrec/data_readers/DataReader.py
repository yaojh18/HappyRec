# coding=utf-8

from ..configs.constants import *
from ..configs.settings import *
from ..utilities.io import *
from ..utilities.formatter import *


class DataReader(object):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def read_train(self, filename: str, formatters: dict) -> dict:
        LOGGER.info("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.train_data = df2dict(df, formatters=formatters)
        LOGGER.debug("train data keys {}:{}".format(len(self.train_data), list(self.train_data.keys())))
        return self.train_data

    def read_validation(self, filename: str, formatters: dict) -> dict:
        LOGGER.info("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.val_data = df2dict(df, formatters=formatters)
        LOGGER.debug("validation data keys {}:{}".format(len(self.val_data), list(self.val_data.keys())))
        return self.val_data

    def read_test(self, filename: str, formatters: dict) -> dict:
        LOGGER.info("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.test_data = df2dict(df, formatters=formatters)
        LOGGER.debug("test data keys {}:{}".format(len(self.test_data), list(self.test_data.keys())))
        return self.test_data
