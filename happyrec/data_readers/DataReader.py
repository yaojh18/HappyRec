# coding=utf-8

from ..configs.constants import *
from ..configs.settings import *
from ..utilities.io import *
from ..utilities.formatter import *
from ..utilities.logging import DEFAULT_LOGGER


class DataReader(object):
    def __init__(self, dataset_dir: str, reader_logger=DEFAULT_LOGGER):
        self.dataset_dir = dataset_dir
        self.reader_logger = reader_logger

    def read_train(self, filename: str, formatters: dict) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.reader_logger.info("train df length = {}".format(len(df)))
        self.train_data = df2dict(df, formatters=formatters)
        self.reader_logger.debug("train data keys {}:{}".format(len(self.train_data), list(self.train_data.keys())))
        return self.train_data

    def read_validation(self, filename: str, formatters: dict) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.reader_logger.info("validation df length = {}".format(len(df)))
        self.val_data = df2dict(df, formatters=formatters)
        self.reader_logger.debug("validation data keys {}:{}".format(len(self.val_data), list(self.val_data.keys())))
        return self.val_data

    def read_test(self, filename: str, formatters: dict) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.reader_logger.info("test df length = {}".format(len(df)))
        self.test_data = df2dict(df, formatters=formatters)
        self.reader_logger.debug("test data keys {}:{}".format(len(self.test_data), list(self.test_data.keys())))
        return self.test_data

    def multihot_features(self, data_dicts: dict or list, combine: str = None,
                          k_filter=lambda x: x.endswith(CAT_F)):
        if type(data_dicts) is dict:
            data_dicts = [data_dicts]
        f_dict, base = {}, 0
        for k in data_dicts[0]:
            if k_filter(k):
                max_f = int(np.max([np.max(d[k]) for d in data_dicts if d is not None and k in d]))
                f_dict[k] = (base, base + max_f + 1)
                base += max_f + 1
        if combine is not None and len(f_dict) != 0:
            for data_dict in data_dicts:
                data_dict[combine] = np.stack([data_dict[k] + f_dict[k][0] for k in f_dict], axis=-1)
                for k in f_dict:
                    data_dict.pop(k)
        return f_dict, base

    def numeric_features(self, data_dicts: dict or list, combine: str = None,
                         k_filter=lambda x: x.endswith(FLOAT_F) or x.endswith(INT_F)):
        if type(data_dicts) is dict:
            data_dicts = [data_dicts]
        f_dict, base = {}, 0
        for k in data_dicts[0]:
            if k_filter(k):
                min_f = np.max([np.min(d[k]) for d in data_dicts if d is not None and k in d])
                max_f = np.max([np.max(d[k]) for d in data_dicts if d is not None and k in d])
                f_dict[k] = (min_f, max_f)
        if combine is not None and len(f_dict) != 0:
            for data_dict in data_dicts:
                data_dict[combine] = np.stack([data_dict[k] for k in f_dict], axis=-1)
                for k in f_dict:
                    data_dict.pop(k)
        return f_dict
