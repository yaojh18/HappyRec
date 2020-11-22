# coding=utf-8

from ..configs.constants import *
from ..configs.settings import *
from ..utilities.io import *
from ..utilities.formatter import *
from ..data_readers.DataReader import DataReader


class RecReader(DataReader):
    def read_user(self, filename: str, formatters: dict) -> dict:
        LOGGER.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.user_data = df2dict(df, formatters=formatters) if df is not None else {UID: np.array([0])}
        assert UID in self.user_data and len(self.user_data[UID]) == self.user_data[UID][-1] + 1
        LOGGER.debug("user data keys {}:{}".format(len(self.user_data), list(self.user_data.keys())))
        self.user_num = len(self.user_data[UID])
        LOGGER.info("user_num = {}".format(self.user_num))
        return self.user_data

    def read_item(self, filename: str, formatters: dict) -> dict:
        LOGGER.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.item_data = df2dict(df, formatters=formatters) if df is not None else {IID: np.array([0])}
        assert IID in self.item_data and len(self.item_data[IID]) == self.item_data[IID][-1] + 1
        LOGGER.debug("item data keys {}:{}".format(len(self.item_data), list(self.item_data.keys())))
        self.item_num = len(self.item_data[IID])
        LOGGER.info("item_num = {}".format(self.item_num))
        return self.item_data

    def read_val_iids(self, filename: str, formatters: dict, eval_sample_n: int = None) -> dict:
        LOGGER.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        val_iids = df2dict(df, formatters=formatters)
        for c in val_iids:
            c_data = filter_seqs(val_iids[c], max_len=eval_sample_n, padding=0)
            self.val_data[c] = c_data if c not in self.val_data \
                else np.concatenate([self.val_data[c], c_data], axis=1)
        LOGGER.info("validation eval_sample_n = {}".format(val_iids[EVAL_IIDS].shape))
        return val_iids

    def read_test_iids(self, filename: str, formatters: dict, eval_sample_n: int = None) -> dict:
        LOGGER.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        test_iids = df2dict(df, formatters=formatters)
        for c in test_iids:
            c_data = filter_seqs(test_iids[c], max_len=eval_sample_n, padding=0)
            self.test_data[c] = c_data if c not in self.test_data \
                else np.concatenate([self.test_data[c], c_data], axis=1)
        LOGGER.info("test eval_sample_n = {}".format(test_iids[EVAL_IIDS].shape))
        return test_iids
