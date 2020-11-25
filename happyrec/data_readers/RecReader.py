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

    def prepare_user_features(self, include_uid: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        LOGGER.debug("prepare user features...")
        uids = self.user_data.pop(UID) if not include_uid else self.user_data[UID]

        mh_f_dict, base = self.multihot_features(
            data_dicts=self.user_data, combine=multihot_features,
            k_filter=lambda x: x.startswith(USER_F) and x.endswith(CAT_F)
        )
        LOGGER.debug('user multihot features = {}'.format(mh_f_dict))
        self.user_multihot_f_num = len(mh_f_dict)
        self.user_multihot_f_dim = base
        LOGGER.info('user_multihot_f_num = {}'.format(self.user_multihot_f_num))
        LOGGER.info('user_multihot_f_dim = {}'.format(self.user_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=self.user_data, combine=numeric_features,
            k_filter=lambda x: x.startswith(USER_F) and (x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        LOGGER.debug('user numeric features = {}'.format(nm_f_dict))
        self.user_numeric_f_num = len(nm_f_dict)
        LOGGER.info('user_numeric_f_num = {}'.format(self.user_numeric_f_num))

        self.user_data[UID] = uids
        return {**mh_f_dict, **nm_f_dict}

    def prepare_item_features(self, include_iid: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        LOGGER.debug("prepare item features...")
        iids = self.item_data.pop(IID) if not include_iid else self.item_data[IID]

        mh_f_dict, base = self.multihot_features(
            data_dicts=self.item_data, combine=multihot_features,
            k_filter=lambda x: x.startswith(ITEM_F) and x.endswith(CAT_F)
        )
        LOGGER.debug('item multihot features = {}'.format(mh_f_dict))
        self.item_multihot_f_num = len(mh_f_dict)
        self.item_multihot_f_dim = base
        LOGGER.info('item_multihot_f_num = {}'.format(self.item_multihot_f_num))
        LOGGER.info('item_multihot_f_dim = {}'.format(self.item_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=self.item_data, combine=numeric_features,
            k_filter=lambda x: x.startswith(ITEM_F) and (x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        LOGGER.debug('item numeric features = {}'.format(nm_f_dict))
        self.item_numeric_f_num = len(nm_f_dict)
        LOGGER.info('item_numeric_f_num = {}'.format(self.item_numeric_f_num))

        self.item_data[IID] = iids
        return {**mh_f_dict, **nm_f_dict}

    def prepare_ctxt_features(self, include_time: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        LOGGER.debug("prepare context features...")
        data_dicts = [d for d in [self.train_data, self.val_data, self.test_data] if d is not None]
        times = [d.pop(TIME) if not include_time else d[TIME] for d in data_dicts]

        mh_f_dict, base = self.multihot_features(
            data_dicts=data_dicts, combine=multihot_features,
            k_filter=lambda x: x.startswith(CTXT_F) and x.endswith(CAT_F)
        )
        LOGGER.debug('context multihot features = {}'.format(mh_f_dict))
        self.ctxt_multihot_f_num = len(mh_f_dict)
        self.ctxt_multihot_f_dim = base
        LOGGER.info('ctxt_multihot_f_num = {}'.format(self.ctxt_multihot_f_num))
        LOGGER.info('ctxt_multihot_f_dim = {}'.format(self.ctxt_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=data_dicts, combine=numeric_features,
            k_filter=lambda x: x.startswith(CTXT_F) and (x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        LOGGER.debug('context numeric features = {}'.format(nm_f_dict))
        self.ctxt_numeric_f_num = len(nm_f_dict)
        LOGGER.info('ctxt_numeric_f_num = {}'.format(self.ctxt_numeric_f_num))
        for i, d in enumerate(data_dicts):
            d[TIME] = times[i]
        return {**mh_f_dict, **nm_f_dict}

    # def user_feature_info(self, include_uid: bool = False,
    #                       float_f: bool = True, int_f: bool = True, cat_f: bool = True) -> dict:
    #     LOGGER.debug("prepare user feature info...")
    #     f_dict, base = {}, 0
    #     for k in self.user_data:
    #         if not k.startswith(USER_F) or (not include_uid and k == UID):
    #             continue
    #         if cat_f and k.endswith(CAT_F):
    #             max_f = np.max(self.user_data[k])
    #             f_dict[k] = (base, base + max_f + 1)
    #             base += max_f + 1
    #         if (float_f and k.endswith(FLOAT_F)) or (int_f and k.endswith(INT_F)):
    #             f_dict[k] = (base, base + 1)
    #             base += 1
    #     self.user_features = f_dict
    #     LOGGER.debug('user_features = {}'.format(f_dict))
    #     self.user_feature_num = len(f_dict)
    #     self.user_feature_dim = base
    #     LOGGER.info('user_feature_num = {}'.format(self.user_feature_num))
    #     LOGGER.info('user_feature_dim = {}'.format(self.user_feature_dim))
    #     return f_dict
    #
    # def item_feature_info(self, include_iid: bool = False,
    #                       float_f: bool = True, int_f: bool = True, cat_f: bool = True) -> dict:
    #     LOGGER.debug("prepare item feature info...")
    #     f_dict, base = {}, 0
    #     for k in self.item_data:
    #         if not k.startswith(ITEM_F) or (not include_iid and k == IID):
    #             continue
    #         if cat_f and k.endswith(CAT_F):
    #             max_f = np.max(self.item_data[k])
    #             f_dict[k] = (base, base + max_f + 1)
    #             base += max_f + 1
    #         if (float_f and k.endswith(FLOAT_F)) or (int_f and k.endswith(INT_F)):
    #             f_dict[k] = (base, base + 1)
    #             base += 1
    #     self.item_features = f_dict
    #     LOGGER.debug('item_features = {}'.format(f_dict))
    #     self.item_feature_num = len(f_dict)
    #     self.item_feature_dim = base
    #     LOGGER.info('item_feature_num = {}'.format(self.item_feature_num))
    #     LOGGER.info('item_feature_dim = {}'.format(self.item_feature_dim))
    #     return f_dict
    #
    # def ctxt_feature_info(self, include_time: bool = False,
    #                       float_f: bool = True, int_f: bool = True, cat_f: bool = True) -> dict:
    #     LOGGER.debug("prepare context feature info...")
    #     sets = [self.train_data, self.val_data, self.test_data]
    #     f_dict, base = {}, 0
    #     for k in self.train_data:
    #         if not k.startswith(CTXT_F) or (not include_time and k == TIME):
    #             continue
    #         if cat_f and k.endswith(CAT_F):
    #             max_f = np.max([np.max(d[k]) for d in sets if d is not None and k in d])
    #             f_dict[k] = (base, base + max_f + 1)
    #             base += max_f + 1
    #         if (float_f and k.endswith(FLOAT_F)) or (int_f and k.endswith(INT_F)):
    #             f_dict[k] = (base, base + 1)
    #             base += 1
    #     self.ctxt_features = f_dict
    #     LOGGER.debug('ctxt_features = {}'.format(f_dict))
    #     self.ctxt_feature_num = len(f_dict)
    #     self.ctxt_feature_dim = base
    #     LOGGER.info('ctxt_feature_num = {}'.format(self.ctxt_feature_num))
    #     LOGGER.info('ctxt_feature_dim = {}'.format(self.ctxt_feature_dim))
    #     return f_dict
