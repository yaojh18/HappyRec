# coding=utf-8

from ..configs.constants import *
from ..configs.settings import *
from ..utilities.io import *
from ..utilities.formatter import *
from ..utilities.rec import group_user_history
from ..data_readers.DataReader import DataReader


class RecReader(DataReader):
    def read_user(self, filename: str, formatters: dict) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.user_data = df2dict(df, formatters=formatters) if df is not None else {UID: np.array([0])}
        assert UID in self.user_data and len(self.user_data[UID]) == self.user_data[UID][-1] + 1
        self.reader_logger.debug("user data keys {}:{}".format(len(self.user_data), list(self.user_data.keys())))
        self.user_num = len(self.user_data[UID])
        self.reader_logger.info("user_num = {}".format(self.user_num))
        return self.user_data

    def read_item(self, filename: str, formatters: dict) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        self.item_data = df2dict(df, formatters=formatters) if df is not None else {IID: np.array([0])}
        assert IID in self.item_data and len(self.item_data[IID]) == self.item_data[IID][-1] + 1
        self.reader_logger.debug("item data keys {}:{}".format(len(self.item_data), list(self.item_data.keys())))
        self.item_num = len(self.item_data[IID])
        self.reader_logger.info("item_num = {}".format(self.item_num))
        return self.item_data

    def read_val_iids(self, filename: str, formatters: dict, sample_n: int = None) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        val_iids = df2dict(df, formatters=formatters)
        for c in val_iids:
            c_data = val_iids[c]
            if c in self.val_data:
                c_data = np.concatenate([self.val_data[c], c_data], axis=1)
            self.val_data[c] = filter_seqs(c_data, max_len=sample_n, padding=0)
        self.reader_logger.info("validation sample_n = {}".format(self.val_data[EVAL_IIDS].shape))
        return val_iids

    def read_test_iids(self, filename: str, formatters: dict, sample_n: int = None) -> dict:
        self.reader_logger.debug("read {}...".format(filename))
        df = read_df(dirname=self.dataset_dir, filename=filename)
        test_iids = df2dict(df, formatters=formatters)
        for c in test_iids:
            c_data = test_iids[c]
            if c in self.test_data:
                c_data = np.concatenate([self.test_data[c], c_data], axis=1)
            self.test_data[c] = filter_seqs(c_data, max_len=sample_n, padding=0)
        self.reader_logger.info("test sample_n = {}".format(self.test_data[EVAL_IIDS].shape))
        return test_iids

    def prepare_user_features(self, include_uid: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        self.reader_logger.debug("prepare user features...")
        uids = self.user_data.pop(UID) if not include_uid else self.user_data[UID]

        mh_f_dict, base = self.multihot_features(
            data_dicts=self.user_data, combine=multihot_features,
            k_filter=lambda x: x.startswith(USER_F) and x.endswith(CAT_F)
        )
        self.reader_logger.debug('user multihot features = {}'.format(mh_f_dict))
        self.user_multihot_f_num = len(mh_f_dict)
        self.user_multihot_f_dim = base
        self.reader_logger.info('user_multihot_f_num = {}'.format(self.user_multihot_f_num))
        self.reader_logger.info('user_multihot_f_dim = {}'.format(self.user_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=self.user_data, combine=numeric_features,
            k_filter=lambda x: x.startswith(USER_F) and (x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        self.reader_logger.debug('user numeric features = {}'.format(nm_f_dict))
        self.user_numeric_f_num = len(nm_f_dict)
        self.reader_logger.info('user_numeric_f_num = {}'.format(self.user_numeric_f_num))

        self.user_data[UID] = uids
        return {**mh_f_dict, **nm_f_dict}

    def prepare_item_features(self, include_iid: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        self.reader_logger.debug("prepare item features...")
        iids = self.item_data.pop(IID) if not include_iid else self.item_data[IID]

        mh_f_dict, base = self.multihot_features(
            data_dicts=self.item_data, combine=multihot_features,
            k_filter=lambda x: x.startswith(ITEM_F) and x.endswith(CAT_F)
        )
        self.reader_logger.debug('item multihot features = {}'.format(mh_f_dict))
        self.item_multihot_f_num = len(mh_f_dict)
        self.item_multihot_f_dim = base
        self.reader_logger.info('item_multihot_f_num = {}'.format(self.item_multihot_f_num))
        self.reader_logger.info('item_multihot_f_dim = {}'.format(self.item_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=self.item_data, combine=numeric_features,
            k_filter=lambda x: x.startswith(ITEM_F) and (x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        self.reader_logger.debug('item numeric features = {}'.format(nm_f_dict))
        self.item_numeric_f_num = len(nm_f_dict)
        self.reader_logger.info('item_numeric_f_num = {}'.format(self.item_numeric_f_num))

        self.item_data[IID] = iids
        return {**mh_f_dict, **nm_f_dict}

    def prepare_ctxt_features(self, include_time: bool = False,
                              multihot_features: str = None, numeric_features: str = None):
        self.reader_logger.debug("prepare context features...")
        data_dicts = [d for d in [self.train_data, self.val_data, self.test_data] if d is not None]
        times = [d.pop(TIME) if not include_time else d[TIME] for d in data_dicts]

        mh_f_dict, base = self.multihot_features(
            data_dicts=data_dicts, combine=multihot_features,
            k_filter=lambda x: x.startswith(CTXT_F) and x.endswith(CAT_F)
        )
        self.reader_logger.debug('context multihot features = {}'.format(mh_f_dict))
        self.ctxt_multihot_f_num = len(mh_f_dict)
        self.ctxt_multihot_f_dim = base
        self.reader_logger.info('ctxt_multihot_f_num = {}'.format(self.ctxt_multihot_f_num))
        self.reader_logger.info('ctxt_multihot_f_dim = {}'.format(self.ctxt_multihot_f_dim))

        nm_f_dict = self.numeric_features(
            data_dicts=data_dicts, combine=numeric_features,
            k_filter=lambda x: x.startswith(CTXT_F) and (x.endswith(INT_F) or x.endswith(FLOAT_F))
        )
        self.reader_logger.debug('context numeric features = {}'.format(nm_f_dict))
        self.ctxt_numeric_f_num = len(nm_f_dict)
        self.reader_logger.info('ctxt_numeric_f_num = {}'.format(self.ctxt_numeric_f_num))
        for i, d in enumerate(data_dicts):
            d[TIME] = times[i]
        return {**mh_f_dict, **nm_f_dict}

    def prepare_user_pos_his(self, label_filter=lambda x: x > 0):
        if HIS_POS_SEQ in self.user_data: return None
        user_dict = {}
        data_dicts = [(self.train_data, HIS_POS_TRAIN), (self.val_data, HIS_POS_VAL), (self.test_data, HIS_POS_SEQ)]
        for data, col_name in data_dicts:
            uids, iids, labels = data[UID], data[IID], data[LABEL]
            hi_list = []
            for uid, iid, label in zip(uids, iids, labels):
                if uid not in user_dict:
                    user_dict[uid] = []
                hi_list.append(len(user_dict[uid]))
                if label_filter(label):
                    user_dict[uid].append(iid)
            data[HIS_POS_IDX] = np.array(hi_list)
            if col_name != HIS_POS_SEQ:
                user_history = [len(user_dict[uid]) if uid in user_dict else 0 for uid in range(self.user_num)]
                self.user_data[col_name] = np.array(user_history)
            else:
                user_history = [np.array(user_dict[uid]) if uid in user_dict else np.array([], dtype=int)
                                for uid in range(self.user_num)]
                self.user_data[col_name] = np.array(user_history, dtype=object)
        return user_dict

    def prepare_item_pos_his(self, label_filter=lambda x: x > 0):
        if HIS_POS_SEQ in self.item_data: return None
        item_dict = {}
        data_dicts = [(self.train_data, HIS_POS_TRAIN), (self.val_data, HIS_POS_VAL), (self.test_data, HIS_POS_SEQ)]
        for data, col_name in data_dicts:
            uids, iids, labels = data[UID], data[IID], data[LABEL]
            hi_list = []
            for uid, iid, label in zip(uids, iids, labels):
                if iid not in item_dict:
                    item_dict[iid] = []
                hi_list.append(len(item_dict[iid]))
                if label_filter(label):
                    item_dict[iid].append(uid)
            data[HIS_POS_IDX] = np.array(hi_list)
            if col_name != HIS_POS_SEQ:
                item_history = [len(item_dict[iid]) if iid in item_dict else 0 for iid in range(self.item_num)]
                self.item_data[col_name] = np.array(item_history)
            else:
                item_history = [np.array(item_dict[iid]) if iid in item_dict else np.array([], dtype=int)
                                for iid in range(self.item_num)]
                self.item_data[col_name] = np.array(item_history, dtype=object)
        return item_dict
