# coding=utf-8
import sys
import pandas as pd
import os
import numpy as np
from shutil import copyfile

from ..configs.constants import *
from ..configs.settings import *
from ..utilities.io import *
from ..utilities.rec import group_user_history_df, sample_iids


def random_split_data(all_data_file, dataset_name, vt_ratio=0.1):
    """
    随机切分已经生成的数据集文件 *.all.csv -> *.train.csv,*.validation.csv,*.test.csv
    :param all_data_file: 数据预处理完的文件 *.all.csv
    :param dataset_name: 给数据集起个名字
    :param vt_ratio: 验证集合测试集比例
    :return: pandas dataframe 训练集,验证集,测试集
    """
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    print('random_split_data', dir_name)
    if not os.path.exists(dir_name):  # 如果数据集文件夹dataset_name不存在，则创建该文件夹，dataset_name是文件夹名字
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep='\t')
    vt_size = int(len(all_data) * vt_ratio)
    validation_set = all_data.sample(n=vt_size).sort_index()
    all_data = all_data.drop(validation_set.index)
    test_set = all_data.sample(n=vt_size).sort_index()
    train_set = all_data.drop(test_set.index)
    print('train=%d validation=%d test=%d' % (len(train_set), len(validation_set), len(test_set)))

    write_df(train_set, dirname=dir_name, filename=TRAIN_FILE, df_type=CSV_TYPE)
    write_df(validation_set, dirname=dir_name, filename=VAL_FILE, df_type=CSV_TYPE)
    write_df(test_set, dirname=dir_name, filename=TEST_FILE, df_type=CSV_TYPE)
    return train_set, validation_set, test_set


def leave_out_df(all_df, leave_n=1, warm_n=5, split_n=1, max_user=-1, neg_thresh=0):
    '''
    TODO: need test whether exist bugs
    Split dfs according to the interaction history of users. Leave the last interactions into validation/test.
    :param all_df: dataframe of all interactions. MUST BE SORTED (usually by time).
    :param leave_n: leave leave_n pos interactions in the validation/test set
    :param warm_n: keep warm_n pos interactions in the training set
    :param split_n: split all_df into train_df and split_n others (each contains leave_n)
    :param max_user: random select max_user users into the split sets if number of users > max_user
    :param neg_thresh: threshold of interaction labels that regarded as positive (label > neg_thresh)
    :return: train_df, [list of split_n dfs]
    '''
    if type(max_user) is float:
        total_uids = all_df[UID].unique()
        max_user = int(max_user * len(total_uids))

    user_group = {}
    uids, labels = all_df[UID].values, all_df[LABEL].values
    for i in range(len(all_df)):
        uid, label = uids[i], labels[i]
        if uid not in user_group:
            user_group[uid], pre_cnt = [], 0
        else:
            pre_cnt = user_group[uid][-1][-1]
        user_group[uid].append((i, label, pre_cnt + 1 if label > neg_thresh else pre_cnt))

    train_index, split_dfs = [], []
    for i in range(split_n):
        all_uids = list(user_group.keys())
        np.random.shuffle(all_uids)
        split_df = []
        uid_cnt = 0
        while len(all_uids) > 0 and (max_user <= 0 or uid_cnt < max_user):
            uid = all_uids.pop(0)
            user_inters = user_group[uid]
            user_total, tmp_l = user_inters[-1][-1], len(user_inters)
            while user_inters[-1][-1] > warm_n and user_total - user_inters[-1][-1] < leave_n:
                split_df.append(user_inters.pop(-1)[0])
            if tmp_l != len(user_inters):
                uid_cnt += 1
            if len(user_inters) == 0 or user_inters[-1][-1] == warm_n:
                train_index.extend([inter[0] for inter in user_inters])
                del user_group[uid]
        split_dfs.append(all_df.loc[split_df].sort_index())
    for uid in user_group:
        train_index.extend([inter[0] for inter in user_group[uid]])
    train_df = all_df.loc[train_index].sort_index()
    return train_df, split_dfs[::-1]


def leave_out_csv(all_data_file, dataset_name, leave_n=1, warm_n=5, max_user=-1, neg_thresh=0):
    """
    默认all_data里的交互是按时间顺序排列的，按交互顺序，把最后的交互划分到验证集合测试集里
    :param all_data_file: 数据预处理完的文件 *.all.csv，交互按时间顺序排列
    :param dataset_name: 给数据集起个名字
    :param leave_n: 验证和测试集保留几个用户交互
    :param warm_n: 保证测试用户在训练集中至少有warm_n个交互，否则交互全部放在训练集中
    :return: pandas dataframe 训练集,验证集,测试集
    """
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    print('leave_out_by_time_csv', dir_name, leave_n, warm_n)
    if not os.path.exists(dir_name):  # 如果数据集文件夹dataset_name不存在，则创建该文件夹，dataset_name是文件夹名字
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep='\t')

    train_set, (validation_set, test_set) = leave_out_df(
        all_data, warm_n=warm_n, leave_n=leave_n, split_n=2, max_user=max_user, neg_thresh=neg_thresh)
    print('train=%d validation=%d test=%d' % (len(train_set), len(validation_set), len(test_set)))
    if UID in train_set.columns:
        print('train_user=%d validation_user=%d test_user=%d' %
              (len(train_set[UID].unique()), len(validation_set[UID].unique()), len(test_set[UID].unique())))

    write_df(train_set, dirname=dir_name, filename=TRAIN_FILE, df_type=CSV_TYPE)
    write_df(validation_set, dirname=dir_name, filename=VAL_FILE, df_type=CSV_TYPE)
    write_df(test_set, dirname=dir_name, filename=TEST_FILE, df_type=CSV_TYPE)
    return train_set, validation_set, test_set


def random_sample_eval_iids(dataset_name, sample_n=1000):
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    train_df = pd.read_csv(os.path.join(dir_name, TRAIN_FILE + '.csv'), sep='\t')
    val_df = pd.read_csv(os.path.join(dir_name, VAL_FILE + '.csv'), sep='\t')
    test_df = pd.read_csv(os.path.join(dir_name, TEST_FILE + '.csv'), sep='\t')
    item_df = pd.read_csv(os.path.join(dir_name, ITEM_FILE + '.csv'), sep='\t')
    item_num = len(item_df[IID])
    tvt_user_his = group_user_history_df([train_df, val_df, test_df])
    test_neg = sample_iids(sample_n=sample_n, uids=test_df[UID].values, item_num=item_num, exclude_iids=tvt_user_his)
    tv_user_his = group_user_history_df([train_df, val_df])
    val_neg = sample_iids(sample_n=sample_n, uids=val_df[UID].values, item_num=item_num, exclude_iids=tv_user_his)
    test_c = [','.join([str(i) for i in neg]) for neg in test_neg]
    test_iids = pd.DataFrame(data={EVAL_IIDS: test_c})
    test_iids.to_csv(os.path.join(dir_name, TEST_IIDS_FILE + '.csv'), sep='\t', index=False)
    val_c = [','.join([str(i) for i in neg]) for neg in val_neg]
    val_iids = pd.DataFrame(data={EVAL_IIDS: val_c})
    val_iids.to_csv(os.path.join(dir_name, VAL_IIDS_FILE + '.csv'), sep='\t', index=False)
    return


def copy_ui_features(dataset_name, user_file, item_file):
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    copyfile(user_file, os.path.join(dir_name, USER_FILE + '.csv'))
    copyfile(item_file, os.path.join(dir_name, ITEM_FILE + '.csv'))


def renumber_ids(df, old_column, new_column):
    old_ids = sorted(df[old_column].unique())
    id_dict = dict(zip(old_ids, range(1, len(old_ids) + 1)))
    id_df = pd.DataFrame({new_column: old_ids, old_column: old_ids})
    id_df[new_column] = id_df[new_column].apply(lambda x: id_dict[x])
    id_df.index = id_df[new_column]
    id_df.loc[0] = [0, '']
    id_df = id_df.sort_index()
    df[old_column] = df[old_column].apply(lambda x: id_dict[x])
    df = df.rename(columns={old_column: new_column})
    return df, id_df, id_dict


def read_id_dict(dict_csv, key_column, value_column, sep='\t'):
    df = pd.read_csv(dict_csv, sep=sep).dropna().astype(int)
    return dict(zip(df[key_column], df[value_column]))
