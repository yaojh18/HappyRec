# coding=utf-8
import sys
import os
import socket
import numpy as np
import pandas as pd
import jsonlines
from collections import defaultdict, Counter

sys.path.insert(0, '../')
sys.path.insert(0, './')

from happyrec.configs.constants import *
from happyrec.configs.settings import *
from happyrec.utilities.dataset import copy_ui_features, leave_out_csv, random_sample_eval_iids
from happyrec.utilities.io import check_mkdir

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = '../Dataset/Amazon2014/'
AMAZON2014_DIR = os.path.join(DATA_DIR, 'Amazon2014/')
check_mkdir(AMAZON2014_DIR)


def category2name(category, ratings_only, label01=False):
    name = category.replace('_and_', '').replace('_', '')
    if not ratings_only:
        name = '5' + name
    if label01:
        name += '01'
    return name


def format_amazon_inter(inter_df, out_name, label01=False):
    # 按时间、uid、iid排序
    inter_df = inter_df.sort_values(by=[TIME, UID], kind='mergesort').reset_index(drop=True)

    # 给uid编号，从1开始
    uids = sorted(inter_df[UID].unique())
    uid_dict = dict(zip(uids, range(1, len(uids) + 1)))
    user_df = pd.DataFrame({UID: uids, 'user_id': uids})
    user_df[UID] = user_df[UID].apply(lambda x: uid_dict[x])
    user_df.index = user_df[UID]
    user_df.loc[0] = [0, '']
    user_df = user_df.sort_index()
    user_df.to_csv(os.path.join(AMAZON2014_DIR, '{}.uid_dict.csv'.format(out_name)), index=False)
    user_df[[UID]].to_csv(os.path.join(AMAZON2014_DIR, '{}.users.csv'.format(out_name)), index=False)

    # 给iid编号，从1开始
    iids = sorted(inter_df[IID].unique())
    iid_dict = dict(zip(iids, range(1, len(iids) + 1)))
    item_df = pd.DataFrame({IID: iids, 'item_id': iids})
    item_df[IID] = item_df[IID].apply(lambda x: iid_dict[x])
    item_df.index = item_df[IID]
    item_df.loc[0] = [0, '']
    item_df = item_df.sort_index()
    item_df.to_csv(os.path.join(AMAZON2014_DIR, '{}.iid_dict.csv'.format(out_name)), index=False)
    item_df[[IID]].to_csv(os.path.join(AMAZON2014_DIR, '{}.items.csv'.format(out_name)), index=False)

    inter_df[UID] = inter_df[UID].apply(lambda x: uid_dict[x])
    inter_df[IID] = inter_df[IID].apply(lambda x: iid_dict[x])

    # 如果要format成0（负向）- 1（正向）两种label，而不是评分，则认为评分大于3的表示喜欢为1，否则不喜欢为0
    if label01:
        inter_df[LABEL] = inter_df[LABEL].apply(lambda x: 1 if x > 3 else 0)
    print('label:', inter_df[LABEL].min(), inter_df[LABEL].max())
    print(Counter(inter_df[LABEL]))
    inter_df.to_csv(os.path.join(AMAZON2014_DIR, '{}.all.csv'.format(out_name)), sep='\t', index=False)
    return inter_df


# http://jmcauley.ucsd.edu/data/amazon/
# format amazon ratings only 数据集
def format_rating_only(category='', in_csv='', label01=False):
    if in_csv == '':
        in_csv = os.path.join(RAW_DATA, 'ratings_{}.csv'.format(category))
    assert in_csv != ''
    out_name = category2name(in_csv.split('ratings_')[-1].split('.csv')[0], ratings_only=True, label01=label01)
    inter_df = pd.read_csv(in_csv, header=None, names=[UID, IID, LABEL, TIME])
    return format_amazon_inter(inter_df=inter_df, out_name=out_name, label01=label01)


# http://jmcauley.ucsd.edu/data/amazon/
# format amazon 5-core 数据集
def format_5core(category='', in_json='', label01=False):
    if in_json == '':
        in_json = os.path.join(RAW_DATA, 'reviews_{}_5.json'.format(category))
    assert in_json != ''
    out_name = category2name(in_json.split('reviews_')[-1].split('_5.json')[0], ratings_only=False, label01=label01)

    # 读入json文件
    records = []
    with jsonlines.open(in_json, 'r') as reader:
        for record in reader:
            records.append(record)

    # 讲json信息转换问pandas DataFrame
    inter_df = pd.DataFrame()
    inter_df[UID] = [r['reviewerID'] for r in records]
    inter_df[IID] = [r['asin'] for r in records]
    inter_df[LABEL] = [r['overall'] for r in records]
    inter_df[TIME] = [r['unixReviewTime'] for r in records]
    return format_amazon_inter(inter_df=inter_df, out_name=out_name, label01=label01)


def main():
    max_eval_user = 10000
    category = 'Baby'
    # category = 'Electronics'

    ratings_only = False
    label01 = False
    if ratings_only:
        format_rating_only(category=category, label01=label01)
    else:
        format_5core(category=category, label01=label01)

    name = category2name(category, ratings_only=ratings_only, label01=label01)
    all_inter_file = os.path.join(AMAZON2014_DIR, '{}.all.csv'.format(name))
    user_file = os.path.join(AMAZON2014_DIR, '{}.users.csv'.format(name))
    item_file = os.path.join(AMAZON2014_DIR, '{}.items.csv'.format(name))

    dataset_name = name + '-1-1'
    leave_out_csv(all_inter_file, dataset_name, warm_n=1, leave_n=1, max_user=max_eval_user)

    # dataset_name = name + '-5-1'
    # leave_out_csv(all_inter_file, dataset_name, warm_n=5, leave_n=1, max_user=max_eval_user)

    copy_ui_features(dataset_name=dataset_name, user_file=user_file, item_file=item_file)
    random_sample_eval_iids(dataset_name, sample_n=1000)
    return


if __name__ == '__main__':
    main()
