# coding=utf-8
import sys
import os
import socket
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, '../')
sys.path.insert(0, './')

from happyrec.configs.constants import *
from happyrec.configs.settings import *
from happyrec.utilities.dataset import copy_ui_features, leave_out_csv, random_sample_eval_iids

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = './data/'
RATINGS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.data')
USERS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.user')
ITEMS_FILE = os.path.join(RAW_DATA, 'ml-100k/u.item')


def format_user_feature(out_file):
    print('format_user_feature', USERS_FILE)
    user_df = pd.read_csv(USERS_FILE, sep='|', header=None)
    user_df = user_df[[0, 1, 2, 3]]
    user_df.columns = [UID, 'u_age_c', 'u_gender_c', 'u_occupation_c']

    user_df['u_age_c'] = user_df['u_age_c'].apply(
        lambda x: 0 if x == 0 else 1 if x < 18 else (x + 5) // 10 if x < 45 else 5 if x < 50 else 6 if x < 56 else 7)

    user_df['u_gender_c'] = user_df['u_gender_c'].apply(lambda x: defaultdict(int, {'M': 1, 'F': 2})[x])
    occupation = {'none': 0, 'other': 1}
    for o in user_df['u_occupation_c'].unique():
        if o not in occupation:
            occupation[o] = len(occupation)
    user_df['u_occupation_c'] = user_df['u_occupation_c'].apply(lambda x: defaultdict(int, occupation)[x])
    user_df.index = user_df[UID]
    user_df.loc[0] = 0
    user_df = user_df.sort_index()
    # print(Counter(user_df['u_age_c']))
    # print(Counter(user_df['u_occupation_c']))
    # print(user_df)
    # user_df.info(null_counts=True)
    # print(user_df.min())
    user_df.to_csv(out_file, index=False, sep='\t')
    return user_df


def format_item_feature(out_file):
    print('format_item_feature', ITEMS_FILE, out_file)
    item_df = pd.read_csv(ITEMS_FILE, sep='|', header=None, encoding="ISO-8859-1")
    item_df = item_df.drop([1, 3, 4], axis=1)
    item_df.columns = [IID, 'i_year_c',
                       'i_Action_c', 'i_Adventure_c', 'i_Animation_c', "i_Children's_c", 'i_Comedy_c',
                       'i_Crime_c', 'i_Documentary_c', 'i_Drama_c', 'i_Fantasy_c', 'i_Film-Noir_c',
                       'i_Horror_c', 'i_Musical_c', 'i_Mystery_c', 'i_Romance_c', 'i_Sci-Fi_c',
                       'i_Thriller_c', 'i_War_c', 'i_Western_c', 'i_Other_c']
    item_df['i_year_c'] = item_df['i_year_c'].apply(lambda x: int(str(x).split('-')[-1]) if pd.notnull(x) else -1)
    seps = [0, 1940, 1950, 1960, 1970, 1980, 1985] + list(range(1990, int(item_df['i_year_c'].max() + 2)))
    year_dict = {}
    for i, sep in enumerate(seps[:-1]):
        for j in range(seps[i], seps[i + 1]):
            year_dict[j] = i + 1
    item_df['i_year_c'] = item_df['i_year_c'].apply(lambda x: defaultdict(int, year_dict)[x])
    for c in item_df.columns[2:]:
        item_df[c] = item_df[c] + 1
    item_df.index = item_df[IID]
    item_df.loc[0] = 0
    item_df = item_df.sort_index()
    # print(Counter(item_df['i_year_c']))
    # print(item_df)
    # item_df.info(null_counts=True)
    item_df.to_csv(out_file, index=False, sep='\t')
    return item_df


def format_all_inter(out_file, label01=False):
    print('format_all_inter', RATINGS_FILE, out_file)
    inter_df = pd.read_csv(RATINGS_FILE, sep='\t', header=None)
    inter_df.columns = [UID, IID, LABEL, TIME]
    inter_df = inter_df.sort_values(by=[TIME, UID], kind='mergesort')
    inter_df = inter_df.drop_duplicates([UID, IID]).reset_index(drop=True)
    if label01:
        inter_df[LABEL] = inter_df[LABEL].apply(lambda x: 1 if x > 3 else 0)
    print('label:', inter_df[LABEL].min(), inter_df[LABEL].max())
    print(Counter(inter_df[LABEL]))
    inter_df.to_csv(out_file, sep='\t', index=False)
    return inter_df


def main():
    user_file = os.path.join(DATA_DIR, 'ml100k.users.csv')
    format_user_feature(user_file)

    item_file = os.path.join(DATA_DIR, 'ml100k.items.csv')
    format_item_feature(item_file)

    all_inter_file = os.path.join(DATA_DIR, 'ml100k.all.csv')
    format_all_inter(all_inter_file, label01=False)
    dataset_name = 'ml100k-5-1'

    # all_inter_file = os.path.join(DATA_DIR, 'ml100k01.all.csv')
    # format_all_inter(all_inter_file, label01=True)
    # dataset_name = 'ml100k01-5-1'

    leave_out_csv(all_inter_file, dataset_name, warm_n=5, leave_n=1)

    copy_ui_features(dataset_name=dataset_name, user_file=user_file, item_file=item_file)
    random_sample_eval_iids(dataset_name, sample_n=1000)
    return


if __name__ == '__main__':
    main()
