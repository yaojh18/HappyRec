# coding=utf-8
import sys
import os
import re
import socket
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, '../')
sys.path.insert(0, './')

from happyrec.configs.constants import *
from happyrec.configs.settings import *
from happyrec.utilities.dataset import copy_ui_features, leave_out_csv, random_sample_eval_iids
from happyrec.utilities.io import check_mkdir

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = '../Dataset/MovieLens/ml-1m/'
RATINGS_FILE = os.path.join(RAW_DATA, 'ratings.dat')
USERS_FILE = os.path.join(RAW_DATA, 'users.dat')
ITEMS_FILE = os.path.join(RAW_DATA, 'movies.dat')


def format_user_feature(out_file):
    print('format_user_feature', USERS_FILE)
    user_df = pd.read_csv(USERS_FILE, sep='::', header=None, engine='python')
    user_df = user_df[[0, 1, 2, 3]]
    user_df.columns = [UID, 'u_gender_c', 'u_age_c', 'u_occupation_c']

    user_df['u_age_c'] = user_df['u_age_c'].apply(
        lambda x: 0 if x == 0 else 1 if x < 18 else (x + 5) // 10 if x < 45 else 5 if x < 50 else 6 if x < 56 else 7)

    user_df['u_gender_c'] = user_df['u_gender_c'].apply(lambda x: defaultdict(int, {'M': 1, 'F': 2})[x])
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
    item_df = pd.read_csv(ITEMS_FILE, sep='::', header=None, engine='python')
    genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "Other"]
    item_df.columns = [IID, 'i_year_c', 'i_genres_c']
    item_df['i_year_c'] = item_df['i_year_c'].apply(lambda x: int(re.search(r'.*?\(([0-9]+)\)$', x.strip()).group(1)))
    for genre in genres:
        item_df['i_' + genre + '_c'] = item_df['i_genres_c'].apply(lambda x: 1 if x.find(genre) == -1 else 2)
    item_df = item_df.drop(columns=['i_genres_c'])
    seps = [0, 1940, 1950, 1960, 1970, 1980, 1985] + list(range(1990, int(item_df['i_year_c'].max() + 2)))
    year_dict = {}
    for i, sep in enumerate(seps[:-1]):
        for j in range(seps[i], seps[i + 1]):
            year_dict[j] = i + 1
    item_df['i_year_c'] = item_df['i_year_c'].apply(lambda x: defaultdict(int, year_dict)[x])
    item_df.index = item_df[IID]
    for i in range(item_df[IID].max()):
        if i not in item_df.index:
            item_df.loc[i] = 0
            item_df.loc[i, IID] = i
    item_df = item_df.sort_index()
    # print(Counter(item_df['i_year_c']))
    # print(item_df)
    # item_df.info(null_counts=True)
    item_df.to_csv(out_file, index=False, sep='\t')
    return item_df


def format_all_inter(out_file, label01=False):
    print('format_all_inter', RATINGS_FILE, out_file)
    inter_df = pd.read_csv(RATINGS_FILE, sep='::', header=None, engine='python')
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
    data_dir = os.path.join(DATA_DIR, 'ml1m/')
    check_mkdir(data_dir)
    user_file = os.path.join(data_dir, 'ml1m.users.csv')
    format_user_feature(user_file)

    item_file = os.path.join(data_dir, 'ml1m.items.csv')
    format_item_feature(item_file)

    all_inter_file = os.path.join(data_dir, 'ml1m.all.csv')
    format_all_inter(all_inter_file, label01=False)
    dataset_name = 'ml1m-5-1'

    # all_inter_file = os.path.join(data_dir, 'ml1m01.all.csv')
    # format_all_inter(all_inter_file, label01=True)
    # dataset_name = 'ml1m01-5-1'

    leave_out_csv(all_inter_file, dataset_name, warm_n=5, leave_n=1)

    copy_ui_features(dataset_name=dataset_name, user_file=user_file, item_file=item_file)
    random_sample_eval_iids(dataset_name, sample_n=1000)
    return


if __name__ == '__main__':
    main()
