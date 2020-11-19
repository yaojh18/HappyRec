# coding=utf-8
import logging

LOGGER = logging.getLogger("lightning")

DEFAULT_SEED = 2020
DATA_DIR = './data/'
DATASET_DIR = './dataset/'

# filenames without suffix, currently support .csv, .pickle, .feather
TRAIN_FILE = 'train'
VAL_FILE = 'val'
TEST_FILE = 'test'

USER_FILE = 'user'
ITEM_FILE = 'item'

VAL_IIDS_FILE = 'val_iids'
TEST_IIDS_FILE = 'test_iids'
