# coding=utf-8
import logging

LOGGER = logging.getLogger("lightning")

DEFAULT_SEED = 1949
DATA_DIR = './data/'
DATASET_DIR = './dataset/'
MODEL_DIR = './model/'

# filenames without suffix, currently support .csv, .pickle, .feather
TRAIN_FILE = 'train'
VAL_FILE = 'val'
TEST_FILE = 'test'

USER_FILE = 'user'
ITEM_FILE = 'item'

VAL_IIDS_FILE = 'val_iids'
TEST_IIDS_FILE = 'test_iids'

DEFAULT_TRAINER_ARGS = {
    'auto_select_gpus': True,
    'deterministic': True,
    'callbacks': [],
    'check_val_every_n_epoch': 1,
    'fast_dev_run': 0,
    'gpus': 1,
    'gradient_clip_val': 0.0,
    'max_epochs': 1000,
    'min_epochs': 1,
    'profiler': None,
    'val_check_interval': 1.0,
    'weights_summary': None
}
