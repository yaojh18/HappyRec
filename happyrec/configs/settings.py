# coding=utf-8
import logging
import torch
from py3nvml import py3nvml

try:
    py3nvml.nvmlInit()
    NUM_CUDA = py3nvml.nvmlDeviceGetCount()
    py3nvml.nvmlShutdown()
except:
    NUM_CUDA = 0

DEFAULT_SEED = 1949
DATA_DIR = './data/'
DATASET_DIR = './dataset/'
MODEL_DIR = './model/'
CMD_DIR = './command/'
LOG_CSV_DIR = './log_csv/'

CKPT_DIR = 'checkpoints/'  # MODEL_DIR/CKPT_DIR
CKPT_F = 'best'  # MODEL_DIR/CKPT_DIR/CKPT_F.ckpt
LOG_F = 'log.txt'  # MODEL_DIR/LOG_F

# filenames without suffix, currently support .csv, .pickle, .feather
TRAIN_FILE = 'train'
VAL_FILE = 'val'
TEST_FILE = 'test'

USER_FILE = 'user'
ITEM_FILE = 'item'

VAL_IIDS_FILE = 'val_iids'
TEST_IIDS_FILE = 'test_iids'

DEFAULT_TRAINER_ARGS = {
    'auto_select_gpus': NUM_CUDA > 0,
    'deterministic': True,
    'callbacks': [],
    'check_val_every_n_epoch': 1,
    'fast_dev_run': 0,
    'gpus': 1 if NUM_CUDA > 0 else 0,
    'gradient_clip_val': 0.0,
    'logger': [],
    'max_epochs': 1000,
    'min_epochs': 1,
    'profiler': None,
    'progress_bar_refresh_rate': 10,
    'val_check_interval': 1.0,
    'weights_summary': None,
}

# Handle create tensor when multiprocessing on some cpu devices, there might be
# 'RuntimeError: received 0 items of ancdata' in pytorch dataloader
if not NUM_CUDA > 0:
    torch.multiprocessing.set_sharing_strategy('file_system')

# GLOBAL_ARGS = {
#     'pbar': True
# }
