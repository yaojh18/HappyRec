# coding=utf-8
from argparse import ArgumentParser
import itertools
from collections import defaultdict
import re
import os
import sys
import subprocess
import traceback
from multiprocessing.pool import Pool
from multiprocessing import Manager, Lock
from py3nvml import py3nvml
import time
import io
import pandas as pd
import numpy as np
from scipy import stats

from ..configs.settings import *
from ..models import *
from ..utilities.io import check_mkdir
from ..utilities.argument import get_class_init_args
from ..utilities.logging import DEFAULT_LOGGER, format_log_metrics_dict


# from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
# # # ignore 'BrokenPipeError: [Errno 32] Broken pipe'
# signal(SIGPIPE, SIG_IGN)

class GridSearch(object):
    lock = Lock()
    cuda_lock = Lock()
    log_lock = Lock()
    columns = ['model_name', 'description', 'train_metrics', 'val_metrics', 'test_metrics',
               'best_iter', 'time', 'date', 'version', 'server', 'cmd']

    @staticmethod
    def add_task_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_f', type=str, default='',
                            help='Input commands.')
        parser.add_argument('--out_csv', type=str, default='log.csv',
                            help='Output csv.')
        parser.add_argument('--auto_cuda', type=int, default=1,
                            help='Auto select {auto_cuda>0} cuda devices with largest free memory.')
        parser.add_argument('--cuda_mem', type=int, default=4096,
                            help='Select cuda devices with memory > {cuda_mem}MB')
        parser.add_argument('--cuda_wait', type=int, default=60,
                            help='Wait cuda devices for {cuda_wait}s for each try')
        parser.add_argument('--repeat', type=int, default=5,
                            help='Repeat times of each command.')
        parser.add_argument('--skip_cmd', type=int, default=0,
                            help='run {total_cmd} commands in in_f after skip {skip_cmd} commands.')
        parser.add_argument('--total_cmd', type=int, default=-1,
                            help='run {total_cmd} commands in in_f after skip {skip_cmd} commands. -1 for all commands.')
        parser.add_argument('--load_csv', type=int, default=-1,
                            help='if load_csv >= 0, load csv and continue run, reload {load_csv} rows of {out_csv}.')
        parser.add_argument('--grid_workers', type=int, default=1,
                            help='Number of processors when grid search.')
        parser.add_argument('--cuda', type=str, default='',
                            help='Set CUDA_VISIBLE_DEVICES')
        parser.add_argument('--try_max', type=int, default=10,
                            help='Maximum number of tries of running a cmd')
        return parser

    def __init__(self, search_dict: dict = None, cmd_list: dict = None,
                 in_f: str = '', out_csv: str = 'log.csv',
                 auto_cuda: int = 1, repeat: int = 5, skip_cmd: int = 0, total_cmd: int = -1,
                 load_csv: int = -1, grid_workers: int = 1,
                 cuda='', try_max=10, cuda_mem=4096, cuda_wait=60,
                 *args, **kwargs):
        self.search_dict = search_dict
        self.cmd_list = cmd_list
        self.in_f = in_f
        self.out_csv = out_csv
        self.auto_cuda = auto_cuda
        self.cuda_mem = cuda_mem
        self.cuda_wait = cuda_wait
        self.repeat = repeat
        self.skip_cmd = skip_cmd
        self.total_cmd = total_cmd
        self.load_csv = load_csv
        self.grid_workers = grid_workers
        self.cuda = cuda
        self.try_max = try_max
        self.grid_logger = DEFAULT_LOGGER

    @staticmethod
    def grid_args_list(search_dict, class_args=None, verbose=False):
        if class_args is not None:
            class_args = class_args if type(class_args) is list else get_class_init_args(class_args)
        grid_list = []
        if verbose:
            DEFAULT_LOGGER.info('grid search arguments:')
        for arg in search_dict:
            if class_args is not None and arg not in class_args:
                continue
            if type(search_dict[arg]) is not list:
                grid_list.append([(arg, search_dict[arg])])
                continue
            arg_list = []
            for v in search_dict[arg]:
                if v not in arg_list:
                    arg_list.append(v)
            grid_list.append([(arg, v) for v in arg_list])
            if verbose:
                DEFAULT_LOGGER.info('{}: {}'.format(arg, arg_list))
        grid_list = list(itertools.product(*grid_list))
        return grid_list

    @staticmethod
    def cmd_update_args(cmd, new_args):
        cmd = cmd.split()
        for arg, v in new_args:
            arg = '--' + arg
            p = 0
            while p < len(cmd):
                if cmd[p] == arg:
                    break
                p += 1
            if p < len(cmd):
                cmd[p + 1] = str(v)
            else:
                cmd.extend((arg, str(v)))
        return ' '.join(cmd).strip()

    @staticmethod
    def read_pycommand(file_path):
        in_f = open(file_path, 'r')
        lines = in_f.readlines()
        in_f.close()
        commands, search_dict = [], defaultdict(list)
        grids = []
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            if line.startswith('['):
                cmd = line.split('#')
                arg_name = cmd[-1].strip()
                arg_list = eval(cmd[0])
                if len(arg_list) > 0:
                    search_dict[arg_name].extend(arg_list)
                continue
            line = eval(line)
            grid_list = GridSearch.grid_args_list(search_dict)
            for grid in grid_list:
                commands.append(GridSearch.cmd_update_args(line, grid))
                grids.append(grid)
        return commands, grids

    @staticmethod
    def find_cuda(cuda_dict, try_max=-1):
        GridSearch.cuda_lock.acquire()
        py3nvml.nvmlInit()
        cuda_devices = [cuda for cuda in cuda_dict.keys() if type(cuda) is int]
        num_cuda = py3nvml.nvmlDeviceGetCount()
        if len(cuda_devices) == 0:
            cuda_devices = [i for i in range(num_cuda)]
            for i in range(num_cuda):
                cuda_dict[i] = -1
        else:
            cuda_devices = [i for i in cuda_devices if 0 <= i < num_cuda]
        try_cnt, select = 0, -1
        if len(cuda_devices) > 0:
            while try_max != try_cnt:
                max_mem, max_cuda = -1, -1
                now_time = time.time()
                for cuda in cuda_devices:
                    if now_time - cuda_dict[cuda] < cuda_dict['cuda_wait']:
                        continue
                    handle = py3nvml.nvmlDeviceGetHandleByIndex(cuda)
                    info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                    free_mem = (info.total - info.used) / (1024 * 1024)
                    if free_mem > max_mem:
                        max_mem, max_cuda = free_mem, cuda
                if max_mem >= cuda_dict['cuda_mem']:
                    select = max_cuda
                    cuda_dict[select] = now_time
                    break
                try_cnt += 1
                time.sleep(cuda_dict['cuda_wait'])
            GridSearch.log_lock.acquire()
            DEFAULT_LOGGER.debug('select cuda = {}'.format(select))
            GridSearch.log_lock.release()
        py3nvml.nvmlShutdown()
        GridSearch.cuda_lock.release()
        return select

    @staticmethod
    def single_run(args):
        cmd, try_max, cuda_dict = args
        try_cnt = 0
        while try_cnt < try_max:
            try:
                if type(cuda_dict) is str:
                    cmd = GridSearch.cmd_update_args(cmd, [('cuda', cuda_dict)])
                else:
                    cuda = GridSearch.find_cuda(cuda_dict)
                    if cuda >= 0:
                        cmd = GridSearch.cmd_update_args(cmd, [('cuda', cuda)])
                GridSearch.log_lock.acquire()
                DEFAULT_LOGGER.info(cmd)
                GridSearch.log_lock.release()
                result = subprocess.check_output(cmd, shell=True, bufsize=-1)
                break
            except subprocess.CalledProcessError as e:
                result = e.output
                try_cnt += 1
                if try_cnt >= try_max:
                    GridSearch.log_lock.acquire()
                    traceback.print_exc()
                    GridSearch.log_lock.release()
        return result.decode('utf-8')

    @staticmethod
    def find_info_in_output(outputs):
        info = {}
        lines = outputs.split(os.linesep)
        for c in GridSearch.columns:
            starts = c + ':'
            for line in lines[::-1]:
                line = line.strip()
                if line.startswith(starts):
                    line = line[len(starts):].strip()
                    info[c] = line
                    break
        return info

    @staticmethod
    def df_row_info(outputs, cmd, des):
        info = GridSearch.find_info_in_output(outputs)
        info['cmd'] = cmd
        des = [str(d[0]) + '=' + str(d[1]) for d in des]
        info['description'] = ','.join(des)
        return info

    def df_check_repeat(self, df):
        if len(df) <= 0 or self.repeat <= 1 or \
                not df['version'].tolist()[-1].endswith(str(DEFAULT_SEED + self.repeat - 1)):
            return df
        info = {}
        for c in ['model_name', 'description']:
            clist = df[c].tolist()
            info[c] = clist[-1]
            for i in range(min(self.repeat, len(df))):
                clist[-i - 1] = ''
            df[c] = clist
        for c in ['train_metrics', 'val_metrics', 'test_metrics']:
            idx = GridSearch.columns.index(c)
            clist = df[c].tolist()[-self.repeat:]
            mdict = defaultdict(list)
            for cline in clist:
                cline = cline.split()
                names = [n[:-1] for n in cline if n.endswith('=')]
                metrics = [float(n) for n in cline if not n.endswith('=') and len(n) > 0]
                for n, m in zip(names, metrics):
                    mdict[n].append(m)
            avgs, stds, sems = {}, {}, {}
            for m in mdict:
                avgs[m] = np.average(mdict[m])
                stds[m] = np.std(mdict[m], ddof=1)
                sems[m] = stats.sem(mdict[m], ddof=1)
            info[c] = format_log_metrics_dict(avgs)
            info[GridSearch.columns[idx + 3]] = format_log_metrics_dict(stds) + os.linesep + \
                                                format_log_metrics_dict(sems)
        df.loc[len(df)] = [info[c] if c in info else '' for c in GridSearch.columns]
        for i in range(3):
            df.loc[len(df)] = [''] * len(GridSearch.columns)
        return df

    def df_to_log_csv(self, df):
        if len(df) > 0:
            df.to_csv(os.path.join(LOG_CSV_DIR, self.out_csv), index=False)
            summary_df = df[(df['model_name'] != '') & (df[GridSearch.columns[-1]] == '')]
            summary_df.to_csv(os.path.join(LOG_CSV_DIR, self.out_csv.replace('.csv', '.summary.csv')), index=False)
            if len(summary_df) > 0:
                self.grid_logger.info('summary csv:')
                info_c = ['model_name', 'description', 'test_metrics']
                self.grid_logger.info(summary_df[info_c])
            if df[GridSearch.columns[-1]].tolist()[-1] != '':
                self.grid_logger.info('out csv:')
                info_c = ['model_name', 'description', 'version', 'test_metrics']
                self.grid_logger.info(df[-4:][info_c])
            self.grid_logger.info('')

    def run(self, *args, **kwargs):
        # # prepare cmd list
        cmd_list, des_list = [], []
        if self.in_f != '':
            cmd_list, des_list = GridSearch.read_pycommand(os.path.join(CMD_DIR, self.in_f))
        elif self.cmd_list is not None:
            grid_list = [] if self.search_dict is None else GridSearch.grid_args_list(self.search_dict)
            cmd_list = [GridSearch.cmd_update_args(cmd, grid) for cmd in self.cmd_list for grid in grid_list]
            des_list = [grid for cmd in self.cmd_list for grid in grid_list]
        if self.repeat > 1:
            cmd_list = [GridSearch.cmd_update_args(cmd, [('random_seed', seed)])
                        for cmd in cmd_list for seed in range(DEFAULT_SEED, DEFAULT_SEED + self.repeat)]
            des_list = [des for des in des_list for seed in range(DEFAULT_SEED, DEFAULT_SEED + self.repeat)]
        if self.skip_cmd > 0:
            cmd_list, des_list = cmd_list[self.skip_cmd:], des_list[self.skip_cmd:]
        if self.total_cmd > 0:
            cmd_list, des_list = cmd_list[:self.total_cmd], des_list[:self.total_cmd]
        self.grid_logger.info('total grid search running: {}'.format(len(cmd_list)))
        # if self.grid_workers > 1:
        #     cmd_list = [GridSearch.cmd_update_args(cmd, [('pbar', 0)]) for cmd in cmd_list]

        # # prepare cuda dict
        if self.auto_cuda > 0:
            manager = Manager()
            cuda_dict = manager.dict()
            for c in self.cuda.split(','):
                c = c.strip()
                if c != '':
                    cuda_dict[int(c)] = -1
            cuda_dict['cuda_mem'] = self.cuda_mem
            cuda_dict['cuda_wait'] = self.cuda_wait
        else:
            cuda_dict = self.cuda

        # # prepare out df
        check_mkdir(LOG_CSV_DIR)
        if self.load_csv >= 0:
            out_df = pd.read_csv(os.path.join(LOG_CSV_DIR, self.out_csv)).fillna('')
            out_df = out_df[:self.load_csv]
            for c in self.columns:
                if c not in out_df:
                    out_df[c] = ''
            out_df = out_df[self.columns]
        else:
            out_df = pd.DataFrame(columns=self.columns)
        out_df = self.df_check_repeat(out_df)
        self.df_to_log_csv(out_df)

        if len(cmd_list) > 0:
            results = [None] * len(cmd_list)
            # # multiprocess run
            if self.grid_workers > 1:
                pool = Pool(processes=self.grid_workers)
                map_args = zip(cmd_list, [self.try_max] * len(cmd_list), [cuda_dict] * len(cmd_list))
                results = pool.imap(GridSearch.single_run, map_args)
                pool.close()

            for idx, result in enumerate(results):
                if self.grid_workers <= 1:
                    result = GridSearch.single_run((cmd_list[idx], self.try_max, cuda_dict))
                GridSearch.log_lock.acquire()
                self.grid_logger.info(os.linesep + 'complete: {}/{}'.format(idx + 1, len(cmd_list)))
                try:
                    info = GridSearch.df_row_info(result, cmd_list[idx], des_list[idx])
                    out_df.loc[len(out_df)] = [info[c] if c in info else '' for c in GridSearch.columns]
                    out_df = self.df_check_repeat(out_df)
                    self.df_to_log_csv(out_df)
                except Exception as e:
                    traceback.print_exc()
                    continue
                GridSearch.log_lock.release()
        return out_df
