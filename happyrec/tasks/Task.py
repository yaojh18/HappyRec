# coding=utf-8

import os
import logging
from pytorch_lightning import seed_everything

from ..models import *
from ..configs.settings import *
from ..configs.constants import *


class Task(object):
    def run(self, args, task_args, model_args, trainer_args):
        # # init task
        seed_everything(task_args.random_seed)
        LOGGER.setLevel(task_args.verbose)

        # # init model
        model_name = eval('{0}.{0}'.format(task_args.model_name))
        model = model_name(**vars(model_args))

        # # read data
        model.read_data(dataset_dir=os.path.join(DATASET_DIR, task_args.dataset))

        # # init modules
        model.init_modules()
        model.summarize(mode='full')

        # # init metrics
        model.init_metrics(train_metrics=args.train_metrics, val_metrics=args.val_metrics,
                           test_metrics=args.test_metrics)

        # # train
        model.fit(**vars(trainer_args))

        # # test
        model.test()
        return
