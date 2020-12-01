# coding=utf-8

import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from happyrec.models.Model import Model
from happyrec.models.RecModel import RecModel
from happyrec.models.BiasedMF import BiasedMF
from happyrec.models.WideDeep import WideDeep
from happyrec.data_readers.DataReader import DataReader
from happyrec.data_readers.RecReader import RecReader
from happyrec.datasets.Dataset import Dataset
from happyrec.datasets.RecDataset import RecDataset
from happyrec.configs.settings import *
from happyrec.configs.constants import *


def main():
    LOGGER.setLevel(logging.DEBUG)
    dataset = 'ml100k-5-1'
    # # define model (required)
    model_name = BiasedMF
    model = model_name(train_sample_n=1, eval_sample_n=-1, num_workers=4, es_patience=20, l2=1e-4)
    # model = model_name(train_sample_n=1, eval_sample_n=999, num_workers=4, es_patience=20, l2=1e-4)

    # # read data (required)
    model.read_data(dataset_dir=os.path.join(DATASET_DIR, dataset))  # option 1
    reader = model.reader
    # reader = RecReader(os.path.join(DATASET_DIR, dataset))
    # reader_para = model.read_data(reader=reader)  # option 2

    # # init modules (required)
    model.init_modules()
    model.summarize(mode='full')

    # # init metrics (optional, but recommended)
    model.init_metrics(train_metrics=None, val_metrics='hit@10', test_metrics='hit@10')

    # # init dataset (optional)
    train_set = model.get_dataset(phase=TRAIN_PHASE)
    train_set = RecDataset(data=reader.train_data, reader=reader, model=model, buffer_ds=1, phase=TRAIN_PHASE)
    val_set = model.get_dataset(phase=VAL_PHASE)
    val_set = RecDataset(data=reader.val_data, reader=reader, model=model, buffer_ds=1, phase=VAL_PHASE)
    test_set = model.get_dataset(phase=TEST_PHASE)
    test_set = RecDataset(data=reader.test_data, reader=reader, model=model, buffer_ds=1, phase=TEST_PHASE)

    # # init trainer (optional)
    # trainer = Trainer(gpus=1, callbacks=[EarlyStopping(mode='max', patience=20)], weights_summary=None)

    # # fit
    model.fit(max_epochs=100)  # option 1
    # model.fit(trainer=trainer, train_data=train_set, val_data=val_set)  # option 2
    # trainer.fit(model=model, train_dataloader=train_set.get_dataloader(),
    #             val_dataloaders=val_set.get_dataloader())  # option 3

    # # test
    test_result = model.test()  # option 1
    # test_result = model.test(trainer=trainer, test_data=test_set)  # option 2
    # test_result = trainer.test(model=model, test_dataloaders=test_set.get_dataloader())  # option 3
    LOGGER.info(test_result)
    return


if __name__ == '__main__':
    main()
