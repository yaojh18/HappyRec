from typing import Optional, Dict

from numpy import ndarray

from ..utilities.table_data import (
    read_table_data, write_table_data, FEATHER_TYPE
)

Data = Dict[str, ndarray]

TRAIN_DATA_FILENAME = "train_data"
VALIDATION_DATA_FILENAME = "validation_data"
TEST_DATA_FILENAME = "test_data"

USER_DATA_FILENAME = "user_data"
ITEM_DATA_FILENAME = "item_data"


class DataReader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        self.train_data: Optional[Data] = None
        self.validation_data: Optional[Data] = None
        self.test_data: Optional[Data] = None

        self.user_data: Optional[Data] = None
        self.item_data: Optional[Data] = None

    def read_train_data(self, cache: bool = True) -> None:
        self.train_data = read_table_data(data_dir=self.data_dir,
                                          filename=TRAIN_DATA_FILENAME,
                                          )
        if self.train_data is None:
            raise RuntimeError()
        if cache:
            write_table_data(data=self.train_data,
                             data_dir=self.data_dir,
                             filename=TRAIN_DATA_FILENAME,
                             file_type=FEATHER_TYPE,
                             )

    def read_validation_data(self, cache: bool = True) -> None:
        self.validation_data = read_table_data(data_dir=self.data_dir,
                                               filename=VALIDATION_DATA_FILENAME,
                                               )
        if self.validation_data is None:
            raise RuntimeError()
        if cache:
            write_table_data(data=self.validation_data,
                             data_dir=self.data_dir,
                             filename=VALIDATION_DATA_FILENAME,
                             file_type=FEATHER_TYPE,
                             )

    def read_test_data(self, cache: bool = True) -> None:
        self.test_data = read_table_data(data_dir=self.data_dir,
                                         filename=TEST_DATA_FILENAME,
                                         )
        if self.test_data is None:
            raise RuntimeError()
        if cache:
            write_table_data(data=self.test_data,
                             data_dir=self.data_dir,
                             filename=TEST_DATA_FILENAME,
                             file_type=FEATHER_TYPE,
                             )

    def read_user_data(self, cache: bool = True) -> None:
        self.user_data = read_table_data(data_dir=self.data_dir,
                                         filename=USER_DATA_FILENAME,
                                         )
        if self.user_data is None:
            raise RuntimeError()
        if cache:
            write_table_data(data=self.user_data,
                             data_dir=self.data_dir,
                             filename=USER_DATA_FILENAME,
                             file_type=FEATHER_TYPE,
                             )

    def read_item_data(self, cache: bool = True) -> None:
        self.item_data = read_table_data(data_dir=self.data_dir,
                                         filename=ITEM_DATA_FILENAME,
                                         )
        if self.item_data is None:
            raise RuntimeError()
        if cache:
            write_table_data(data=self.item_data,
                             data_dir=self.data_dir,
                             filename=ITEM_DATA_FILENAME,
                             file_type=FEATHER_TYPE,
                             )
