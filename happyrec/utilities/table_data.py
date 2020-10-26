import os
import pickle
from typing import Dict, Callable, Optional

import pandas as pd
from numpy import ndarray
from pandas import DataFrame


def dafaframe_to_data(dataframe: DataFrame) -> Dict[str, ndarray]:
    return {key: dataframe[key].values for key in dataframe}


def data_to_dataframe(data: Dict[str, ndarray]) -> DataFrame:
    return DataFrame(data)


def read_feather(file_path: str) -> DataFrame:
    return pd.read_feather(file_path)


def write_feather(dataframe: DataFrame, file_path: str) -> None:
    dataframe.to_feather(file_path)


def read_pickle(file_path: str) -> DataFrame:
    return pd.read_pickle(file_path)


def write_pickle(dataframe: DataFrame, file_path: str) -> None:
    dataframe.to_pickle(file_path, protocol=pickle.HIGHEST_PROTOCOL)


def read_csv(file_path: str) -> DataFrame:
    return pd.read_csv(file_path, sep="\t")


def write_csv(dataframe: DataFrame, file_path: str) -> None:
    dataframe.to_csv(file_path, sep="\t", index=False)


FEATHER_TYPE = "feather"
PICKLE_TYPE = "pickle"
CSV_TYPE = "csv"

FILE_TYPE_PRIORITY = [FEATHER_TYPE, PICKLE_TYPE, CSV_TYPE]

TABLE_DATA_READ_FUNC: Dict[str, Callable[[str], DataFrame]] = {
    CSV_TYPE: read_csv,
    PICKLE_TYPE: read_pickle,
    FEATHER_TYPE: read_feather,
}

TABLE_DATA_WRITE_FUNC: Dict[str, Callable[[DataFrame, str], None]] = {
    CSV_TYPE: write_csv,
    PICKLE_TYPE: write_pickle,
    FEATHER_TYPE: write_feather,
}


def read_table_data(data_dir: str,
                    filename: str,
                    ) -> Optional[Dict[str, ndarray]]:
    dataframe: Optional[DataFrame] = None
    for file_type in TABLE_DATA_READ_FUNC:
        file_path = os.path.join(data_dir, filename + "." + file_type)
        if os.path.exists(file_path):
            dataframe = TABLE_DATA_READ_FUNC[file_type](file_path)
            break
    if dataframe is None:
        return dataframe
    return dafaframe_to_data(dataframe)


def write_table_data(data: Dict[str, ndarray],
                     data_dir: str,
                     filename: str,
                     file_type: str
                     ) -> None:
    file_path = os.path.join(data_dir, filename + "." + file_type)
    if not os.path.exists(file_path):
        TABLE_DATA_WRITE_FUNC[file_type](data_to_dataframe(data), file_path)
