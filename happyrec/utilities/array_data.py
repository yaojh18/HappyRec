import os
from typing import Dict, Callable, Optional, Any

import numpy as np
from numpy import ndarray


def read_npy(file_path: str, **kwargs) -> ndarray:
    return np.load(file_path)


def write_npy(array: ndarray, file_path: str) -> None:
    np.save(file_path, array)


def read_csv(file_path: str, dtype) -> ndarray:
    return np.loadtxt(file_path, dtype=dtype, delimiter="\t")


def write_csv(array: ndarray, file_path: str) -> None:
    fmt = "%d" if np.issubdtype(array.dtype, np.integer) else "%g"
    np.savetxt(file_path, array, fmt=fmt, delimiter="\t")  # noqa


NPY_TYPE = "npy"
CSV_TYPE = "csv"

FILE_TYPE_PRIORITY = [NPY_TYPE, CSV_TYPE]

ARRAY_DATA_READ_FUNC: Dict[str, Callable[[str, Any], ndarray]] = {
    CSV_TYPE: read_csv,
    NPY_TYPE: read_npy,
}

ARRAY_DATA_WRITE_FUNC: Dict[str, Callable[[ndarray, str], None]] = {
    CSV_TYPE: write_csv,
    NPY_TYPE: write_npy,
}


def read_array_data(data_dir: str,
                    filename: str,
                    dtype,
                    ) -> Optional[ndarray]:
    array: Optional[ndarray] = None
    for file_type in ARRAY_DATA_READ_FUNC:
        file_path = os.path.join(data_dir, filename + "." + file_type)
        if os.path.exists(file_path):
            array = ARRAY_DATA_READ_FUNC[file_type](file_path, dtype)
            break
    return array


def write_array_data(data: ndarray,
                     data_dir: str,
                     filename: str,
                     file_type: str
                     ) -> None:
    file_path = os.path.join(data_dir, filename + "." + file_type)
    if not os.path.exists(file_path):
        ARRAY_DATA_WRITE_FUNC[file_type](data, file_path)
