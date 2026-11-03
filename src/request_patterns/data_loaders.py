import gc
import glob
import logging
import re
from typing import Callable

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import ctypes

from src.local import data_path

logging.basicConfig(level=logging.INFO)


def _read_azure19(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    file_pattern = f"{data_path}/azure19/invocations_per_function_md.anon.*.csv"
    minute_cols = [str(i) for i in range(1, 1441)]
    dtype_dict = {col: "Int64" for col in minute_cols}
    df = dd.read_csv(
        file_pattern,
        blocksize="16MB",
        encoding="utf-8",
        assume_missing=True,
        include_path_column=True,
        dtype=dtype_dict,
        usecols=["HashFunction"] + minute_cols,
    )
    df = transformer(df)
    return df


def _read_azure21(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    file = glob.glob(
        f"{data_path}/azure21/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    )
    df = dd.read_csv(
        file,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
    )
    df = transformer(df)
    return df


def _read_globus24(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    file_pattern = f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        assume_missing=True,
        usecols=["received", "function_uuid"],
    )
    df = transformer(df)
    return df


def _read_huawei23(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    file_pattern = f"{data_path}/huawei23/private_dataset/requests_minute/*.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        assume_missing=True,
    )
    df = transformer(df)
    return df


def _read_huawei23_second(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    file_pattern = f"{data_path}/huawei23/private_dataset/requests_second/*.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="8MB",
        assume_missing=True,
    )
    df = transformer(df)
    return df


def _read_huawei25(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    dfs = []
    for r in range(1, 6):
        files = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{r}/requests/*.csv")
        )
        dfs.append(
            dd.read_csv(
                files,
                dtype={"day": "Int64", "time": "Int64"},
                blocksize="32MB",
                encoding="utf-8",
                assume_missing=True,
            )
        )

    df = dd.concat(dfs, axis=0, interleave_partitions=True)
    df = transformer(df)
    return df

def trim_memory():
    """Force the OS to reclaim unused memory from the process."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


@dask.delayed
def _process_ibm26_rp(file):
    df = pd.read_pickle(file)
    df = df[df["InvocationTimes"].map(len) > 0]

    if df.empty:
        return pd.DataFrame({"func_id": [], "start_timestamp": [], "requests": []}).astype(
            {"func_id": "object", "start_timestamp": "int64", "requests": "int64"}
        )

    app_hashes = df["AppHash"].values
    invocations = df["InvocationTimes"].values
    lengths = np.array([len(lst) for lst in invocations])

    all_times = np.concatenate(list(invocations)).astype(np.int64)
    repeated_hashes = np.repeat(app_hashes, lengths)

    del df, app_hashes, invocations, lengths
    gc.collect()

    local_df = pd.DataFrame({
        "func_id": repeated_hashes,
        "start_timestamp": all_times
    })

    aggregated = local_df.groupby(["func_id", "start_timestamp"]).size().reset_index()
    aggregated.columns = ["func_id", "start_timestamp", "requests"]

    del local_df, all_times, repeated_hashes
    gc.collect()
    trim_memory()

    return aggregated


def _read_ibm26(transformer: Callable[[dd.DataFrame], dd.DataFrame]) -> dd.DataFrame:
    files = glob.glob(f"{data_path}/ibm26/week_*_processed.pickle")
    file_list = sorted(
        files, key=lambda f: int(re.search(r"week_(\d+)_processed", f).group(1))
    )

    meta = pd.DataFrame({
        "func_id": pd.Series(dtype=object),  # String
        "start_timestamp": pd.Series(dtype="int64"),
        "requests": pd.Series(dtype="int64")
    })

    if not file_list:
        return transformer(dd.from_pandas(meta, npartitions=1))

    dfs = [_process_ibm26_rp(f) for f in file_list]
    df_dask = dd.from_delayed(dfs, meta=meta)

    return transformer(df_dask)

def read_data(
        dataset: str, transformer: Callable[[dd.DataFrame], dd.DataFrame]
) -> dd.DataFrame:
    if dataset == "huawei23" and (
            transformer.__name__ == "_huawei23_iat_transform"
            or transformer.__name__ == "_huawei23_popularity_transform"
    ):
        return _read_huawei23_second(transformer)

    dataset_readers = {
        "azure19": _read_azure19,
        "azure21": _read_azure21,
        "globus24": _read_globus24,
        "huawei23": _read_huawei23,
        "huawei25": _read_huawei25,
        "ibm26": _read_ibm26,
    }
    reader = dataset_readers.get(dataset)
    if reader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return reader(transformer)