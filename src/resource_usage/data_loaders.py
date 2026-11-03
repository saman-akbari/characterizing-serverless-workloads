import glob
import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.local import data_path


def _load_alibaba21_memory_limit() -> dd.DataFrame:
    file_pattern = f"{data_path}/alibaba21/*.csv"
    return dd.read_csv(
        file_pattern,
        dtype={"day": "Int64", "time": "Int64"},
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["functionName", "memoryMB"],
    )


def _load_azure19_memory_usage() -> dd.DataFrame:
    file_pattern = f"{data_path}/azure19/app_memory_percentiles.anon.d*.csv"
    return dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["HashApp", "AverageAllocatedMb"],
    )


def _load_huawei23_memory_usage() -> dd.DataFrame:
    file_pattern = f"{data_path}/huawei23/private_dataset/memory_usage_minute/*.csv"
    return dd.read_csv(
        file_pattern,
        dtype={"day": "Int64", "time": "Int64"},
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
    )


def _load_huawei23_memory_limit():
    file_pattern = f"{data_path}/huawei23/private_dataset/memory_limit_minute/*.csv"
    return dd.read_csv(
        file_pattern,
        dtype={"day": "Int64", "time": "Int64"},
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
    )


def _load_huawei23_cpu_limit():
    file_pattern = f"{data_path}/huawei23/private_dataset/cpu_limit_minute/*.csv"
    return dd.read_csv(
        file_pattern,
        dtype={"day": "Int64", "time": "Int64"},
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
    )


def _load_huawei23_cpu_usage() -> dd.DataFrame:
    file_pattern = f"{data_path}/huawei23/private_dataset/cpu_usage_minute/*.csv"
    return dd.read_csv(
        file_pattern,
        dtype={"day": "Int64", "time": "Int64"},
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
    )


def _load_huawei25(metric_type: str) -> dd.DataFrame:
    dfs = []
    for region in range(1, 6):
        files = sorted(
            glob.glob(
                f"{data_path}/huawei25/time_series/R{region}/{metric_type}_avg/*.csv"
            )
        )

        df = dd.read_csv(
            files,
            dtype={"day": "Int64", "time": "Int64"},
            blocksize="64MB",
            encoding="utf-8",
            assume_missing=True,
        )

        cols_to_drop = [c for c in ["day", "time"] if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        means_series = df.mean()
        region_df = means_series.to_frame(name="value")
        region_df = region_df.reset_index().rename(columns={"index": "pool"})
        region_df["pool"] = f"R{region}_" + region_df["pool"].astype("string")
        dfs.append(region_df)

    return dd.concat(dfs, axis=0, interleave_partitions=True)


def _load_huawei25_memory_usage() -> dd.DataFrame:
    return _load_huawei25("memory_usage")


def _load_huawei25_memory_limit() -> dd.DataFrame:
    return _load_huawei25("memory_usage")


def _load_huawei25_cpu_usage() -> dd.DataFrame:
    return _load_huawei25("cpu_usage")


def _load_huawei25_cpu_limit() -> dd.DataFrame:
    return _load_huawei25("cpu_usage")


def _load_ibm26_memory_limit() -> dd.DataFrame:
    df = pd.read_pickle(f"{data_path}/ibm26/app_configs_processed.pickle")
    all_memory = []
    for lst in df["AppContainerRequestMemory_MB"]:
        all_memory.extend([x for x in lst if not np.isnan(x) and x > 0])
    return dd.from_pandas(pd.DataFrame({"memory_limit_mb": all_memory}), npartitions=1)


def _load_ibm26_cpu_limit() -> dd.DataFrame:
    df = pd.read_pickle(f"{data_path}/ibm26/app_configs_processed.pickle")
    all_cpu = []
    for lst in df["AppContainerRequestCpu"]:
        all_cpu.extend([x for x in lst if not np.isnan(x) and x > 0])
    return dd.from_pandas(pd.DataFrame({"cpu_limit_cores": all_cpu}), npartitions=1)


def data_loaders(dataset: str):
    loaders = {
        "alibaba21_memory_limit": _load_alibaba21_memory_limit,
        "azure19_memory_usage": _load_azure19_memory_usage,
        "huawei23_memory_usage": _load_huawei23_memory_usage,
        "huawei23_memory_limit": _load_huawei23_memory_limit,
        "huawei23_cpu_limit": _load_huawei23_cpu_limit,
        "huawei23_cpu_usage": _load_huawei23_cpu_usage,
        "huawei25_memory_usage": _load_huawei25_memory_usage,
        "huawei25_memory_limit": _load_huawei25_memory_limit,
        "huawei25_cpu_usage": _load_huawei25_cpu_usage,
        "huawei25_cpu_limit": _load_huawei25_cpu_limit,
        "ibm26_memory_limit": _load_ibm26_memory_limit,
        "ibm26_cpu_limit": _load_ibm26_cpu_limit,
    }
    if dataset not in loaders:
        logging.warning(f"No data loader found for dataset: {dataset}")
    return loaders.get(dataset, lambda: dd.DataFrame([]))
