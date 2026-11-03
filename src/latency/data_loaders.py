import gc
import glob
from typing import List

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.local import data_path


def _read_azure19() -> dd.DataFrame:
    file = glob.glob(f"{data_path}/azure19/function_durations_percentiles.anon*.csv")
    df = dd.read_csv(
        file,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["Average", "Count"],
    )
    return df


def _read_azure19_func() -> dd.DataFrame:
    file = glob.glob(f"{data_path}/azure19/function_durations_percentiles.anon*.csv")
    df = dd.read_csv(
        file,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["HashFunction", "Average", "Count"],
    )
    return df


def _read_azure21() -> dd.DataFrame:
    file = glob.glob(
        f"{data_path}/azure21/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    )
    df = dd.read_csv(
        file,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["duration"],
    )
    return df


def _read_azure21_func() -> dd.DataFrame:
    file = glob.glob(
        f"{data_path}/azure21/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    )
    df = dd.read_csv(
        file,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["func", "duration"],
    )
    return df


def _read_globus24() -> dd.DataFrame:
    file = glob.glob(f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv")
    df = dd.read_csv(
        file,
        blocksize="64MB",
        assume_missing=True,
        usecols=["execution_start", "execution_end"],
    )
    return df


def _read_globus24_func() -> dd.DataFrame:
    file = glob.glob(f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv")
    df = dd.read_csv(
        file,
        blocksize="64MB",
        assume_missing=True,
        usecols=["function_uuid", "execution_start", "execution_end"],
    )
    return df


def _read_globus24_total() -> dd.DataFrame:
    file = glob.glob(f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv")
    df = dd.read_csv(
        file,
        blocksize="64MB",
        assume_missing=True,
        usecols=["received", "result_received"],
    )
    return df


def _read_globus24_overhead() -> dd.DataFrame:
    file = glob.glob(f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv")
    df = dd.read_csv(
        file,
        blocksize="64MB",
        assume_missing=True,
        usecols=["received", "execution_start"],
    )
    return df


def _read_globus24_infra() -> dd.DataFrame:
    tasks_file = f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv"
    ends_file = f"{data_path}/globus24/GlobusComputeDataset-v1-endpoints.csv"

    tasks = dd.read_csv(
        tasks_file,
        blocksize="64MB",
        usecols=["endpoint_uuid", "execution_start", "execution_end"],
    )
    ends = dd.read_csv(
        ends_file, blocksize="64MB", usecols=["endpoint_uuid", "endpoint_type"]
    )

    df = tasks.merge(ends, on="endpoint_uuid", how="left")
    return df


def _read_huawei23() -> List[dd.DataFrame]:
    f_files = sorted(
        glob.glob(f"{data_path}/huawei23/private_dataset/function_delay_minute/*.csv")
    )
    r_files = sorted(
        glob.glob(f"{data_path}/huawei23/private_dataset/requests_minute/*.csv")
    )
    return [
        dd.read_csv(f_files, blocksize=None, assume_missing=True),
        dd.read_csv(r_files, blocksize=None, assume_missing=True),
    ]


def _read_huawei23_func() -> List[dd.DataFrame]:
    return _read_huawei23()


def _read_huawei23_total() -> List[dd.DataFrame]:
    p_files = sorted(
        glob.glob(f"{data_path}/huawei23/private_dataset/platform_delay_minute/*.csv")
    )
    f_files = sorted(
        glob.glob(f"{data_path}/huawei23/private_dataset/function_delay_minute/*.csv")
    )
    r_files = sorted(
        glob.glob(f"{data_path}/huawei23/private_dataset/requests_minute/*.csv")
    )
    return [
        dd.read_csv(p_files, blocksize=None, assume_missing=True),
        dd.read_csv(f_files, blocksize=None, assume_missing=True),
        dd.read_csv(r_files, blocksize=None, assume_missing=True),
    ]


def _read_huawei23_overhead() -> List[dd.DataFrame]:
    p_files = sorted(
        glob.glob(f"{data_path}/huawei23/private_dataset/platform_delay_minute/*.csv")
    )
    r_files = sorted(
        glob.glob(f"{data_path}/huawei23/private_dataset/requests_minute/*.csv")
    )
    return [
        dd.read_csv(p_files, blocksize=None, assume_missing=True),
        dd.read_csv(r_files, blocksize=None, assume_missing=True),
    ]


def _read_huawei25() -> List[List[dd.DataFrame]]:
    m_dfs, r_dfs = [], []
    for r in range(1, 6):
        m_files = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{r}/totalCost_avg/*.csv")
        )
        r_files = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{r}/requests/*.csv")
        )
        if m_files and r_files:
            m_dfs.append(dd.read_csv(m_files, blocksize=None, assume_missing=True))
            r_dfs.append(dd.read_csv(r_files, blocksize=None, assume_missing=True))
    return [m_dfs, r_dfs]


def _read_huawei25_func() -> List[List[dd.DataFrame]]:
    return _read_huawei25()


def _read_huawei25_total() -> List[List[dd.DataFrame]]:
    m_dfs, r_dfs = [], []
    for r in range(1, 6):
        m_files = sorted(
            glob.glob(
                f"{data_path}/huawei25/time_series/R{r}/totalCost_frontend_avg/*.csv"
            )
        )
        r_files = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{r}/requests/*.csv")
        )
        if m_files and r_files:
            m_dfs.append(dd.read_csv(m_files, blocksize=None, assume_missing=True))
            r_dfs.append(dd.read_csv(r_files, blocksize=None, assume_missing=True))
    return [m_dfs, r_dfs]


def _read_huawei25_overhead() -> List[List[dd.DataFrame]]:
    e_dfs, t_dfs, r_dfs = [], [], []
    for r in range(1, 6):
        e_files = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{r}/totalCost_avg/*.csv")
        )
        t_files = sorted(
            glob.glob(
                f"{data_path}/huawei25/time_series/R{r}/totalCost_frontend_avg/*.csv"
            )
        )
        r_files = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{r}/requests/*.csv")
        )
        if e_files and t_files and r_files:
            e_dfs.append(dd.read_csv(e_files, blocksize=None, assume_missing=True))
            t_dfs.append(dd.read_csv(t_files, blocksize=None, assume_missing=True))
            r_dfs.append(dd.read_csv(r_files, blocksize=None, assume_missing=True))
    return [e_dfs, t_dfs, r_dfs]


@dask.delayed
def _process_ibm26(file):
    df = pd.read_pickle(file)
    durations = []
    for lst in df["AppExecTimes"]:
        arr = np.array(lst)
        valid = arr[~np.isnan(arr)]
        if valid.size > 0:
            durations.append(valid)

    del df
    gc.collect()

    if durations:
        return pd.DataFrame({"duration": np.concatenate(durations)})
    return pd.DataFrame({"duration": pd.Series(dtype=float)})


def _read_ibm26() -> dd.DataFrame:
    file_list = sorted(glob.glob(f"{data_path}/ibm26/week_*_processed.pickle"))
    if not file_list:
        return dd.from_pandas(pd.DataFrame({"duration": pd.Series(dtype=float)}), npartitions=1)
    dfs = [_process_ibm26(f) for f in file_list]
    return dd.from_delayed(dfs)


@dask.delayed
def _process_ibm26_func(file_list):
    app_stats = {}  # hash -> [total_dur, count]
    for file in file_list:
        df = pd.read_pickle(file)

        if "AppHash" not in df.columns:
            del df
            gc.collect()
            continue

        for app, times in zip(df["AppHash"], df["AppExecTimes"]):
            t = np.array(times)
            valid = t[~np.isnan(t)]
            if len(valid) > 0:
                s = np.sum(valid)
                c = len(valid)
                if app not in app_stats:
                    app_stats[app] = [0.0, 0]
                app_stats[app][0] += s
                app_stats[app][1] += c

        del df
        gc.collect()

    data = []
    for app in app_stats:
        if app_stats[app][1] > 0:
            data.append(app_stats[app][0] / app_stats[app][1])
    return pd.DataFrame({"duration": data})


def _read_ibm26_func() -> dd.DataFrame:
    file_list = sorted(glob.glob(f"{data_path}/ibm26/week_*_processed.pickle"))
    if not file_list:
        return dd.from_pandas(pd.DataFrame({"duration": pd.Series(dtype=float)}), npartitions=1)
    return dd.from_delayed([_process_ibm26_func(file_list)])


@dask.delayed
def _process_ibm26_total(file):
    df = pd.read_pickle(file)
    durations = []
    for lst in df["TotalExecTimes"]:
        arr = np.array(lst)
        valid = arr[~np.isnan(arr)]
        if valid.size > 0:
            durations.append(valid)

    del df
    gc.collect()

    if durations:
        return pd.DataFrame({"duration": np.concatenate(durations)})
    return pd.DataFrame({"duration": pd.Series(dtype=float)})


def _read_ibm26_total() -> dd.DataFrame:
    file_list = sorted(glob.glob(f"{data_path}/ibm26/week_*_processed.pickle"))
    if not file_list:
        return dd.from_pandas(pd.DataFrame({"duration": pd.Series(dtype=float)}), npartitions=1)
    dfs = [_process_ibm26_total(f) for f in file_list]
    return dd.from_delayed(dfs)


@dask.delayed
def _process_ibm26_overhead(file):
    df = pd.read_pickle(file)
    durations = []
    for t_lst, a_lst in zip(df["TotalExecTimes"], df["AppExecTimes"]):
        t = np.array(t_lst)
        a = np.array(a_lst)
        valid = ~np.isnan(t) & ~np.isnan(a)
        if np.any(valid):
            durations.append(t[valid] - a[valid])

    del df
    gc.collect()

    if durations:
        return pd.DataFrame({"duration": np.concatenate(durations)})
    return pd.DataFrame({"duration": pd.Series(dtype=float)})


def _read_ibm26_overhead() -> dd.DataFrame:
    file_list = sorted(glob.glob(f"{data_path}/ibm26/week_*_processed.pickle"))
    if not file_list:
        return dd.from_pandas(pd.DataFrame({"duration": pd.Series(dtype=float)}), npartitions=1)
    dfs = [_process_ibm26_overhead(f) for f in file_list]
    return dd.from_delayed(dfs)


def read_data(dataset: str, transformer=None) -> dd.DataFrame:
    dataset_readers = {
        "azure19": _read_azure19,
        "azure19_func": _read_azure19_func,
        "azure21": _read_azure21,
        "azure21_func": _read_azure21_func,
        "globus24": _read_globus24,
        "globus24_func": _read_globus24_func,
        "globus24_total": _read_globus24_total,
        "globus24_overhead": _read_globus24_overhead,
        "globus24_infra": _read_globus24_infra,
        "huawei23": _read_huawei23,
        "huawei23_func": _read_huawei23_func,
        "huawei23_total": _read_huawei23_total,
        "huawei23_overhead": _read_huawei23_overhead,
        "huawei25": _read_huawei25,
        "huawei25_func": _read_huawei25_func,
        "huawei25_total": _read_huawei25_total,
        "huawei25_overhead": _read_huawei25_overhead,
        "huawei25_runtime": _read_huawei25,
        "ibm26": _read_ibm26,
        "ibm26_func": _read_ibm26_func,
        "ibm26_total": _read_ibm26_total,
        "ibm26_overhead": _read_ibm26_overhead,
    }
    reader = dataset_readers.get(dataset)
    raw_df = reader()
    return transformer(raw_df) if transformer else raw_df
