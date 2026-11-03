import glob
import logging
import re
from typing import Callable

import dask.dataframe as dd
import pandas as pd

from src.local import data_path

logging.basicConfig(level=logging.INFO)


def _read_huawei23(transformer: Callable[[dd.DataFrame], dd.DataFrame]) -> dd.DataFrame:
    # instances_minute contains the number of pods allocated to that function per minute
    # Wide format: day, time, func1, func2, ...
    file_pattern = f"{data_path}/huawei23/private_dataset/instances_minute/*.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        assume_missing=True,
    )
    return transformer(df)


def _read_huawei25(transformer: Callable[[dd.DataFrame], dd.DataFrame]) -> dd.DataFrame:
    # num_pods contains the number of running pods per function
    # Wide format: day, time, func1, func2, ...
    dfs = []
    for r in range(1, 6):
        files = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{r}/num_pods/*.csv")
        )
        if files:
            d = dd.read_csv(
                files,
                dtype={"day": "Int64", "time": "Int64"},
                blocksize="32MB",
                encoding="utf-8",
                assume_missing=True,
            )
            dfs.append(d)

    df = dd.concat(dfs, axis=0, interleave_partitions=True)
    return transformer(df)


def _read_huawei25_lifespan(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    file_pattern = f"{data_path}/huawei25/day30/R5/*.csv"
    df = dd.read_csv(
        file_pattern,
        usecols=["time_worker", "podID"],
        blocksize="64MB",
        dtype={"time_worker": "float64", "podID": "object"},
        assume_missing=True,
    )
    return transformer(df)


def _read_ibm26_scaling(
        transformer: Callable[[pd.DataFrame], pd.DataFrame],
) -> dd.DataFrame:
    """
    Reads IBM26 trace data for scaling metrics.
    Aggregates metrics (max, rates) per function across all weeks.
    """
    files = sorted(
        glob.glob(f"{data_path}/ibm26/week_*_processed.pickle"),
        key=lambda f: int(re.search(r"week_(\d+)_processed", f).group(1)),
    )

    results = []
    for file in files:
        logging.info(f"Processing scaling stats for IBM26 file: {file}")
        df_week = pd.read_pickle(file)
        results.append(transformer(df_week))

    full_df = pd.concat(results, ignore_index=True)
    final_df = (
        full_df.groupby("func_id")[
            ["max_instances", "scale_up_rate", "scale_down_rate"]
        ]
        .max()
        .reset_index()
    )

    return dd.from_pandas(final_df, npartitions=1)


def _read_ibm26_lifespan(
        transformer: Callable[[pd.DataFrame], pd.DataFrame],
) -> dd.DataFrame:
    """
    Reads IBM26 trace data for lifespan metrics.
    Aggregates min/max timestamps per pod across weeks to calculate lifespan.
    """
    files = sorted(
        glob.glob(f"{data_path}/ibm26/week_*_processed.pickle"),
        key=lambda f: int(re.search(r"week_(\d+)_processed", f).group(1)),
    )

    partial_aggs = []
    for file in files:
        logging.info(f"Processing lifespan stats for IBM26 file: {file}")
        df_week = pd.read_pickle(file)
        partial_aggs.append(transformer(df_week))

    full_df = pd.concat(partial_aggs, ignore_index=True)

    final_df = (
        full_df.groupby("pod_hash")
        .agg({"min_timestamp": "min", "max_timestamp": "max"})
        .reset_index()
    )

    final_df["lifespan"] = final_df["max_timestamp"] - final_df["min_timestamp"]
    return dd.from_pandas(final_df[["pod_hash", "lifespan"]], npartitions=1)


def _read_ibm26_config(
        transformer: Callable[[pd.DataFrame], pd.DataFrame],
) -> dd.DataFrame:
    df = pd.read_pickle(f"{data_path}/ibm26/app_configs_processed.pickle")
    return dd.from_pandas(transformer(df), npartitions=1)


def read_data(dataset: str, transformer: Callable) -> dd.DataFrame:
    readers = {
        "huawei23": _read_huawei23,
        "huawei25": _read_huawei25,
        "huawei25_lifespan": _read_huawei25_lifespan,
        "ibm26": _read_ibm26_scaling,
        "ibm26_lifespan": _read_ibm26_lifespan,
        "ibm26_concurrency_limit": _read_ibm26_config,
        "ibm26_min_scale": _read_ibm26_config,
    }

    reader = readers.get(dataset)
    if reader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return reader(transformer)
