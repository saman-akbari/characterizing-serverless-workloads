from typing import Callable

import dask.dataframe as dd

from src.local import data_path


def _read_alibaba21(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    file_pattern = f"{data_path}/alibaba21/*.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["functionName", "runtime"],
    )
    return transformer(df)


def _read_huawei25(transformer: Callable[[dd.DataFrame], dd.DataFrame]) -> dd.DataFrame:
    file_pattern = (
        f"{data_path}/huawei25/trigger_types/R2/df_funcID_runtime_triggerType.csv"
    )
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["runtime"],
    )
    return transformer(df)


def read_data(
        dataset: str, transformer: Callable[[dd.DataFrame], dd.DataFrame]
) -> dd.DataFrame:
    dataset_readers = {
        "alibaba21": _read_alibaba21,
        "huawei25": _read_huawei25,
    }
    reader = dataset_readers.get(dataset)
    if reader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return reader(transformer)
