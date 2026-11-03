from typing import Callable

import dask.dataframe as dd

from src.local import data_path


def _read_azure19(transformer: Callable[[dd.DataFrame], dd.DataFrame]) -> dd.DataFrame:
    file_pattern = f"{data_path}/azure19/invocations_per_function_md.anon.*.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="32MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["HashFunction", "Trigger"],
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
    )
    return transformer(df)


def read_data(
        dataset: str, transformer: Callable[[dd.DataFrame], dd.DataFrame]
) -> dd.DataFrame:
    dataset_readers = {
        "azure19": _read_azure19,
        "huawei25": _read_huawei25,
    }
    reader = dataset_readers.get(dataset)
    if reader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return reader(transformer)
