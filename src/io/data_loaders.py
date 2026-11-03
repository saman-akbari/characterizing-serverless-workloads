import glob
import logging
from typing import Callable, List

import dask.dataframe as dd

from src.local import data_path

logging.basicConfig(level=logging.INFO)


def _read_globus24_request_body_size(
        transformer: Callable[[dd.DataFrame], dd.DataFrame],
) -> dd.DataFrame:
    file = glob.glob(f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv")
    df = dd.read_csv(
        file,
        blocksize="64MB",
        assume_missing=True,
        usecols=["argument_size"],
    )
    return transformer(df)


def _read_huawei25_metrics(
        transformer: Callable[[List[dd.DataFrame]], dd.DataFrame], metric: str
) -> dd.DataFrame:
    dfs_metric, dfs_req = [], []
    for region in range(1, 6):
        files_metric = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{region}/{metric}_avg/*.csv")
        )
        files_req = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{region}/requests/*.csv")
        )

        dfs_metric.append(
            dd.read_csv(
                files_metric,
                dtype={"day": "Int64", "time": "Int64"},
                blocksize=None,
                encoding="utf-8",
                assume_missing=True,
            )
        )

        dfs_req.append(
            dd.read_csv(
                files_req,
                dtype={"day": "Int64", "time": "Int64"},
                blocksize=None,
                encoding="utf-8",
                assume_missing=True,
            )
        )

    df_metric = dd.concat(dfs_metric, axis=0, interleave_partitions=True)
    df_req = dd.concat(dfs_req, axis=0, interleave_partitions=True)
    return transformer([df_metric, df_req])


def _read_huawei25_request_body_size(
        transformer: Callable[[List[dd.DataFrame]], dd.DataFrame],
) -> dd.DataFrame:
    return _read_huawei25_metrics(transformer, "requestBodySize")


def _read_huawei25_read_body_latency(
        transformer: Callable[[List[dd.DataFrame]], dd.DataFrame],
) -> dd.DataFrame:
    return _read_huawei25_metrics(transformer, "readBodyCost")


def _read_huawei25_write_rsp_latency(
        transformer: Callable[[List[dd.DataFrame]], dd.DataFrame],
) -> dd.DataFrame:
    return _read_huawei25_metrics(transformer, "writeRspCost")


def read_data(dataset: str, transformer: Callable) -> dd.DataFrame:
    dataset_readers = {
        "globus24_request_body_size": _read_globus24_request_body_size,
        "huawei25_request_body_size": _read_huawei25_request_body_size,
        "huawei25_read_body_latency": _read_huawei25_read_body_latency,
        "huawei25_write_rsp_latency": _read_huawei25_write_rsp_latency,
    }
    reader = dataset_readers.get(dataset)
    if reader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return reader(transformer)
