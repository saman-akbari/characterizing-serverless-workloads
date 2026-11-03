import logging
from typing import Callable, List

import dask.dataframe as dd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def _azure21_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["total_latency"] = df["duration"] * 1000  # convert to ms
    df["time"] = df["end_timestamp"] - df["duration"]
    df["time"] = (df["time"] // 60) * 60
    df["requests"] = 1

    df = df.groupby("time")[["total_latency", "requests"]].sum().reset_index()
    df["time"] = df["time"].astype(int)
    return df[["time", "total_latency", "requests"]]


def _globus24_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["execution_start", "execution_end"])
    df["duration"] = df["execution_end"] - df["execution_start"]
    df["total_latency"] = df["duration"] * 1000
    df["time"] = df["execution_start"].astype(np.int64).clip(lower=0)
    df["time"] = (df["time"] // 60) * 60
    df["requests"] = 1

    df = df.groupby("time")[["total_latency", "requests"]].sum().reset_index()
    return df[["time", "total_latency", "requests"]]


def _globus24_total_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["received", "result_received"])
    df["duration"] = df["result_received"] - df["received"]
    df["total_latency"] = df["duration"] * 1000
    df["time"] = df["received"].astype(np.int64).clip(lower=0)
    df["time"] = (df["time"] // 60) * 60
    df["requests"] = 1

    df = df.groupby("time")[["total_latency", "requests"]].sum().reset_index()
    return df[["time", "total_latency", "requests"]]


def _huawei23_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    df_val, df_req = dfs[0], dfs[1]

    def flatten(pdf_val, pdf_req):
        val_data = pdf_val.iloc[:, 2:]
        req_data = pdf_req.iloc[:, 2:]

        total_latency = (val_data * req_data).sum(axis=1)
        total_requests = req_data.sum(axis=1)

        result = pd.DataFrame(
            {
                "time": pdf_val["time"],
                "total_latency": total_latency,
                "requests": total_requests,
            }
        )
        return result

    meta = pd.DataFrame(
        {"time": [], "total_latency": [], "requests": []}, dtype="float64"
    )
    df = dd.map_partitions(flatten, df_val, df_req, meta=meta)
    return df[["time", "total_latency", "requests"]]


def _huawei23_total_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    df_func, df_plat, df_req = dfs[0], dfs[1], dfs[2]

    def flatten(pdf_func, pdf_plat, pdf_req):
        func_data = pdf_func.iloc[:, 2:]
        plat_data = pdf_plat.iloc[:, 2:]
        req_data = pdf_req.iloc[:, 2:]

        # platform delay + function delay
        total_delay = func_data + plat_data

        total_latency = (total_delay * req_data).sum(axis=1)
        total_requests = req_data.sum(axis=1)

        result = pd.DataFrame(
            {
                "time": pdf_func["time"],
                "total_latency": total_latency,
                "requests": total_requests,
            }
        )
        return result

    meta = pd.DataFrame(
        {"time": [], "total_latency": [], "requests": []}, dtype="float64"
    )
    df = dd.map_partitions(flatten, df_func, df_plat, df_req, meta=meta)
    return df[["time", "total_latency", "requests"]]


def _huawei25_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    df_val, df_req = dfs[0], dfs[1]

    def reduce_partition(pdf_val, pdf_req) -> pd.DataFrame:
        metadata = {"day", "time", "region"}
        cols = [c for c in pdf_val.columns if c not in metadata]

        total_latency = (pdf_val[cols] * pdf_req[cols]).sum(axis=1)
        requests = pdf_req[cols].sum(axis=1)

        return pd.DataFrame(
            {
                "day": pdf_val["day"],
                "time": pdf_val["time"],
                "total_latency": total_latency,
                "requests": requests,
            }
        )

    meta = pd.DataFrame(
        {"day": [], "time": [], "total_latency": [], "requests": []}, dtype="float64"
    )
    df = dd.map_partitions(reduce_partition, df_val, df_req, meta=meta)

    df = df.groupby(["day", "time"])[["total_latency", "requests"]].sum().reset_index()
    df["total_latency"] = df["total_latency"] * 1000

    return df[["time", "total_latency", "requests"]]


def _huawei25_total_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    return _huawei25_transform(dfs)


def _ibm26_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.groupby("time")[["total_latency", "requests"]].sum().reset_index()
    return df[["time", "total_latency", "requests"]]


def _ibm26_total_transform(df: dd.DataFrame) -> dd.DataFrame:
    return _ibm26_transform(df)


# Public factory -----------------------------------------------------------------


def latency_over_time_transformer(dataset: str) -> Callable:
    logging.info(f"Calling latency_over_time_transformer for dataset: {dataset}")
    mapping = {
        "azure21": _azure21_transform,
        "globus24": _globus24_transform,
        "globus24_total": _globus24_total_transform,
        "huawei23": _huawei23_transform,
        "huawei23_total": _huawei23_total_transform,
        "huawei25": _huawei25_transform,
        "huawei25_total": _huawei25_total_transform,
        "ibm26": _ibm26_transform,
        "ibm26_total": _ibm26_total_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
