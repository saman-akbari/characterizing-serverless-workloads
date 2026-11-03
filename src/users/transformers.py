import logging
from typing import Callable

import dask.dataframe as dd
import pandas as pd


def _azure19_transform(df: dd.DataFrame) -> dd.DataFrame:
    def reduce_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        minute_cols = [str(i) for i in range(1, 1441)]
        result = pd.DataFrame()
        result["user_id"] = pdf["HashOwner"]
        result["function_id"] = pdf["HashFunction"]
        result["requests"] = pdf[minute_cols].sum(axis=1)
        return result

    meta = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="object"),
            "function_id": pd.Series(dtype="object"),
            "requests": pd.Series(dtype="float64"),
        }
    )

    df = df.map_partitions(reduce_partition, meta=meta)
    df_agg = df.groupby(["user_id", "function_id"])["requests"].sum().reset_index()
    return df_agg


def _azure20_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.rename(columns={"AnonUserId": "user_id", "AnonAppName": "function_id"})
    df_agg = (
        df.groupby(["user_id", "function_id"])["AnonFunctionInvocationId"]
        .nunique()
        .to_frame("requests")
        .reset_index()
    )
    return df_agg


def _globus24_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.rename(
        columns={"anonymized_user_uuid": "user_id", "function_uuid": "function_id"}
    )
    df_agg = (
        df.groupby(["user_id", "function_id"])
        .agg(requests=("task_uuid", "count"))
        .reset_index()
    )
    return df_agg


def _huawei25_transform(df: dd.DataFrame) -> dd.DataFrame:
    df_agg = df.groupby(["user_id", "function_id"])["requests"].sum().reset_index()
    return df_agg


def _ibm26_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.rename(
        columns={
            "NamespaceHash": "user_id",
            "AppHash": "function_id",
            "NumEvents": "requests",
        }
    )
    df_agg = df.groupby(["user_id", "function_id"])["requests"].sum().reset_index()
    return df_agg


def user_transformer(dataset: str) -> Callable:
    logging.info(f"Calling user transformer for dataset: {dataset}")
    mapping = {
        "azure19": _azure19_transform,
        "azure20": _azure20_transform,
        "globus24": _globus24_transform,
        "huawei25": _huawei25_transform,
        "ibm26": _ibm26_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
