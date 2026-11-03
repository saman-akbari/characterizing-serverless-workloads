import logging
import re
from typing import Callable

import dask.dataframe as dd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def _azure19_transform(df: dd.DataFrame) -> dd.DataFrame:
    if "day" not in df.columns and "path" in df.columns:
        def extract_day(path):
            match = re.search(r"d(\d{2})\.csv$", str(path))
            return int(match.group(1)) if match else None

        df["day"] = df["path"].map(extract_day, meta=("day", "int64"))

    df["day"] = df["day"].astype("int64")

    minute_cols = [str(i) for i in range(1, 1441)]
    df = dd.melt(
        df,
        id_vars=["day"],
        value_vars=minute_cols,
        var_name="minute",
        value_name="requests",
    )

    df = df.dropna(subset=["requests"])
    df = df[df["requests"] > 0]
    df["requests"] = df["requests"].astype("int64")
    df["minute"] = df["minute"].astype("int64")
    df["time"] = ((df["day"] - 1) * 1440 * 60) + ((df["minute"] - 1) * 60)
    df = df.groupby("time")["requests"].sum().reset_index()
    return df


def _azure19_popularity_transform(df: dd.DataFrame) -> dd.DataFrame:
    minute_cols = [str(i) for i in range(1, 1441)]
    df["requests"] = df[minute_cols].sum(axis=1)
    df = df[["HashFunction", "requests"]]
    df_grouped = df.groupby("HashFunction")["requests"].sum().reset_index()
    return df_grouped.rename(columns={"HashFunction": "func_id"})


def _azure21_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["time"] = df["end_timestamp"] - df["duration"]
    df["time"] = (df["time"] // 60) * 60
    df = df.groupby("time").size().reset_index()
    df = df.rename(columns={df.columns[0]: "time", df.columns[1]: "requests"})
    df = df[["time", "requests"]]
    df["time"] = df["time"].astype(int)
    df["requests"] = df["requests"].astype(int)
    return df


def _azure21_iat_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["start_timestamp"] = df["end_timestamp"] - df["duration"]
    df["func_id"] = df["app"] + "_" + df["func"]
    df["requests"] = 1
    return df[["func_id", "start_timestamp", "requests"]]


def _globus24_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["received"])
    df["received"] = df["received"].astype(np.float64).astype(np.int64)
    df["received"] = df["received"].clip(lower=0)
    df = df.rename(columns={"received": "time"})
    df["time"] = (df["time"] // 60) * 60
    df = df.groupby("time").size().reset_index()
    df = df.rename(columns={0: "requests"})
    df = df[["time", "requests"]]
    return df


def _globus24_iat_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["received"])
    df["received"] = df["received"].astype(np.float64)
    df["requests"] = 1
    return df[["function_uuid", "received", "requests"]].rename(
        columns={"function_uuid": "func_id", "received": "start_timestamp"}
    )


def _huawei23_transform(df: dd.DataFrame) -> dd.DataFrame:
    df[df.columns[2]] = df.iloc[:, 2:].sum(axis=1)
    df = df.iloc[:, :3]
    df = df.rename(columns={df.columns[2]: "requests"})
    df = df[["time", "requests"]]
    return df


def _huawei23_iat_transform(df: dd.DataFrame) -> dd.DataFrame:
    value_vars = [c for c in df.columns if c not in ["day", "time"]]
    df_long = df.melt(
        id_vars=["day", "time"],
        value_vars=value_vars,
        var_name="func_id",
        value_name="requests",
    )
    df_long = df_long.dropna(subset=["requests"])
    df_long = df_long[df_long["requests"] > 0]

    df_long["requests"] = df_long["requests"].astype("int32")
    df_long["time"] = df_long["time"].astype("int32")
    df_long["func_id"] = (
            df_long["func_id"].astype(str) + "_day" + df_long["day"].astype(str)
    )

    # smooth out 1-second quantization artifacts
    df_long["time"] = df_long["time"].map_partitions(
        lambda s: s.astype("float64") + np.random.default_rng(0).random(len(s)),
        meta=("time", "float64"),
    )

    return df_long.rename(columns={"time": "start_timestamp"})[
        ["func_id", "start_timestamp", "requests"]
    ]


def _huawei25_transform(df: dd.DataFrame) -> dd.DataFrame:
    def reduce_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        metadata = {"day", "time", "region"}
        data_cols = [c for c in pdf.columns if c not in metadata]
        pdf["requests"] = pdf[data_cols].sum(axis=1)
        return pdf[["day", "time", "requests"]]

    meta = pd.DataFrame({"day": [], "time": [], "requests": []}, dtype="float64")
    df = df.map_partitions(reduce_partition, meta=meta)
    df = df.groupby(["day", "time"])["requests"].sum().reset_index()
    df = df[["time", "requests"]]
    return df


def _huawei25_popularity_transform(df: dd.DataFrame) -> dd.DataFrame:
    meta_cols = [c for c in df.columns if c in ["day", "time", "region"]]
    id_vars = []
    if meta_cols:
        id_vars = meta_cols
    value_vars = [c for c in df.columns if c not in meta_cols]
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="func_id",
        value_name="requests",
    )
    df_melted = df_melted.dropna(subset=["requests"])
    df_melted = df_melted[df_melted["requests"] > 0]
    return df_melted.groupby("func_id")["requests"].sum().reset_index()


def _ibm26_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.rename(columns={"start_timestamp": "time"})
    df["time"] = (df["time"] // 60) * 60
    df = df.groupby("time")["requests"].sum().reset_index()
    return df


def _ibm26_iat_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["start_timestamp"] = df["start_timestamp"].astype(float)
    return df


# Public factory -----------------------------------------------------------------


def request_patterns_transformer(dataset: str) -> Callable:
    logging.info(f"Calling request_patterns_transformer for dataset: {dataset}")
    mapping = {
        "azure19": _azure19_transform,
        "azure21": _azure21_transform,
        "globus24": _globus24_transform,
        "huawei23": _huawei23_transform,
        "huawei25": _huawei25_transform,
        "ibm26": _ibm26_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer


def iat_transformer(dataset: str) -> Callable:
    logging.info(f"Calling iat_transformer for dataset: {dataset}")
    mapping = {
        "azure21": _azure21_iat_transform,
        "globus24": _globus24_iat_transform,
        "huawei23": _huawei23_iat_transform,
        "ibm26": _ibm26_iat_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(
            f"IAT analysis not supported/configured for dataset: {dataset}"
        )

    return transformer


def popularity_transformer(dataset: str) -> Callable:
    logging.info(f"Calling popularity_transformer for dataset: {dataset}")
    mapping = {
        "azure19": _azure19_popularity_transform,
        "azure21": _azure21_iat_transform,
        "globus24": _globus24_iat_transform,
        "huawei23": _huawei23_iat_transform,
        "huawei25": _huawei25_popularity_transform,
        "ibm26": _ibm26_iat_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(
            f"Popularity analysis not supported/configured for dataset: {dataset}"
        )

    return transformer
