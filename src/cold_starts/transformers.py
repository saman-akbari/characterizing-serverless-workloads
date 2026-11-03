import logging
from typing import Callable

import dask.dataframe as dd


def _huawei25_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["totalCost_cold_start"])
    df = df.rename(columns={"totalCost_cold_start": "cold_start"})

    if "runtime" in df.columns:
        df["runtime"] = df["runtime"].astype(str)
        df["runtime"] = df["runtime"].replace(
            {
                "Python2": "Python",
                "Python3": "Python",
                "Node.js": "Node.js",
                "Java": "Java",
                "Go1.x": "Go",
                "C#": ".NET (C\\#)",
                "PHP7.3": "PHP",
                "unknown": "Unknown",
                "http": "HTTP",
                "Custom": "Custom Container",
            }
        )

    return df


def _alibaba21_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["latency"])
    df = df.rename(columns={"latency": "cold_start"})

    if "runtime" in df.columns:
        df["runtime"] = df["runtime"].str.replace(r"[\d.]+$", "", regex=True)
        df["runtime"] = df["runtime"].replace(
            {
                "python": "Python",
                "nodejs": "Node.js",
                "java": "Java",
                "php": "PHP",
                "golang": "Go",
                "dotnetcore": ".NET (C\\#)",
                "custom": "Custom Runtime",
                "custom-container": "Custom Container",
                "debian": "Debian",
            }
        )
    return df


def _huawei25_breakdown_transform(df: dd.DataFrame) -> dd.DataFrame:
    cost_names = [
        "podAllocationCost",
        "deployCodeCost",
        "deployDependencyCost",
        "schedulingCost",
    ]
    df = df.dropna(subset=cost_names)
    return df


def _huawei25_rate_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["rate_req"] = (df["cold_starts"] / df["requests"]) * 100
    if "rate_func" in df.columns:
        df["rate_func"] = df["rate_func"] * 100
    return df


def _ibm26_rate_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["rate_req"] = (df["cold_starts"] / df["requests"]) * 100
    if "rate_func" in df.columns:
        df["rate_func"] = df["rate_func"] * 100
    df = df.dropna()
    return df


def _globus24_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df[df["waiting_for_nodes"].notnull()]
    df["cold_start"] = df["execution_start"] - df["waiting_for_nodes"]
    df = df.dropna(subset=["cold_start"])
    df = df[df["cold_start"] >= 0]
    return df


def _globus24_rate_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["rate_req"] = (df["cold_starts"] / df["requests"]) * 100
    if "rate_func" in df.columns:
        df["rate_func"] = df["rate_func"] * 100
    return df


def cold_starts_transformer(dataset: str) -> Callable:
    logging.info(f"Calling transformer for dataset: {dataset}")
    mapping = {
        "huawei25": _huawei25_transform,
        "huawei25_with_runtime": _huawei25_transform,
        "huawei25_breakdown": _huawei25_breakdown_transform,
        "huawei25_rate": _huawei25_rate_transform,
        "alibaba21": _alibaba21_transform,
        "ibm26_rate": _ibm26_rate_transform,
        "globus24": _globus24_transform,
        "globus24_rate": _globus24_rate_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
