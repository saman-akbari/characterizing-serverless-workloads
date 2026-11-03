import logging
from typing import Callable

import dask.dataframe as dd

logging.basicConfig(level=logging.INFO)


def _alibaba21_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.drop_duplicates(subset=["functionName"])
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


def _huawei25_transform(df: dd.DataFrame) -> dd.DataFrame:
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


def runtimes_transformer(dataset: str) -> Callable:
    logging.info(f"Calling runtimes_transformer for dataset: {dataset}")
    mapping = {
        "alibaba21": _alibaba21_transform,
        "huawei25": _huawei25_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
