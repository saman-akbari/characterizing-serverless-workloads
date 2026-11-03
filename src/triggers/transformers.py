import logging
from typing import Callable

import dask.dataframe as dd

logging.basicConfig(level=logging.INFO)


def _azure19_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.drop_duplicates(subset=["HashFunction"])
    df = df.rename(columns={"Trigger": "trigger"})

    df["trigger"] = df["trigger"].str.capitalize()
    df["trigger"] = df["trigger"].replace({"Http": "HTTP"})
    return df[["trigger"]]


def _huawei25_transform(df: dd.DataFrame) -> dd.DataFrame:
    target_col = df.columns[3]

    def extract_trigger(val):
        if not isinstance(val, str):
            return str(val)
        return val.split("-")[0].upper()

    df["trigger"] = df[target_col].map(extract_trigger, meta=("trigger", "object"))

    df["trigger"] = df["trigger"].replace(
        {
            "TIMER": "Timer",
            "KAFKA": "Queue",
            "UNKNOWN": "Unknown",
            "APIG": "HTTP",
            "OBS": "Storage",
            "WORKFLOW": "Orchestration",
            "CTS": "Event",
            "LTS": "Event",
            "DIS": "Event",
            "SMN": "Event",
        }
    )

    return df[["trigger"]]


def triggers_transformer(dataset: str) -> Callable:
    logging.info(f"Calling triggers_transformer for dataset: {dataset}")
    mapping = {
        "azure19": _azure19_transform,
        "huawei25": _huawei25_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
