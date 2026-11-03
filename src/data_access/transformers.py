import logging
import time
from typing import Callable

import pandas as pd


def _azure20_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transform azure20 data access dataset."""
    start_time = time.time()
    logging.info("Transforming azure20 dataset...")

    datetime_start = time.time()
    df["Datetime"] = pd.to_datetime(df["time"], unit="ms")
    logging.info(
        f"Datetime conversion completed (elapsed: {time.time() - datetime_start:.2f}s)"
    )

    logging.info(
        f"Azure20 transformation completed (total elapsed: {time.time() - start_time:.2f}s)"
    )
    return df


# Public factory -----------------------------------------------------------------


def data_access_transformer(dataset: str) -> Callable:
    logging.info(f"Calling transformer for dataset: {dataset}")
    mapping = {
        "azure20": _azure20_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
