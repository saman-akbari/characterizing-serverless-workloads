import logging
import time
from pathlib import Path

import pandas as pd

from src.local import data_path

logging.basicConfig(level=logging.INFO)


def _read_azure20() -> pd.DataFrame:
    start_time = time.time()
    logging.info("Reading azure20 dataset...")
    file_path = Path(data_path) / "azure20" / "azurefunctions-accesses-2020.csv"

    dtypes = {
        "AnonRegion": "category",
        "BlobType": "category",
        "Read": "bool",
        "Write": "bool",
        "BlobBytes": "float32",
    }

    df = pd.read_csv(file_path, dtype=dtypes)
    elapsed = time.time() - start_time
    logging.info(
        f"Azure20 dataset loaded: {len(df)} rows, {len(df.columns)} columns (elapsed: {elapsed:.2f}s)"
    )
    return df


def read_data(dataset: str) -> pd.DataFrame:
    """Factory function to read data for different datasets."""
    start_time = time.time()
    logging.info(f"Loading dataset: {dataset}")

    mapping = {
        "azure20": _read_azure20,
    }

    reader = mapping.get(dataset)
    if reader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    df = reader()
    logging.info(
        f"Dataset {dataset} loading completed (total elapsed: {time.time() - start_time:.2f}s)"
    )
    return df
