import gc
import logging

import dask.dataframe as dd
import pandas as pd

from src.local import data_path, raw_data_path
from src.util import check_column_for_negative_values, to_csv_safe

logging.basicConfig(level=logging.INFO)


def preprocess_blob_accesses(blocksize: str = "128MB") -> None:
    file_path = f"{raw_data_path}/azure20/azurefunctions-accesses-2020.csv"
    filename = file_path.replace(f"{raw_data_path}/", "")
    logging.info("Processing file: %s", filename)

    df = dd.read_csv(
        file_path,
        dtype={
            "Timestamp": "float64",
            "BlobBytes": "float64",
            "Read": "object",
            "Write": "object",
        },
        blocksize=blocksize,
        assume_missing=True,
    ).compute()

    numeric_cols = ["Timestamp", "BlobBytes"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    dropped = before - len(df)
    if dropped:
        logging.info(
            "Dropped %d rows with missing numeric fields in %s", dropped, filename
        )

    df["Timestamp"] = df["Timestamp"].astype("Int64")
    df["BlobBytes"] = df["BlobBytes"].round().astype("Int64")

    # unix timestamps start on thursday, add 3-day offset (259,200s)
    df["Timestamp"] = (df["Timestamp"] // 1000) + 259200

    for col in numeric_cols:
        negs = check_column_for_negative_values(df[col], col, filename)
        if negs:
            df = df.drop(index=negs)

    bool_cols = ["Read", "Write"]
    for col in bool_cols:
        normalized = df[col].astype(str).str.strip().str.lower()
        invalid_mask = ~normalized.isin(["true", "false"]) & normalized.notna()
        if invalid_mask.any():
            logging.warning(
                "Found %d unexpected values in %s for %s",
                invalid_mask.sum(),
                col,
                filename,
            )
        df[col] = normalized.map({"true": True, "false": False}).astype("boolean")

    duplicate_count = df.duplicated().sum()
    if duplicate_count:
        logging.info("Dropping %d duplicate rows in %s", duplicate_count, filename)
        df = df.drop_duplicates()

    df = df.rename(columns={"Timestamp": "time"})
    df = df.sort_values("time").reset_index(drop=True)

    output_path = file_path.replace(raw_data_path, data_path)
    to_csv_safe(df, output_path)
    logging.info(
        "Finished processing and saved %d rows in cleaned file: %s",
        len(df),
        output_path,
    )

    del df
    gc.collect()


if __name__ == "__main__":
    preprocess_blob_accesses()
