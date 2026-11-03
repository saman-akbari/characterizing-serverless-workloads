import glob
import logging

import pandas as pd

from src.local import data_path, raw_data_path
from src.util import check_column_for_negative_values, to_csv_safe

logging.basicConfig(level=logging.INFO)


def alibaba21() -> None:
    file_list = glob.glob(f"{raw_data_path}/alibaba21/*.csv")
    total_raw_rows = 0
    total_final_rows = 0

    # dataset starts at 2021-05-08 00:00:00 UTC, which was a saturday
    start_ts = pd.Timestamp("2021-05-08 00:00:00", tz="UTC")
    day_of_week_offset = start_ts.dayofweek
    seconds_offset = day_of_week_offset * 24 * 3600
    start_unix_seconds = int(start_ts.value / 1e9)

    for file in file_list:
        df = pd.read_csv(file, encoding="utf-8")
        total_raw_rows += len(df)
        filename = file.replace(f"{raw_data_path}/", "")

        # Drop the extra index column if present
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Check for duplicate columns
        if df.columns.duplicated().any():
            logging.warning(
                f"Duplicate column names found in {filename}: {df.columns[df.columns.duplicated()].tolist()}"
            )

        # Check for duplicate rows
        if df.duplicated().any():
            logging.warning(
                f"Duplicate rows found in {filename}: {df[df.duplicated()].to_dict(orient='records')}"
            )
            df.drop_duplicates(inplace=True)

        # Convert original timestamps to numeric values
        df["__time__"] = pd.to_numeric(df["__time__"], errors="coerce")
        df["__time__"] = (df["__time__"] - start_unix_seconds) + seconds_offset

        # Check time continuity
        time_diff = pd.to_numeric(df["__time__"].diff(), errors="coerce")
        if not (time_diff >= 0).all():
            discontinuities = df.index[time_diff < 0].tolist()
            for idx in discontinuities:
                logging.warning(
                    f"Time discontinuity found in {filename} at index {idx}: {df.loc[idx, '__time__']} < {df.loc[idx - 1, '__time__']}"
                )
        df = df.rename(columns={"__time__": "time"})

        # Check latency for negative values and cast to float
        negs = check_column_for_negative_values(df["latency"], "latency", filename)
        if negs:
            df = df.drop(index=negs)
        df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
        df["latency"] = df["latency"].astype(float, errors="raise")

        # latency is stated to be in s but is clearly in ms; convert ms to s
        df["latency"] = df["latency"] / 1000.0

        # Check memoryMB for negative values and cast to int
        negs = check_column_for_negative_values(df["memoryMB"], "memoryMB", filename)
        if negs:
            df = df.drop(index=negs)
        df["memoryMB"] = pd.to_numeric(df["memoryMB"], errors="coerce")
        df["memoryMB"] = df["memoryMB"].astype(int, errors="raise")

        # Ensure runtime and functionName are strings
        df["runtime"] = df["runtime"].astype(str)
        df["functionName"] = df["functionName"].astype(str)

        df = df.reset_index(drop=True)
        total_final_rows += len(df)

        # Save cleaned file
        output_file = file.replace(raw_data_path, data_path)
        to_csv_safe(df, output_file)

        logging.info(f"Finished processing and saved cleaned file: {output_file}")

    logging.info(
        f"Alibaba21: Raw rows = {total_raw_rows}, Final rows = {total_final_rows}"
    )


if __name__ == "__main__":
    alibaba21()
