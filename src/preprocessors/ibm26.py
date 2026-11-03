import gc
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.local import data_path, raw_data_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def check_and_convert_memory(series_gb_list, summary):
    converted = []
    for lst in series_gb_list:
        arr = np.array(lst, dtype=float)
        neg_count = np.sum(arr < 0)
        summary["memory_negatives"] += int(neg_count)

        arr[arr < 0] = np.nan
        arr = arr * 1024  # GB -> MB
        converted.append(arr.tolist())
    return pd.Series(converted)


def check_and_convert_cpu(series_list, summary):
    converted = []
    for lst in series_list:
        arr = np.array(lst, dtype=float)
        neg_count = np.sum(arr < 0)
        summary["cpu_negatives"] += int(neg_count)

        arr[arr < 0] = np.nan
        converted.append(arr.tolist())
    return pd.Series(converted)


def check_and_convert_int_list(series_list, summary, key):
    converted = []
    for lst in series_list:
        arr = np.array(lst, dtype=float)
        neg_count = np.sum(arr < 0)
        summary[key] += int(neg_count)

        arr[arr < 0] = 0
        arr = np.nan_to_num(arr, nan=0).astype(int)
        converted.append(arr.tolist())
    return pd.Series(converted)


def validate_list_lengths(row, expected_length, cols, summary):
    for col in cols:
        if len(row[col]) != expected_length:
            summary["length_mismatches"] += 1
            logging.warning(f"{col} length mismatch for app {row['AppHash']}")
            row[col] = row[col][:expected_length]
    return row


def clean_invocations_row(row):
    inv = np.array(row["InvocationTimes"], dtype=float)
    app = np.array(row["AppExecTimes"], dtype=float)
    tot = np.array(row["TotalExecTimes"], dtype=float)
    pod = np.array(row["PodHash"], dtype=object)

    invalid = (
            (inv < 0)
            | np.isnan(inv)
            | (app < 0)
            | np.isnan(app)
            | (tot < 0)
            | np.isnan(tot)
    )
    total_dropped = invalid.sum()

    valid_mask = ~invalid

    # unix timestamps start on thursday, add 3-day offset (259,200s)
    row["InvocationTimes"] = (inv[valid_mask] + 259200).tolist()

    row["AppExecTimes"] = app[valid_mask].tolist()
    row["TotalExecTimes"] = tot[valid_mask].tolist()
    row["PodHash"] = pod[valid_mask].tolist()
    row["NumEvents"] = len(row["InvocationTimes"])

    # temporarily store in row
    row["DroppedCount"] = int(total_dropped)

    return row


def preprocess_ibm26_config() -> tuple[int, int]:
    raw_path = Path(raw_data_path) / "ibm26"
    processed_path = Path(data_path) / "ibm26"
    processed_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Raw IBM26 config path: {raw_path}")
    logging.info(f"Processed IBM26 path: {processed_path}")

    app_configs_file = raw_path / "app_configs.pickle"
    if not app_configs_file.exists():
        logging.error(f"App configs file not found: {app_configs_file}")
        return 0, 0

    df = pd.read_pickle(app_configs_file)
    raw_rows = len(df)
    summary = {
        "total_apps": len(df),
        "memory_negatives": 0,
        "cpu_negatives": 0,
        "concurrency_negatives": 0,
        "minscale_negatives": 0,
        "length_mismatches": 0,
    }

    logging.info(f"Loaded {summary['total_apps']} application configurations")

    df["AppContainerRequestMemory_MB"] = check_and_convert_memory(
        df["AppContainerRequestMemory"], summary
    )
    df["AppContainerRequestCpu"] = check_and_convert_cpu(
        df["AppContainerRequestCpu"], summary
    )
    df["AppContainerConcurrency"] = check_and_convert_int_list(
        df["AppContainerConcurrency"], summary, "concurrency_negatives"
    )
    df["AppMinScale"] = check_and_convert_int_list(
        df["AppMinScale"], summary, "minscale_negatives"
    )

    df = df.apply(
        lambda row: validate_list_lengths(
            row,
            expected_length=62,
            cols=[
                "AppContainerRequestMemory",
                "AppContainerRequestCpu",
                "AppContainerConcurrency",
                "AppMinScale",
            ],
            summary=summary,
        ),
        axis=1,
    )

    final_rows = len(df)

    processed_file = processed_path / "app_configs_processed.pickle"
    processed_file.write_bytes(pickle.dumps(df))
    logging.info(f"Saved processed app configs to {processed_file}")

    # SUMMARY LOG
    logging.info(
        "\n--- IBM26 CONFIG PREPROCESSING SUMMARY ---\n"
        f"Total apps:                    {summary['total_apps']}\n"
        f"Negative memory values fixed:  {summary['memory_negatives']}\n"
        f"Negative CPU values fixed:     {summary['cpu_negatives']}\n"
        f"Negative concurrency values:   {summary['concurrency_negatives']}\n"
        f"Negative minscale values:      {summary['minscale_negatives']}\n"
        f"List-length mismatches fixed:  {summary['length_mismatches']}\n"
        f"Output file:                   {processed_file}\n"
        "-------------------------------------------"
    )

    return raw_rows, final_rows


def preprocess_ibm26_weekly() -> tuple[int, int]:
    raw_path = Path(raw_data_path) / "ibm26"
    processed_path = Path(data_path) / "ibm26"
    processed_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Raw IBM26 path: {raw_path}")
    logging.info(f"Processed IBM26 path: {processed_path}")

    week_files = sorted(raw_path.glob("week_*.pickle"))
    if not week_files:
        logging.error(f"No weekly files found in {raw_path}")
        return 0, 0

    total_raw_invocations = 0
    total_dropped_invocations = 0
    processed_files = []

    for f in week_files:
        df = pd.read_pickle(f)
        total_raw_invocations += df["InvocationTimes"].str.len().sum()

        logging.info(f"Loaded {len(df)} rows from {f.name}")
        df = df.apply(clean_invocations_row, axis=1)
        total_dropped = df["DroppedCount"].sum()
        total_dropped_invocations += total_dropped
        df = df.drop(columns=["DroppedCount"])

        processed_file = processed_path / f"{f.stem}_processed.pickle"
        processed_file.write_bytes(pickle.dumps(df))

        logging.info(f"Saved cleaned week data to {processed_file}")
        logging.info(f"Total dropped invocations in {f.name}: {total_dropped}")

        processed_files.append(str(processed_file))

        del df
        gc.collect()

    total_final_invocations = total_raw_invocations - total_dropped_invocations
    return total_raw_invocations, total_final_invocations


if __name__ == "__main__":
    preprocess_ibm26_config()

    weekly_raw_invocations, weekly_final_invocations = preprocess_ibm26_weekly()
    logging.info(
        f"IBM (IBM26): Raw rows = {weekly_raw_invocations}, Final rows = {weekly_final_invocations}"
    )
    logging.info("IBM26 preprocessing complete")
