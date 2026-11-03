import concurrent.futures
import glob
import logging
import warnings

import pandas as pd

from src.local import data_path, raw_data_path
from src.util import (
    check_continuity,
    check_column_for_negative_values,
    log_timing,
    replace_regex_with_nan,
    to_csv_safe,
)

huawei25_path: str = f"{data_path}/huawei25"

logging.basicConfig(level=logging.INFO)


def _process_cold_start_file(file: str) -> tuple[int, int]:
    df = pd.read_csv(file)
    raw_rows = len(df)
    df.drop_duplicates(inplace=True)
    filename = file.replace(f"{raw_data_path}/", "")
    print(f"Processing {filename}")

    if df.columns.duplicated().any():
        logging.warning(
            f"Duplicate column names found in {filename}: {df.columns[df.columns.duplicated()].tolist()}"
        )
    cols = [c for c in df.columns if c not in ["userID", "requestID", "podID"]]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    for col in cols:
        negs = check_column_for_negative_values(df[col], col, filename)
        if negs:
            df = df.drop(index=negs)

    cols.remove("day")
    cols.remove("deployDependencyCost")
    df = df.reset_index(drop=True)
    final_rows = len(df)
    output_file = file.replace(f"{raw_data_path}/huawei25", huawei25_path)
    to_csv_safe(df, output_file)
    return raw_rows, final_rows


@log_timing
def _process_day30_file(file: str) -> tuple[int, int]:
    df = pd.read_csv(file)
    raw_rows = len(df)
    df.drop_duplicates(inplace=True)
    filename = file.replace(f"{raw_data_path}/", "")

    non_numeric = ["requestID", "podID"]
    numeric_cols = [c for c in df.columns if c not in non_numeric]
    df[numeric_cols] = df[numeric_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    )
    df = df.dropna(subset=["time_worker", "podID"])

    df = df.reset_index(drop=True)
    final_rows = len(df)
    output_file = file.replace(f"{raw_data_path}/huawei25", huawei25_path)
    to_csv_safe(df, output_file)
    logging.info(f"Finished processing {filename}")
    return raw_rows, final_rows


def _process_time_series_usage_file(file: str) -> tuple[int, int]:
    exclude = ["day", "time"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = pd.read_csv(file)
        raw_rows = len(df)
        df.drop_duplicates(inplace=True)
        for warning in w:
            logging.warning(f"Warning in file {file}: {warning.message}")
    filename = file.replace(f"{raw_data_path}/", "")
    print(f"Processing {filename}")

    for col in ["day", "time"]:
        negs = check_column_for_negative_values(df[col], col, filename)
        if negs:
            df = df.drop(index=negs)
    check_continuity(df, "time", 60, filename)
    for col in [c for c in df.columns if c not in exclude]:
        df[col] = replace_regex_with_nan(df[col], r"[^\d\.]")
        df[col] = df[col].apply(lambda s: pd.to_numeric(s, errors="raise"))
        negs = check_column_for_negative_values(df[col], col, filename)
        if negs:
            df = df.drop(index=negs)
    final_rows = len(df)
    output_file = file.replace(f"{raw_data_path}/huawei25", huawei25_path)
    to_csv_safe(df, output_file)
    return raw_rows, final_rows


def _process_time_series_file(file: str) -> tuple[int, int]:
    df = pd.read_csv(file)
    raw_rows = len(df)
    df.drop_duplicates(inplace=True)
    filename = file.replace(f"{raw_data_path}/", "")
    print(f"Processing {filename}")

    if df.columns.duplicated().any():
        logging.warning(
            f"Duplicate column names found in {filename}: {df.columns[df.columns.duplicated()].tolist()}"
        )
    for col in df.columns:
        negs = check_column_for_negative_values(df[col], col, filename)
        if negs:
            df = df.drop(index=negs)
    check_continuity(df, "time", 60, filename)
    df = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    df = df.reset_index(drop=True)
    final_rows = len(df)
    output_file = file.replace(f"{raw_data_path}/huawei25", huawei25_path)
    to_csv_safe(df, output_file)
    return raw_rows, final_rows


def cold_starts() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(glob.glob(f"{raw_data_path}/huawei25/cold_starts/R{r}/*.csv"))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_cold_start_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {cold_starts.__name__}")
    return total_raw, total_final


def day30() -> tuple[int, int]:
    file_list = []
    # we only analyze R5
    for r in range(5, 6):
        file_list.extend(glob.glob(f"{raw_data_path}/huawei25/day30/R{r}/*.csv"))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_day30_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {day30.__name__}")
    return total_raw, total_final


def trigger_types() -> tuple[int, int]:
    pattern = (
        f"{raw_data_path}/huawei25/trigger_types/R2/df_funcID_runtime_triggerType.csv"
    )
    files = glob.glob(pattern)
    file = files[0]
    df = pd.read_csv(file)
    raw_rows = len(df)
    df.drop_duplicates(inplace=True)
    output_file = file.replace(f"{raw_data_path}/huawei25", huawei25_path)
    to_csv_safe(df, output_file)
    final_rows = len(df)
    logging.info(f"Finished processing {trigger_types.__name__}")
    return raw_rows, final_rows


def time_series_cpu_usage() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(f"{raw_data_path}/huawei25/time_series/R{r}/cpu_usage_avg/*.csv")
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_usage_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_cpu_usage.__name__}")
    return total_raw, total_final


def time_series_memory_usage() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(
                f"{raw_data_path}/huawei25/time_series/R{r}/memory_usage_avg/*.csv"
            )
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_usage_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_memory_usage.__name__}")
    return total_raw, total_final


def time_series_requests() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(f"{raw_data_path}/huawei25/time_series/R{r}/requests/*.csv")
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_requests.__name__}")
    return total_raw, total_final


def time_series_num_pods() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(f"{raw_data_path}/huawei25/time_series/R{r}/num_pods/*.csv")
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_num_pods.__name__}")
    return total_raw, total_final


def time_series_total_cost() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(f"{raw_data_path}/huawei25/time_series/R{r}/totalCost_avg/*.csv")
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_total_cost.__name__}")
    return total_raw, total_final


def time_series_total_cost_frontend() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(
                f"{raw_data_path}/huawei25/time_series/R{r}/totalCost_frontend_avg/*.csv"
            )
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_total_cost_frontend.__name__}")
    return total_raw, total_final


def time_series_request_body_size() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(
                f"{raw_data_path}/huawei25/time_series/R{r}/requestBodySize_avg/*.csv"
            )
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_request_body_size.__name__}")
    return total_raw, total_final


def time_series_read_body_cost() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(
                f"{raw_data_path}/huawei25/time_series/R{r}/readBodyCost_avg/*.csv"
            )
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_read_body_cost.__name__}")
    return total_raw, total_final


def time_series_write_rsp_cost() -> tuple[int, int]:
    file_list = []
    for r in range(1, 6):
        file_list.extend(
            glob.glob(
                f"{raw_data_path}/huawei25/time_series/R{r}/writeRspCost_avg/*.csv"
            )
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_time_series_file, sorted(file_list)))
        total_raw = sum(r[0] for r in results)
        total_final = sum(r[1] for r in results)

    logging.info(f"Finished processing {time_series_write_rsp_cost.__name__}")
    return total_raw, total_final


if __name__ == "__main__":
    total_raw = 0
    total_final = 0

    funcs = [
        cold_starts,
        day30,
        trigger_types,
        time_series_cpu_usage,
        time_series_memory_usage,
        time_series_requests,
        time_series_num_pods,
        time_series_total_cost,
        time_series_total_cost_frontend,
        time_series_request_body_size,
        time_series_read_body_cost,
        time_series_write_rsp_cost,
    ]

    for func in funcs:
        raw, final = func()
        total_raw += raw
        total_final += final

    logging.info(
        f"Huawei (Huawei25): Raw rows = {total_raw}, Final rows = {total_final}"
    )
