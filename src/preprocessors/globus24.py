import glob
import logging

import pandas as pd

from src.local import data_path, raw_data_path
from src.util import check_column_for_negative_values, to_csv_safe

logging.basicConfig(level=logging.INFO)


def endpoints() -> tuple[int, int]:
    filepattern = f"{raw_data_path}/globus24/GlobusComputeDataset-v1-endpoints.csv"
    df = pd.read_csv(filepattern, encoding="utf-8")
    raw_rows = len(df)

    filename = filepattern.replace(f"{raw_data_path}/", "")
    # Check for duplicate rows
    if df.duplicated().any():
        logging.warning(
            f"Duplicate rows found in {filename}: {df[df.duplicated()].to_dict(orient='records')}"
        )
        df.drop_duplicates(inplace=True)
    final_rows = len(df)

    output_file = filepattern.replace(raw_data_path, data_path)
    to_csv_safe(df, output_file)
    logging.info(f"Finished processing endpoints, saved cleaned file: {output_file}")
    return raw_rows, final_rows


def functions() -> tuple[int, int]:
    filepattern = f"{raw_data_path}/globus24/GlobusComputeDataset-v1-functions.csv"
    df = pd.read_csv(filepattern, encoding="utf-8")
    raw_rows = len(df)

    if df.duplicated().any():
        df.drop_duplicates(inplace=True)

    number_cols = ["loc", "cyc_complexity", "num_of_imports"]

    # Check that number_cols contain only floats or missing (NaN)
    for col in number_cols:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    final_rows = len(df)

    output_file = filepattern.replace(raw_data_path, data_path)
    to_csv_safe(df, output_file)
    logging.info(f"Finished processing functions, saved cleaned file: {output_file}")
    return raw_rows, final_rows


def tasks() -> tuple[int, int]:
    file_list = glob.glob(f"{raw_data_path}/globus24/GlobusComputeDataset-v1-tasks.csv")
    total_raw_rows = 0
    total_final_rows = 0

    reference_ts_ns = pd.Timestamp(
        "2022-11-28 00:00:00", tz="UTC"
    ).value  # nanoseconds since epoch

    for file in file_list:
        df = pd.read_csv(file, encoding="utf-8")
        total_raw_rows += len(df)
        filename = file.replace(f"{raw_data_path}/", "")

        # Check for duplicate rows
        if df.duplicated().any():
            logging.warning(
                f"Duplicate rows found in {filename}: {df[df.duplicated()].to_dict(orient='records')}"
            )
            df.drop_duplicates(inplace=True)

        # Argument size column (5th col, index 4)
        arg_col = df.columns[4]
        negs = check_column_for_negative_values(df[arg_col], arg_col, filename)
        if negs:
            df = df.drop(index=negs)
        df[arg_col] = pd.to_numeric(df[arg_col], errors="coerce")

        # Timestamp columns (all columns from index 5 onward)
        for col in df.columns[5:]:
            # Convert to float (nanoseconds), subtract reference, then to seconds
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = (df[col] - reference_ts_ns) / 1e9
            negs = check_column_for_negative_values(df[col], col, filename)
            if negs:
                df = df.drop(index=negs)

        df = df.reset_index(drop=True)
        total_final_rows += len(df)

        # Save cleaned file
        output_file = file.replace(raw_data_path, data_path)
        to_csv_safe(df, output_file)
        logging.info(f"Finished processing tasks, saved cleaned file: {output_file}")

    return total_raw_rows, total_final_rows


if __name__ == "__main__":
    total_raw = 0
    total_final = 0

    raw, final = endpoints()
    total_raw += raw
    total_final += final

    raw, final = functions()
    total_raw += raw
    total_final += final

    raw, final = tasks()
    total_raw += raw
    total_final += final

    logging.info(
        f"Globus (Globus24): Raw rows = {total_raw}, Final rows = {total_final}"
    )
