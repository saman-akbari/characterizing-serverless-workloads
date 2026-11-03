import gc
import glob
import logging

import dask.dataframe as dd
import pandas as pd

from src.local import data_path, raw_data_path
from src.util import to_csv_safe_dask

logging.basicConfig(level=logging.INFO)


def invocations_per_function() -> tuple[int, int]:
    file_list = glob.glob(
        f"{raw_data_path}/azure19/invocations_per_function_md.anon.*.csv"
    )
    total_raw_rows = 0
    total_final_rows = 0

    for file in file_list:
        filename = file.replace(f"{raw_data_path}/", "")
        logging.info(f"Processing file: {filename}")

        df = dd.read_csv(file, encoding="utf-8", dtype=object)
        total_raw_rows += len(df)

        def _count_negatives(pdf):
            numeric_part = pdf.iloc[:, 4:].apply(pd.to_numeric, errors="coerce")
            return (numeric_part < 0).sum().to_frame().T

        neg_stats = df.map_partitions(_count_negatives).sum().compute()

        for col_name, count in neg_stats.items():
            if count > 0:
                logging.warning(
                    f"Column {col_name} in {filename} contains {count} negative values."
                )

        if df.columns.duplicated().any():
            logging.warning(
                f"Duplicate column names found in {filename}: {df.columns[df.columns.duplicated()].tolist()}"
            )

        duplicates = df.map_partitions(lambda x: x[x.duplicated()]).compute()
        if not duplicates.empty:
            logging.warning(
                f"Duplicate rows found in {filename}: {duplicates.to_dict(orient='records')}"
            )

        df = df.drop_duplicates()

        meta = df._meta.copy()
        for col in meta.columns[4:]:
            meta[col] = pd.Series([], dtype="float64")

        def _clean(pdf):
            cols = pdf.columns[4:]
            numeric = pdf[cols].apply(pd.to_numeric, errors="coerce")
            mask = (numeric < 0).any(axis=1)
            if mask.any():
                pdf = pdf[~mask].copy()
            pdf[cols] = pdf[cols].apply(pd.to_numeric, errors="raise")
            return pdf

        df = df.map_partitions(_clean, meta=meta)
        total_final_rows += len(df)

        output_dir = file.replace(raw_data_path, data_path)
        to_csv_safe_dask(df, output_dir)

        logging.info(
            f"Finished processing and saved {len(df)} rows in cleaned file: {output_dir}"
        )

        del df, neg_stats, duplicates
        gc.collect()
    return total_raw_rows, total_final_rows


def memory_percentiles() -> tuple[int, int]:
    file_list = glob.glob(f"{raw_data_path}/azure19/app_memory_percentiles.anon.*.csv")
    total_raw_rows = 0
    total_final_rows = 0

    for file in file_list:
        filename = file.replace(f"{raw_data_path}/", "")
        logging.info(f"Processing file: {filename}")

        df = dd.read_csv(file, encoding="utf-8", dtype=object)
        total_raw_rows += len(df)

        def _count_dropped(pdf):
            cols = pdf.columns[2:]
            view = pdf[cols].apply(pd.to_numeric, errors="coerce")
            return (view < 0).any(axis=1).sum()

        n_dropped = df.map_partitions(_count_dropped).sum().compute()

        if n_dropped > 0:
            logging.info(
                f"Dropping {n_dropped} rows with negative values in {filename}"
            )

        meta = df._meta.copy()
        for col in meta.columns[2:]:
            meta[col] = pd.Series([], dtype="float64")

        def _clean(pdf):
            cols = pdf.columns[2:]
            view = pdf[cols].apply(pd.to_numeric, errors="coerce")
            mask = (view < 0).any(axis=1)
            if mask.any():
                pdf = pdf[~mask].copy()
            pdf[cols] = pdf[cols].apply(pd.to_numeric, errors="raise")
            return pdf

        df = df.map_partitions(_clean, meta=meta)
        total_final_rows += len(df)

        output_dir = file.replace(raw_data_path, data_path)
        to_csv_safe_dask(df, output_dir)

        logging.info(
            f"Finished processing and saved {len(df)} rows in cleaned file: {output_dir}"
        )

        del df
        gc.collect()
    return total_raw_rows, total_final_rows


def function_durations() -> tuple[int, int]:
    file_list = glob.glob(
        f"{raw_data_path}/azure19/function_durations_percentiles.anon.*.csv"
    )
    total_raw_rows = 0
    total_final_rows = 0

    for file in file_list:
        filename = file.replace(f"{raw_data_path}/", "")
        logging.info(f"Processing file: {filename}")

        df = dd.read_csv(file, encoding="utf-8", dtype=object)
        total_raw_rows += len(df)

        def _count_dropped(pdf):
            cols = pdf.columns[3:]
            view = pdf[cols].apply(pd.to_numeric, errors="coerce")
            return (view < 0).any(axis=1).sum()

        n_dropped = df.map_partitions(_count_dropped).sum().compute()

        if n_dropped > 0:
            logging.info(
                f"Dropping {n_dropped} rows with negative values in {filename}"
            )

        meta = df._meta.copy()
        for col in meta.columns[3:]:
            meta[col] = pd.Series([], dtype="float64")

        def _clean(pdf):
            cols = pdf.columns[3:]
            view = pdf[cols].apply(pd.to_numeric, errors="coerce")
            mask = (view < 0).any(axis=1)
            if mask.any():
                pdf = pdf[~mask].copy()
            pdf[cols] = pdf[cols].apply(pd.to_numeric, errors="raise")
            return pdf

        df = df.map_partitions(_clean, meta=meta)
        total_final_rows += len(df)

        output_dir = file.replace(raw_data_path, data_path)
        to_csv_safe_dask(df, output_dir)

        logging.info(
            f"Finished processing and saved {len(df)} rows in cleaned file: {output_dir}"
        )

        del df
        gc.collect()
    return total_raw_rows, total_final_rows


if __name__ == "__main__":
    total_raw = 0
    total_final = 0

    raw, final = invocations_per_function()
    total_raw += raw
    total_final += final

    raw, final = function_durations()
    total_raw += raw
    total_final += final

    raw, final = memory_percentiles()
    total_raw += raw
    total_final += final

    logging.info(
        f"AzureFirst (Azure19): Raw rows = {total_raw}, Final rows = {total_final}"
    )
