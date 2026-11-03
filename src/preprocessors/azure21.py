import glob
import logging

import pandas as pd

from src.local import data_path, raw_data_path
from src.util import check_column_for_negative_values, to_csv_safe

logging.basicConfig(level=logging.INFO)


def functions_invocations_trace_for_two_weeks_jan2021() -> None:
    file_list = glob.glob(
        f"{raw_data_path}/azure21/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    )
    total_raw_rows = 0
    total_final_rows = 0

    # The trace starts on 2021-01-31 (Sunday)
    day_of_week_offset = 6
    seconds_offset = day_of_week_offset * 24 * 3600

    for file in file_list:
        df = pd.read_csv(file, encoding="utf-8")
        total_raw_rows += len(df)

        filename = file.replace(f"{raw_data_path}/", "")
        logging.info(f"Processing file: {filename}, length: {len(df)}")

        dupe = df.duplicated()
        if dupe.any():
            logging.warning(f"Duplicate rows found in {filename}")
            logging.warning(f"Total duplicate rows: {dupe.sum()} out of {len(df)}")
            df = df[~dupe].reset_index(drop=True)

        # Shift the timestamps so that the start of the trace (0) aligns with Sunday (the 6th day of the week)
        df["end_timestamp"] = (
                pd.to_numeric(df["end_timestamp"], errors="coerce") + seconds_offset
        )

        for col in df.columns[2:]:
            negs = check_column_for_negative_values(df[col], col, filename)
            if negs:
                df = df.drop(index=negs)

        for col in df.columns[2:]:
            df[col] = df[col].apply(lambda s: pd.to_numeric(s, errors="raise"))
        df = df.reset_index(drop=True)

        total_final_rows += len(df)

        output_file = file.replace(raw_data_path, data_path)
        to_csv_safe(df, output_file)
        logging.info(
            f"Finished processing and saved {len(df)} rows in cleaned file: {output_file}"
        )

    logging.info(
        f"AzureSecond (Azure21): Raw rows = {total_raw_rows}, Final rows = {total_final_rows}"
    )


if __name__ == "__main__":
    functions_invocations_trace_for_two_weeks_jan2021()
