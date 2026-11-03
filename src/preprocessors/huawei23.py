import glob
import logging

import pandas as pd

from src.local import data_path, raw_data_path
from src.util import check_continuity, check_column_for_negative_values, to_csv_safe

logging.basicConfig(level=logging.INFO)


def _process_files(
        pattern: str,
        continuity_step: int,
        exclude: list[str] | None = None,
        check_day_time: bool = False,
        coerce_errors: bool = False,
) -> tuple[int, int]:
    file_list = glob.glob(pattern)
    total_raw = 0
    total_final = 0

    for file in file_list:
        df = pd.read_csv(file)
        total_raw += len(df)
        df.drop_duplicates(inplace=True)

        filename = file.replace(f"{data_path}/", "")

        if df.columns.duplicated().any():
            logging.warning(
                f"Duplicate column names found in {filename}: {df.columns[df.columns.duplicated()].tolist()}"
            )

        if check_day_time and (df["day"].isnull().any() or df["time"].isnull().any()):
            logging.warning(
                f"Missing values found in 'day' or 'time' columns in {filename}"
            )

        for col in df.columns:
            if exclude and col in exclude:
                continue
            negs = check_column_for_negative_values(df[col], col, filename)
            if negs:
                df = df.drop(index=negs)

        check_continuity(df, "time", continuity_step, filename)

        errors_mode = "coerce" if coerce_errors else "raise"
        df = df.apply(lambda s: pd.to_numeric(s, errors=errors_mode))

        df = df.reset_index(drop=True)
        total_final += len(df)

        output_file = file.replace(raw_data_path, data_path)
        to_csv_safe(df, output_file)

    return total_raw, total_final


def memory_usage_minute_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/memory_usage_minute/*.csv",
        continuity_step=60,
        exclude=["day", "time"],
        check_day_time=True,
    )
    logging.info(f"Finished processing {memory_usage_minute_private.__name__}")
    return raw, final


def memory_limit_minute_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/memory_limit_minute/*.csv",
        continuity_step=60,
        exclude=["day", "time"],
        check_day_time=True,
    )
    logging.info(f"Finished processing {memory_limit_minute_private.__name__}")
    return raw, final


def cpu_limit_minute_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/cpu_limit_minute/*.csv",
        continuity_step=60,
        exclude=["day", "time"],
        check_day_time=True,
    )
    logging.info(f"Finished processing {cpu_limit_minute_private.__name__}")
    return raw, final


def cpu_usage_minute_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/cpu_usage_minute/*.csv",
        continuity_step=60,
        exclude=["day", "time"],
        check_day_time=True,
    )
    logging.info(f"Finished processing {cpu_usage_minute_private.__name__}")
    return raw, final


def function_delay_minute_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/function_delay_minute/*.csv",
        continuity_step=60,
        coerce_errors=True,
    )
    logging.info(f"Finished processing {function_delay_minute_private.__name__}")
    return raw, final


def requests_minute_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/requests_minute/*.csv",
        continuity_step=60,
    )
    logging.info(f"Finished processing {requests_minute_private.__name__}")
    return raw, final


def requests_second_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/requests_second/*.csv",
        continuity_step=1,
    )
    logging.info(f"Finished processing {requests_second_private.__name__}")
    return raw, final


def platform_delay_minute_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/platform_delay_minute/*.csv",
        continuity_step=60,
    )
    logging.info(f"Finished processing {platform_delay_minute_private.__name__}")
    return raw, final


def instances_minute_private() -> tuple[int, int]:
    raw, final = _process_files(
        f"{raw_data_path}/huawei23/private_dataset/instances_minute/*.csv",
        continuity_step=60,
        exclude=["day", "time"],
        check_day_time=True,
    )
    logging.info(f"Finished processing {instances_minute_private.__name__}")
    return raw, final


if __name__ == "__main__":
    total_raw = 0
    total_final = 0

    funcs = [
        requests_minute_private,
        requests_second_private,
        function_delay_minute_private,
        memory_usage_minute_private,
        memory_limit_minute_private,
        cpu_usage_minute_private,
        cpu_limit_minute_private,
        platform_delay_minute_private,
        instances_minute_private,
    ]

    for func in funcs:
        raw, final = func()
        total_raw += raw
        total_final += final

    logging.info(
        f"Huawei (Huawei23): Raw rows = {total_raw}, Final rows = {total_final}"
    )
