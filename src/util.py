import inspect
import logging
import math
import os
import sys
import time
from functools import wraps
from typing import Callable, Union, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from matplotlib import pyplot as plt


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    for name in [
        "distributed.worker",
        "distributed.client",
        "distributed.scheduler",
        "distributed.nanny",
        "distributed.core",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)


# -- decorators


def log_timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(
            f"{time.ctime()} - Function '{func.__name__}' executed in {elapsed:.2f} seconds."
        )
        return result

    return wrapper


def dask_client(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        client = Client(local_directory="temp")
        logging.info(f"Dask dashboard: {client.dashboard_link}")
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame is not None else "unknown"
            logging.error(
                f"Error in function '{func_name}': {e}",
                exc_info=True,
            )
            raise
        finally:
            client.close()
            logging.info("Dask client closed.")

    return wrapper


# -- utils for preprocessing


def replace_regex_with_nan(
        col: pd.Series, regex: str = "", ignore_nan: bool = True
) -> pd.Series:
    mask = col.notna()
    found = col[mask].astype(str).str.contains(regex, regex=True)
    if found.any():
        logging.warning(
            f"Found values matching pattern {regex} in column: {col.name}, replacing them with NaN"
        )
        logging.warning(f"Replacing values: {col[mask][found].unique()}")
    r = col.copy()
    r[mask][found] = pd.NA
    return r


def check_column_for_negative_values(
        column: pd.Series, col_name: str, file_name: str = ""
) -> list[int]:
    negatives = column[column < 0]
    if not negatives.empty:
        for idx, value in negatives.items():
            logging.warning(
                f"Negative value found in '{file_name}' column '{col_name}' at index {idx}: {value}"
            )
    return negatives.index.tolist()


def check_continuity(
        df: pd.DataFrame, column: str, step: int, file_name: str = ""
) -> pd.Index:
    # Check if the values in the specified column are continuous with a given step size between each
    diffs = df[column].diff()
    # Skip the first row (index 0) which is always NaN
    discontinuities = diffs != step
    if not discontinuities.empty:
        discontinuities.iloc[0] = False  # Ignore the first row
    indices = df[discontinuities].index
    for idx in indices:
        logging.warning(
            f"Discontinuity found in '{file_name}' column '{column}' at index {idx}: {df.at[idx, column]}"
        )
    return indices


# -- filesystem utils


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def to_csv_safe(df: pd.DataFrame, path: str, *args, **kwargs) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, *args, **kwargs)


def to_csv_safe_dask(df: dd.DataFrame, path: str, **kwargs) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False, single_file=True, **kwargs)


def savefig_safe(path: str, *args, **kwargs) -> None:
    ensure_parent_dir(path)
    plt.savefig(path, *args, **kwargs)
    plt.close()
    logging.info(f"Figure saved to '{path}'")


# -- utils for plotting


def get_dataset_style(dataset_name: str) -> dict:
    name = dataset_name.lower()

    colors = {
        "azure": "#0d49fb",  # Blue
        "huawei25": "#e6091c",  # Red
        "huawei23": "#26eb47",  # Green
        "ibm": "#8936df",  # Purple
        "globus": "#fec32d",  # Yellow
        "alibaba": "#c20078",  # Dark Teal
        "default": "#333333",  # Dark Grey
    }

    style = {}
    if "azure" in name:
        style["color"] = colors["azure"]
        if "19" in name:
            style["linestyle"] = "-"
        elif "20" in name:
            style["linestyle"] = "-."
        elif "21" in name:
            style["linestyle"] = "--"
    elif "huawei25" in name:
        style["color"] = colors["huawei25"]
        style["linestyle"] = "-"
    elif "huawei23" in name:
        style["color"] = colors["huawei23"]
        style["linestyle"] = "--"
    elif "ibm" in name:
        style["color"] = colors["ibm"]
        style["linestyle"] = "-"
    elif "globus" in name:
        style["color"] = colors["globus"]
        style["linestyle"] = "--"
    elif "alibaba" in name:
        style["color"] = colors["alibaba"]
        style["linestyle"] = ":"
    else:
        style["color"] = colors["default"]
        style["linestyle"] = "-"

    return style


def set_nice_yticks(ax, max_tick: float = None, target_ticks: int = 4) -> None:
    if max_tick is None:
        max_tick = ax.get_ylim()[1]
    if max_tick <= 0:
        ax.set_yticks([0, 1])
        ax.set_ylim(0, 1)
        return

    raw_step = max_tick / target_ticks
    mag = 10 ** math.floor(math.log10(raw_step))
    norm_step = raw_step / mag

    step = (
        1.0 * mag
        if norm_step <= 1.5
        else (
            2.0 * mag
            if norm_step <= 3.0
            else 5.0 * mag if norm_step <= 7.0 else 10.0 * mag
        )
    )

    upper_bound = math.ceil(max_tick / step) * step
    yticks = np.round(np.arange(0, upper_bound + step / 1000, step), 10)

    ax.set_yticks(yticks)
    ax.set_ylim(0, yticks[-1])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${int(x)}$" if x.is_integer() else f"${x:g}$")
    )


def group_by_hour(df: dd.DataFrame, action: str) -> dd.DataFrame:
    """
    Group the DataFrame by hour (3600 seconds) and perform the specified action.
    Assumes the df is in format ["time", <value_column>].
    """

    def do_sum(df: dd.DataFrame) -> dd.DataFrame:
        df["time"] = df["time"] // 3600
        df = df.groupby("time").sum().reset_index()
        return df

    def do_mean(df: dd.DataFrame) -> dd.DataFrame:
        df["time"] = df["time"] // 3600
        df = df.groupby("time").mean().reset_index()
        return df

    actions = {
        "sum": do_sum,
        "mean": do_mean,
    }

    if action not in actions:
        logging.warning(
            f"No valid action '{action}' given to group_by_hour. Returning original DataFrame."
        )
    result = actions.get(action, lambda df: df)(df)
    return result


def create_time_axis_format(
        steps_in: str, multiplier: int = 1
) -> tuple[int, Callable, str]:
    """
    Create a time axis format for the given total time in seconds.

    Args:
        steps_in (str): The unit of time for the x-axis (e.g., "days", "hours", "minutes", "seconds").
        multiplier (int): A multiplier to adjust the tick step.

    Returns:
        tuple[int, Callable]: A tuple containing the tick step, label formatting function and the label.
    """
    if steps_in == "days":
        # Daily intervals
        tick_step = multiplier * 86400
        label_fmt = lambda x, _: f"${int(x // 86400)}$"
    elif steps_in == "hours":
        # Hourly intervals
        tick_step = multiplier * 3600
        label_fmt = lambda x, _: f"${int(x // 3600)}$"
    elif steps_in == "minutes":
        # Minute intervals
        tick_step = multiplier * 60
        label_fmt = lambda x, _: f"${int(x // 60)}$"
    else:
        # Seconds
        tick_step = 1
        label_fmt = lambda x, _: f"${int(x)}$"

    label = f"Time (in {'days' if steps_in == 'days' else 'hours' if steps_in == 'hours' else 'minutes' if steps_in == 'minutes' else 'seconds'})"
    return tick_step, label_fmt, label


def scale_column_by_average(
        df: dd.DataFrame, col: str, return_array: bool = False
) -> dd.DataFrame | np.ndarray:
    """
    Scale a column by its average value.

    Args:
        df: Dask DataFrame
        col: Column name to scale
        return_array: If True, return scaled_values as numpy array
                      If False, return modified DataFrame (default behavior)
    """
    average = df[col].mean().compute()
    if average == 0:
        raise ValueError(f"Average of column '{col}' is zero, cannot scale.")

    if return_array:
        scaled_values = (df[col] / average).compute().values
        print(f"Column '{col}' scaled by its average value: {average}")
        return scaled_values
    else:
        df[col] = df[col] / average
        print(f"Column '{col}' scaled by its average value: {average}")
        return df


# -- utils for statistics


def calc_stats(
        data_map: dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
        output_csv_path: str,
):
    combined_metrics = {}

    for name, data_item in data_map.items():
        if isinstance(data_item, tuple):
            data, weights = data_item
            data = np.asarray(data, dtype=float)
            weights = np.asarray(weights, dtype=float)
            mask = ~np.isnan(data) & (data >= 0) & ~np.isnan(weights) & (weights > 0)
            data, weights = data[mask], weights[mask]
            if len(data) == 0:
                continue

            mean = np.average(data, weights=weights)
            var = np.average((data - mean) ** 2, weights=weights)
            std_dev = float(np.sqrt(var))
            cv = (std_dev / mean) if mean != 0 else 0.0

            sorter = np.argsort(data)
            d_sort, w_sort = data[sorter], weights[sorter]
            cumsum = np.cumsum(w_sort)
            total = cumsum[-1]
            quantiles = np.interp([0.5, 0.9, 0.95, 0.99], cumsum / total, d_sort)
            median, p90, p95, p99 = quantiles
            min_val, max_val = d_sort[0], d_sort[-1]
            sample_count = int(len(data))
        else:
            data = np.asarray(data_item, dtype=float)
            data = data[~np.isnan(data) & (data >= 0)]
            if len(data) == 0:
                continue

            mean = float(np.mean(data))
            # ddof=1 is undefined for n=1; set spread metrics to 0.
            if len(data) > 1:
                std_dev = float(np.std(data, ddof=1))
            else:
                std_dev = 0.0
            cv = (std_dev / mean) if mean != 0 else 0.0
            median, p90, p95, p99 = np.percentile(data, [50, 90, 95, 99])
            min_val, max_val = np.min(data), np.max(data)
            sample_count = int(len(data))

        name_formatted = name.title().replace("Ibm", "IBM")
        combined_metrics[name_formatted] = {
            "Count": sample_count,
            "Mean": mean,
            "Median": median,
            "Std Dev": std_dev,
            "CV": cv,
            "P90": p90,
            "P95": p95,
            "P99": p99,
            "Min": min_val,
            "Max": max_val,
        }

    df = pd.DataFrame.from_dict(combined_metrics, orient="index")
    df.index.name = "Dataset"
    df.reset_index(inplace=True)
    to_csv_safe(df, output_csv_path)
    logging.info(f"Statistics successfully calculated and saved to '{output_csv_path}'")


def compute_weighted_boxplot_statistics(df: dd.DataFrame, col):
    weighted_sum = (df[col] * df["count"]).sum()
    total_count = df["count"].sum()
    sorted_df = df[[col, "count"]].sort_values(col).reset_index(drop=True)

    weighted_sum, total_count, sorted_df = dd.compute(
        weighted_sum, total_count, sorted_df
    )

    vals = sorted_df[col].to_numpy()
    weights = sorted_df["count"].to_numpy().astype(np.int64)

    cumsum = np.cumsum(weights)
    norm_cumsum = cumsum / total_count

    q1 = vals[np.searchsorted(norm_cumsum, 0.25)]
    med = vals[np.searchsorted(norm_cumsum, 0.50)]
    q3 = vals[np.searchsorted(norm_cumsum, 0.75)]

    return {
        "q1": q1,
        "med": med,
        "q3": q3,
        "whislo": vals[0],
        "whishi": vals[-1],
        "mean": weighted_sum / total_count,
    }


def compute_boxplot_statistics(df: dd.DataFrame, col):
    mean, min_val, q1, med, q3, max_val = dd.compute(
        df[col].mean(),
        df[col].min(),
        df[col].quantile(0.25),
        df[col].quantile(0.50),
        df[col].quantile(0.75),
        df[col].max(),
    )

    return {
        "q1": q1,
        "med": med,
        "q3": q3,
        "whislo": min_val,
        "whishi": max_val,
        "mean": mean,
    }
