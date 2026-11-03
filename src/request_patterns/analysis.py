import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.util import (
    create_time_axis_format,
    group_by_hour,
    savefig_safe,
    to_csv_safe,
    set_nice_yticks,
    get_dataset_style,
)

plt.style.use("src/scientific.mplstyle")

DATASETS = {
    "azure19": "Azure19",
    "azure21": "Azure21",
    "globus24": "Globus24",
    "huawei23": "Huawei23",
    "huawei25": "Huawei25",
    "ibm26": "IBM26",
}


def save_arrival_time_rates(df: dd.DataFrame, dataset: str) -> None:
    df = df.compute().sort_values("time").reset_index(drop=True)
    df["time"] -= df["time"].min()
    to_csv_safe(df, f"results/request_patterns/time_series/csv/{dataset}.csv")


def plot_request_patterns_time_series(dataset: str) -> None:
    df = dd.read_csv(f"results/request_patterns/time_series/csv/{dataset}.csv")

    s: int = 60
    df["time_blocks"] = (df["time"] // s) * s
    df = df.groupby("time_blocks")["requests"].sum().reset_index()
    df = df.rename(columns={"time_blocks": "time"})

    df_pd = df.compute()
    actual_start = df_pd["time"].min()

    df_week = df_pd[
        (df_pd["time"] >= actual_start) & (df_pd["time"] < actual_start + 7 * 86400)
        ].copy()

    df_week["requests"] = df_week["requests"] / df_week["requests"].mean()

    df_week = df_week.sort_values("time").reset_index(drop=True)
    times = df_week["time"].to_numpy() - actual_start
    requests = df_week["requests"].to_numpy()

    plt.figure()
    plt.plot(times, requests)

    min_time = times[0]
    max_time = times[-1]
    duration_seconds = max_time - min_time

    if duration_seconds <= 86400:
        xaxis_unit = "hours"
        xmultiplier = 4
    elif duration_seconds > 10 * 86400:
        xaxis_unit = "days"
        xmultiplier = 2
    else:
        xaxis_unit = "days"
        xmultiplier = 1

    xtick_step, xlabel_fmt, xlabel = create_time_axis_format(xaxis_unit, xmultiplier)
    xticks = np.arange(min_time, max_time + xtick_step, xtick_step)
    plt.gca().set_xticks(xticks)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(xlabel_fmt))
    plt.xlabel(xlabel)

    plt.ylabel(f"Requests / Average")
    if dataset != "globus24":
        set_nice_yticks(plt.gca(), max(requests) * 1.05)
    else:
        plt.yscale("symlog", linthresh=10)
        plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.yticks([0, 5, 10, 100, 500])
        plt.ylim(0, 500)

    savefig_safe(f"results/request_patterns/time_series/{dataset}.pdf")

def average_requests_per_hour_csv(df: dd.DataFrame, dataset: str) -> None:
    # per hour
    df_hourly_base = group_by_hour(df, "sum")
    df_hourly_base["hour"] = (df_hourly_base["time"] % 24).astype(int)
    df_hourly_base["day_of_week"] = ((df_hourly_base["time"] // 24) % 7).astype(int)

    # overall
    df_all_days_agg = df_hourly_base.groupby("hour")["requests"].mean().reset_index()

    # weekdays (0-4)
    df_weekdays = df_hourly_base[df_hourly_base["day_of_week"] < 5]
    df_weekdays_agg = df_weekdays.groupby("hour")["requests"].mean().reset_index()

    # weekends (5-6)
    df_weekends = df_hourly_base[df_hourly_base["day_of_week"] >= 5]
    df_weekends_agg = df_weekends.groupby("hour")["requests"].mean().reset_index()

    df_all_pd, df_weekdays_pd, df_weekends_pd = dd.compute(
        df_all_days_agg, df_weekdays_agg, df_weekends_agg
    )

    to_csv_safe(
        df_all_pd,
        f"results/request_patterns/average_requests_per_hour/csv/{dataset}.csv",
    )
    to_csv_safe(
        df_weekdays_pd,
        f"results/request_patterns/average_requests_per_hour/csv/{dataset}_weekdays.csv",
    )
    to_csv_safe(
        df_weekends_pd,
        f"results/request_patterns/average_requests_per_hour/csv/{dataset}_weekends.csv",
    )


def plot_requests_per_hour_on_ax(
        ax, df_hourly: pd.DataFrame, pow10: float, max_y_scaled: float, title: str = ""
):
    scaled_requests = df_hourly["requests"] / pow10
    lowest_hours, avg_lowest, avg_other = find_lowest_8_hour_period(df_hourly)

    if lowest_hours is not None:
        bar_colors = [
            "blue" if hour in lowest_hours else "orange" for hour in df_hourly["hour"]
        ]
    else:
        bar_colors = "blue"

    ax.bar(df_hourly["hour"], scaled_requests, color=bar_colors)
    ax.set_xticks([0, 6, 12, 18, 24])
    set_nice_yticks(ax, max_y_scaled)
    ax.set_title(title)


def plot_requests_per_hour(dataset: str) -> None:
    suffixes = ["", "_weekdays", "_weekends"]
    titles = [
        f"Overall",
        f"Weekdays",
        f"Weekends",
    ]

    dataframes = []
    global_max = 0

    for suffix in suffixes:
        df = pd.read_csv(
            f"results/request_patterns/average_requests_per_hour/csv/{dataset}{suffix}.csv"
        )
        dataframes.append(df)
        global_max = max(global_max, df["requests"].max())

    if global_max > 0:
        pow10 = 10 ** int(np.floor(np.log10(global_max)) - 1)
    else:
        pow10 = 1.0

    scaled_global_max = global_max / pow10

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(5, 9))
    fig.supxlabel("Hour of Day")
    fig.supylabel(f"Requests ($\\times$ {int(pow10)})", x=-0.1)

    for i, df in enumerate(dataframes):
        if df is not None and not df.empty:
            plot_requests_per_hour_on_ax(
                ax=axes[i],
                df_hourly=df,
                pow10=pow10,
                max_y_scaled=scaled_global_max,
                title=titles[i],
            )

    savefig_safe(f"results/request_patterns/average_requests_per_hour/{dataset}.pdf")


def find_lowest_8_hour_period(
        df_hourly_pd: pd.DataFrame,
) -> tuple[list[int], float, float]:
    """
    Finds the continuous 8-hour period with the lowest total requests.
    Prints the period, average requests during this period, average requests during other hours, and their difference.
    Returns (lowest_hours, avg_lowest, avg_other).
    """
    requests = df_hourly_pd["requests"].to_numpy()
    hours = df_hourly_pd["hour"].to_numpy()
    window_size = 8
    min_sum = float("inf")
    min_start = 0

    # Since hours are 0-23, handle wrap-around for continuous periods
    extended_requests = np.concatenate([requests, requests[: window_size - 1]])
    for start in range(24):
        window_sum = extended_requests[start: start + window_size].sum()
        if window_sum < min_sum:
            min_sum = window_sum
            min_start = start

    # Get the hours in the lowest 8-hour period (0-indexed)
    lowest_hours = [(min_start + i) % 24 for i in range(window_size)]
    other_hours = [h for h in hours if h not in lowest_hours]

    avg_lowest = df_hourly_pd[df_hourly_pd["hour"].isin(lowest_hours)][
        "requests"
    ].mean()
    avg_other = df_hourly_pd[df_hourly_pd["hour"].isin(other_hours)]["requests"].mean()

    return lowest_hours, avg_lowest, avg_other


def save_load_over_time_data(dataset: str) -> None:
    df_requests = pd.read_csv(
        f"results/request_patterns/average_requests_per_hour/csv/{dataset}.csv"
    )
    df_durations = pd.read_csv(
        f"results/latency_over_time/average_latency_per_hour/csv/{dataset}.csv"
    )

    df = pd.merge(df_requests, df_durations, on="hour")
    df["duration"] = df["duration"] / 1000
    df["load"] = df["requests"] * df["duration"]
    df = df[["hour", "load"]]

    to_csv_safe(df, f"results/request_patterns/load_over_time/csv/{dataset}.csv")


def plot_load_over_time(datasets: list[str]) -> None:
    for dataset in datasets:
        plt.figure()
        df = pd.read_csv(f"results/request_patterns/load_over_time/csv/{dataset}.csv")
        avg_load = df["load"].mean()
        df["normalized_load"] = df["load"] / avg_load

        label = DATASETS.get(dataset, dataset.replace("_", " ").title().replace("Ibm", "IBM"))
        plt.plot(df["hour"], df["normalized_load"], label=label)

        plt.xlabel("Hour of Day")
        plt.xticks([0, 6, 12, 18, 24])
        plt.ylabel("Normalized Load")
        set_nice_yticks(plt.gca())
        savefig_safe(f"results/request_patterns/load_over_time/{dataset}.pdf")


def save_iat_cdf(df: dd.DataFrame, dataset: str) -> None:
    if dataset == "ibm26":
        df["func_id"] = df["func_id"].astype(str)

    df = df.set_index("func_id")
    bins = np.logspace(np.log10(0.001), np.log10(1000), 1000)

    def compute_histogram_partition(pdf: pd.DataFrame) -> pd.Series:
        if pdf.empty:
            return pd.Series(0, index=bins[:-1])


        if dataset == "ibm26":
            pdf.index = pdf.index.astype("category")

        pdf = pdf.sort_values("start_timestamp")
        pdf_reset = pdf.reset_index()

        prev_func = pdf_reset["func_id"].shift(1)
        prev_time = pdf_reset["start_timestamp"].shift(1)
        mask = pdf_reset["func_id"] == prev_func

        inter_iats = pdf_reset.loc[mask, "start_timestamp"] - prev_time.loc[mask]
        inter_iats = inter_iats[inter_iats > 0]

        hist_counts, _ = np.histogram(inter_iats, bins=bins)

        # IATs within one-second buckets are modeled a Poisson process assumption
        multi = pdf_reset[pdf_reset["requests"] > 1]
        if not multi.empty:
            rng = np.random.default_rng(seed=0)
            req_counts = multi["requests"].value_counts()

            for n_req, count_of_rows in req_counts.items():
                if n_req <= 1:
                    continue
                total_samples = (int(n_req) - 1) * count_of_rows
                scale = 1.0 / n_req
                vals = rng.exponential(scale=scale, size=total_samples)
                c, _ = np.histogram(vals, bins=bins)
                hist_counts += c

        return pd.Series(hist_counts, index=bins[:-1])

    meta = pd.Series(0, index=bins[:-1], dtype="int64")

    res = df.map_partitions(compute_histogram_partition, meta=meta).compute()

    if hasattr(res, "columns") and not isinstance(res, pd.Series):
        total_counts = res.sum(axis=1)
    else:
        total_counts = res.groupby(level=0).sum()

    cdf = np.cumsum(total_counts.values)
    if len(cdf) > 0 and cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    cdf_df = pd.DataFrame({"iat": bins[:-1], "cdf": cdf})
    to_csv_safe(cdf_df, f"results/request_patterns/iat_cdf/csv/{dataset}.csv")

def plot_iat_cdf(datasets: list[str]) -> None:
    plt.figure()

    for dataset in datasets:
        df = pd.read_csv(f"results/request_patterns/iat_cdf/csv/{dataset}.csv")
        label = DATASETS.get(dataset, dataset.replace("_", " ").title().replace("Ibm", "IBM"))
        style = get_dataset_style(dataset)
        plt.plot(df["iat"], df["cdf"] * 100, label=label, **style)

    plt.xscale("log")
    plt.xlabel("Keep-Alive Window (s)")
    plt.ylabel("Warm Starts (\\%)")
    plt.xticks([0.1, 1, 10, 100, 1000])
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:g}$"))
    plt.xlim(0.1, 1000)
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylim(0, 100)
    plt.legend(loc="lower right", fontsize=18, bbox_to_anchor=(1.05, -0.05))
    savefig_safe(f"results/request_patterns/iat_cdf/iat_cdf.pdf")


def save_function_popularity_cdf(df: dd.DataFrame, dataset: str) -> None:
    func_counts = df.groupby("func_id")["requests"].sum().compute()
    func_counts_sorted = func_counts.sort_values()

    total_invocations = func_counts_sorted.sum()
    cumulative_invocations = np.cumsum(func_counts_sorted.values)

    num_functions = len(func_counts_sorted)
    func_percentage = np.arange(1, num_functions + 1) / num_functions * 100

    request_percentage = cumulative_invocations / total_invocations * 100

    cdf_df = pd.DataFrame(
        {
            "function_percentage": func_percentage,
            "request_percentage": request_percentage,
        }
    )
    to_csv_safe(
        cdf_df, f"results/request_patterns/function_popularity/csv/{dataset}.csv"
    )


def plot_function_popularity_cdf(datasets: list[str]) -> None:
    plt.figure()

    for dataset in datasets:
        df = pd.read_csv(
            f"results/request_patterns/function_popularity/csv/{dataset}.csv"
        )
        label = DATASETS.get(dataset, dataset.replace("_", " ").title().replace("Ibm", "IBM"))
        style = get_dataset_style(dataset)
        plt.plot(
            df["function_percentage"], df["request_percentage"], label=label, **style
        )

    x = [0, 100]
    y = [0, 100]
    plt.plot(x, y, "k--")
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    angle = np.rad2deg(np.arctan2(dy, dx))
    plt.text(
        50,
        50 - 10,
        "Uniform Distribution",
        ha="center",
        va="center",
        transform_rotates_text=True,
        rotation=angle,
    )
    plt.xlabel("Functions (\\%)")
    plt.ylabel("Requests (\\%)")
    plt.xticks([0, 25, 50, 75, 100])
    plt.yticks([0, 25, 50, 75, 100])
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc="upper left", fontsize=18, bbox_to_anchor=(0, 1.1))
    savefig_safe(
        f"results/request_patterns/function_popularity/function_popularity_cdf.pdf"
    )