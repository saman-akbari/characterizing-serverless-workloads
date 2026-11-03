import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd

from src.util import (
    savefig_safe,
    to_csv_safe,
    group_by_hour,
    set_nice_yticks,
)

plt.style.use("src/scientific.mplstyle")


def average_latency_per_hour_csv(df: dd.DataFrame, dataset: str) -> None:
    # per hour using sum (df has total_latency and requests)
    df_hourly_base = group_by_hour(df, "sum")
    df_hourly_base["hour"] = (df_hourly_base["time"] % 24).astype(int)  # 0-23
    df_hourly_base["day_of_week"] = ((df_hourly_base["time"] // 24) % 7).astype(int)

    def weighted_agg(d):
        agg = d.groupby("hour")[["total_latency", "requests"]].sum().reset_index()
        agg["duration"] = agg["total_latency"] / agg["requests"]
        return agg

    # overall
    df_all_days_agg = weighted_agg(df_hourly_base)

    # weekdays (0-4)
    df_weekdays = df_hourly_base[df_hourly_base["day_of_week"] < 5]
    df_weekdays_agg = weighted_agg(df_weekdays)

    # weekends (5-6)
    df_weekends = df_hourly_base[df_hourly_base["day_of_week"] >= 5]
    df_weekends_agg = weighted_agg(df_weekends)

    df_all_pd, df_weekdays_pd, df_weekends_pd = dd.compute(
        df_all_days_agg, df_weekdays_agg, df_weekends_agg
    )

    to_csv_safe(
        df_all_pd[["hour", "duration"]],
        f"results/latency_over_time/average_latency_per_hour/csv/{dataset}.csv",
    )
    to_csv_safe(
        df_weekdays_pd[["hour", "duration"]],
        f"results/latency_over_time/average_latency_per_hour/csv/{dataset}_weekdays.csv",
    )
    to_csv_safe(
        df_weekends_pd[["hour", "duration"]],
        f"results/latency_over_time/average_latency_per_hour/csv/{dataset}_weekends.csv",
    )


def plot_duration_per_hour(dataset: str) -> None:
    df_hourly_pd = pd.read_csv(
        f"results/latency_over_time/average_latency_per_hour/csv/{dataset}.csv"
    )
    df_hourly_pd["duration"] = df_hourly_pd["duration"] / 1000.0

    plt.figure()
    plt.bar(df_hourly_pd["hour"], df_hourly_pd["duration"])
    plt.xlabel("Hour of Day")
    metric = "Response Time" if "total" in dataset else "Execution Time"
    plt.ylabel(f"{metric} (s)")

    plt.xticks([0, 6, 12, 18, 24])

    set_nice_yticks(plt.gca(), df_hourly_pd["duration"].max())

    savefig_safe(f"results/latency_over_time/average_latency_per_hour/{dataset}.pdf")


def plot_duration_per_hour_on_ax(
        ax, df_hourly: pd.DataFrame, max_y: float, title: str = ""
):
    ax.bar(df_hourly["hour"], df_hourly["duration"])
    ax.set_xticks([0, 6, 12, 18, 24])
    set_nice_yticks(ax, max_y)
    ax.set_title(title)


def plot_duration_per_hour_all_subplots(dataset: str):
    suffixes = ["", "_weekdays", "_weekends"]
    titles = [
        f"Overall",
        f"Weekdays",
        f"Weekends",
    ]

    dataframes = []
    global_max_duration = 0

    for suffix in suffixes:
        df = pd.read_csv(
            f"results/latency_over_time/average_latency_per_hour/csv/{dataset}{suffix}.csv"
        )
        if not df.empty:
            df["duration"] = df["duration"] / 1000.0
            global_max_duration = max(global_max_duration, df["duration"].max())
        dataframes.append(df)

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 8))
    fig.supxlabel("Hour of Day", y=-0.07)
    metric = "Response Time" if "total" in dataset else "Execution Time"
    fig.supylabel(f"{metric} (s)", x=-0.1)

    for i, df in enumerate(dataframes):
        if not df.empty:
            plot_duration_per_hour_on_ax(
                ax=axes[i],
                df_hourly=df,
                max_y=global_max_duration,
                title=titles[i],
            )

    savefig_safe(
        f"results/latency_over_time/average_latency_per_hour/{dataset}_all.pdf"
    )
