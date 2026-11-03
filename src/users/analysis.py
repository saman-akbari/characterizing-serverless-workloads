import logging
import os
from typing import List

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from src.util import to_csv_safe, savefig_safe, calc_stats, get_dataset_style

plt.style.use("src/scientific.mplstyle")
logging.basicConfig(level=logging.INFO)

DAYS_PER_DATASET = {
    "azure19": 14,
    "azure20": 14,
    "globus24": 217,
    "huawei25": 31,
    "ibm26": 66,  # dataset covers 66 full days, the original paper only analyzes 62 days
}


def _save_cdf_data(series: pd.Series, output_path: str):
    df = series.to_frame()
    df.columns = ["value"]
    df = df.sort_values("value").reset_index(drop=True)
    df["cdf"] = (df.index + 1) / len(df)
    to_csv_safe(df, output_path)


def users_functions_to_csv(df: dd.DataFrame, dataset_name: str):
    logging.info(f"Processing functions-per-user for {dataset_name}...")
    num_users = df["user_id"].nunique().compute()
    logging.info(f"[{dataset_name}] Number of unique users: {num_users}")

    functions_per_user = df.groupby("user_id")["function_id"].nunique().compute()

    _save_cdf_data(
        functions_per_user, f"results/users/functions_per_user/csv/{dataset_name}.csv"
    )


def users_requests_to_csv(df: dd.DataFrame, dataset_name: str):
    logging.info(f"Processing requests-per-user for {dataset_name}...")
    num_users = df["user_id"].nunique().compute()
    logging.info(f"[{dataset_name}] Number of unique users: {num_users}")

    # divide by number of days in the dataset
    requests_per_user = (
            df.groupby("user_id")["requests"].sum().compute()
            / DAYS_PER_DATASET.get(dataset_name, 1)
    )

    _save_cdf_data(
        requests_per_user, f"results/users/requests_per_user/csv/{dataset_name}.csv"
    )


def calculate_users_stats(datasets: List[str]):
    stats_functions = {}
    stats_requests = {}

    for dataset in datasets:
        df_func = pd.read_csv(f"results/users/functions_per_user/csv/{dataset}.csv")
        stats_functions[dataset] = df_func["value"].to_numpy()

        df_req = pd.read_csv(f"results/users/requests_per_user/csv/{dataset}.csv")
        stats_requests[dataset] = df_req["value"].to_numpy()

    calc_stats(stats_functions, "results/users/stats/functions_per_user_stats.csv")
    calc_stats(stats_requests, "results/users/stats/requests_per_user_stats.csv")


def plot_functions_per_user_cdf(functions_datasets: List[str]):
    plot_cdf(
        datasets=functions_datasets,
        input_dir="results/users/functions_per_user/csv",
        output_path="results/users/functions_per_user/functions_per_user_cdf.pdf",
        xlabel="Functions per User",
        ylabel="Users CDF",
        xticks=[1, 10, 100, 1000, 10000, 100000],
        xlim_left=1
    )


def plot_requests_per_user_cdf(requests_datasets: List[str]):
    plot_cdf(
        datasets=requests_datasets,
        input_dir="results/users/requests_per_user/csv",
        output_path="results/users/requests_per_user/requests_per_user_cdf.pdf",
        xlabel="Daily Requests per User",
        ylabel="Users CDF",
        xticks=[0, 10, 1000, 100000, 10000000, 1000000000],
        xlim_left=0
    )


def plot_cdf(
        datasets: List[str],
        input_dir: str,
        output_path: str,
        xlabel: str,
        ylabel: str,
        xticks: list[int],
        xlim_left: int
):
    logging.info(f"Plotting CDF for: {xlabel}")
    plt.figure()
    ax = plt.gca()

    for dataset in datasets:
        df = pd.read_csv(os.path.join(input_dir, f"{dataset}.csv"))
        style = get_dataset_style(dataset)
        plt.step(df["value"], df["cdf"], where="post", label=dataset.title().replace("Ibm", "IBM"), **style)

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(xticks)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(0, 1)
    ax.set_xlim(left=xlim_left)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="lower right", bbox_to_anchor=(1.05, -0.08), handlelength=1.8, labelspacing=0.08, handletextpad=0.5)

    def custom_format(x, pos):
        if x <= 10:
            return f"${int(x)}$"
        return f"$10^{{{int(np.log10(x))}}}$"

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_format))

    savefig_safe(output_path)
