import os

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd

from src.util import savefig_safe, to_csv_safe, get_dataset_style

plt.style.use("src/scientific.mplstyle")

DATASET_LABELS = {
    "huawei23": "Huawei23",
    "huawei25": "Huawei25",
    "huawei25_lifespan": "Huawei25",
    "ibm26": "IBM26",
    "ibm26_lifespan": "IBM26",
}


def compute_instance_metrics_cdf(df: dd.DataFrame, dataset: str) -> None:
    df_pd = df.compute()
    metrics = ["max_instances", "scale_up_rate", "scale_down_rate"]

    for metric in metrics:
        df_sorted = df_pd.sort_values(metric).reset_index(drop=True)
        df_sorted["cdf"] = (df_sorted.index + 1) / len(df_sorted)

        output_path = os.path.join(
            "results/instances/csv", f"{dataset}_{metric}_cdf.csv"
        )
        to_csv_safe(df_sorted[[metric, "cdf"]], output_path)


def compute_lifespan_cdf(df: dd.DataFrame, dataset: str) -> None:
    df_pd = df.compute()

    df_sorted = df_pd.sort_values("lifespan").reset_index(drop=True)
    df_sorted["cdf"] = (df_sorted.index + 1) / len(df_sorted)

    output_path = os.path.join("results/instances/csv", f"{dataset}_lifespan_cdf.csv")
    to_csv_safe(df_sorted[["lifespan", "cdf"]], output_path)


def compute_config_cdf(df: dd.DataFrame, dataset: str, col_name: str) -> None:
    df_pd = df.compute()
    df_sorted = df_pd.sort_values(col_name).reset_index(drop=True)
    df_sorted["cdf"] = (df_sorted.index + 1) / len(df_sorted)

    output_path = os.path.join("results/instances/csv", f"{dataset}_cdf.csv")
    to_csv_safe(df_sorted[[col_name, "cdf"]], output_path)


def plot_metric_cdf(datasets: list[str], metric: str, xlabel: str) -> None:
    plt.figure()

    for dataset in datasets:
        file_path = f"results/instances/csv/{dataset}_{metric}_cdf.csv"
        df = pd.read_csv(file_path)
        label = DATASET_LABELS.get(dataset, dataset)
        style = get_dataset_style(dataset)
        plt.step(df[metric], df["cdf"], where="post", label=label, **style)

    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Functions CDF")
    plt.xticks(
        [1, 10, 100, 1000, 10000], labels=["$1$", "$10$", "$100$", "$1000$", "$10000$"]
    )
    plt.xlim(1, 10000)
    plt.ylim(0, 1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.legend(loc="lower right")

    savefig_safe(f"results/instances/{metric}_cdf.pdf")


def plot_lifespan_cdf(datasets: list[str]) -> None:
    plt.figure()

    for dataset in datasets:
        file_path = f"results/instances/csv/{dataset}_lifespan_cdf.csv"
        df = pd.read_csv(file_path)
        label = DATASET_LABELS.get(dataset, dataset)
        style = get_dataset_style(dataset)
        plt.step(df["lifespan"], df["cdf"], where="post", label=label, **style)

    plt.xscale("log")
    plt.xlabel("Active Instance Lifespan (s)")
    plt.ylabel("Instances CDF")
    plt.ylim(0, 1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.xlim(1, 10 * 604800)

    week = 604800
    plt.xticks([1, 60, 3600, 86400, week, 10 * week])
    plt.gca().set_xticklabels(["1s", "1m", "1h", "1d", "1w", "10w"])

    plt.legend(loc="lower right")
    savefig_safe(f"results/instances/lifespan_cdf.pdf")


def plot_concurrency_limit_cdf(dataset: str) -> None:
    file_path = f"results/instances/csv/{dataset}_cdf.csv"
    df = pd.read_csv(file_path)

    plt.figure()
    plt.step(
        df["concurrency_limit"], df["cdf"], where="post", color="#9467bd"
    )
    plt.text(115, 0.2, "Default (100)", rotation=90, va="bottom")

    plt.xlabel("Instance Concurrency Limit")
    plt.ylabel("Functions CDF")
    plt.xlim(0, 1000)
    plt.ylim(0, 1)
    plt.xticks([0, 250, 500, 750, 1000])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])

    savefig_safe(f"results/instances/{dataset}_cdf.pdf")


def plot_min_scale_cdf(dataset: str) -> None:
    file_path = f"results/instances/csv/{dataset}_cdf.csv"
    df = pd.read_csv(file_path)

    plt.figure()
    plt.step(df["min_scale"], df["cdf"], where="post", color="#8c564b")

    plt.xlabel("Minimum Instance Scale")
    plt.ylabel("Functions CDF")
    plt.xlim(0, 15)
    plt.ylim(0, 1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])

    savefig_safe(f"results/instances/{dataset}_cdf.pdf")
