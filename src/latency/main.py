import logging
import os

import pandas as pd

from src.latency.analysis import (
    compute_cdf,
    plot_latency_cdf,
    plot_latency_boxplot,
    compute_boxplot,
    compute_infra_cdf,
    plot_infra_cdf,
    plot_latency_violin,
    compute_violin,
)
from src.latency.data_loaders import read_data
from src.latency.transformers import latency_transformer
from src.util import (
    calc_stats,
    setup_logging,
    dask_client,
    log_timing,
    to_csv_safe_dask,
)

setup_logging()

EXECUTION_TIME_DATASETS = [
    "azure19",
    "azure21",
    "globus24",
    "huawei23",
    "huawei25",
    "ibm26",
]

EXECUTION_TIME_FUNC_DATASETS = [
    "azure19_func",
    "azure21_func",
    "globus24_func",
    "huawei23_func",
    "huawei25_func",
    "ibm26_func",
]

RESPONSE_TIME_DATASETS = [
    "globus24_total",
    "huawei23_total",
    "huawei25_total",
    "ibm26_total",
]

OVERHEAD_DATASETS = [
    "globus24_overhead",
    "huawei23_overhead",
    "huawei25_overhead",
    "ibm26_overhead",
]

INFRA_DATASETS = [
    "globus24_infra",
]

RUNTIME_DATASETS = [
    "huawei25_runtime"
]


def process_metric(datasets, metric_name):
    for dataset in datasets:
        logging.info(f"Loading dataset: {dataset}")
        df = read_data(dataset, latency_transformer(dataset))

        df = df.persist()

        to_csv_safe_dask(df, f"results/latency/csv/{dataset}_durations.csv")

        logging.info(f"Processing CDF and Boxplot for dataset: {dataset}")
        compute_cdf(dataset, df)
        compute_boxplot(dataset, df)


def process_infra_metric(datasets):
    for dataset in datasets:
        logging.info(f"Loading dataset for infra impact: {dataset}")
        df = read_data(dataset, latency_transformer(dataset))
        df = df.persist()
        to_csv_safe_dask(df, f"results/latency/csv/{dataset}_infra_durations.csv")
        logging.info(f"Processing infra CDF for dataset: {dataset}")
        compute_infra_cdf(dataset, df)


def process_violin_metric(datasets):
    for dataset in datasets:
        logging.info(f"Loading dataset for violin plot: {dataset}")
        df = read_data(dataset, latency_transformer(dataset))
        logging.info(f"Processing violin data for dataset: {dataset}")
        compute_violin(dataset, df)


@log_timing
@dask_client
def main():
    process_metric(EXECUTION_TIME_DATASETS, "execution_time")
    process_metric(EXECUTION_TIME_FUNC_DATASETS, "execution_time_by_function")
    process_metric(RESPONSE_TIME_DATASETS, "response_time")
    process_metric(OVERHEAD_DATASETS, "system_overhead")
    process_infra_metric(INFRA_DATASETS)
    process_violin_metric(RUNTIME_DATASETS)


def generate_stats():
    logging.info("Generating statistics for latency...")

    def calculate_duration_stats(datasets, metric_name):
        stats_data = {}
        for dataset in datasets:
            df = pd.read_csv(f"results/latency/csv/{dataset}_durations.csv")
            if "count" in df.columns:
                grouped = df.groupby("duration")["count"].sum().reset_index()
                stats_data[dataset] = (
                    grouped["duration"].to_numpy(),
                    grouped["count"].to_numpy(),
                )
            else:
                stats_data[dataset] = df["duration"].to_numpy()
        if stats_data:
            calc_stats(stats_data, f"results/latency/stats/{metric_name}_stats.csv")

    calculate_duration_stats(EXECUTION_TIME_DATASETS, "execution_time")
    calculate_duration_stats(EXECUTION_TIME_FUNC_DATASETS, "execution_time_by_function")
    calculate_duration_stats(RESPONSE_TIME_DATASETS, "response_time")
    calculate_duration_stats(OVERHEAD_DATASETS, "system_overhead")

    infra_stats = {}
    csv_dir = "results/latency/csv"
    if os.path.exists(csv_dir):
        for dataset in INFRA_DATASETS:
            files = [
                f for f in os.listdir(csv_dir)
                if f.startswith(f"{dataset}_") and f.endswith("_cdf.csv")
            ]
            for f in files:
                infra = f.replace(f"{dataset}_", "").replace("_cdf.csv", "")
                df = pd.read_csv(os.path.join(csv_dir, f))
                if "count" in df.columns:
                    infra_stats[f"{dataset}_{infra}"] = (df["duration"].to_numpy(), df["count"].to_numpy())
                else:
                    infra_stats[f"{dataset}_{infra}"] = df["duration"].to_numpy()

    if infra_stats:
        calc_stats(infra_stats, "results/latency/stats/infra_execution_time_stats.csv")

    runtime_stats = {}
    for dataset in RUNTIME_DATASETS:
        df = pd.read_csv(f"results/latency/csv/{dataset}_violin.csv")
        if "count" in df.columns:
            grouped = df.groupby(["infra", "duration"])["count"].sum().reset_index()
            for runtime, group in grouped.groupby("infra"):
                runtime_stats[f"{dataset}_{runtime}"] = (group["duration"].to_numpy(), group["count"].to_numpy())
        else:
            for runtime, group in df.groupby("infra"):
                runtime_stats[f"{dataset}_{runtime}"] = group["duration"].to_numpy()
    if runtime_stats:
        calc_stats(runtime_stats, "results/latency/stats/runtime_stats.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    for datasets in (
            EXECUTION_TIME_DATASETS,
            EXECUTION_TIME_FUNC_DATASETS,
            RESPONSE_TIME_DATASETS,
            OVERHEAD_DATASETS,
    ):
        if datasets:
            plot_latency_cdf(datasets)
            plot_latency_boxplot(datasets)

    if INFRA_DATASETS:
        for dataset in INFRA_DATASETS:
            plot_infra_cdf(dataset)

    if RUNTIME_DATASETS:
        plot_latency_violin(RUNTIME_DATASETS)
