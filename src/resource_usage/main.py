import logging
import os

import pandas as pd

from src.resource_usage.analysis import (
    plot_memory_limit_cdf,
    plot_memory_cpu_usage,
    plot_memory_cpu_usage_density,
    plot_cpu_limit_cdf,
)
from src.resource_usage.data_loaders import data_loaders
from src.resource_usage.transformers import resource_usage_transformer
from src.util import (
    log_timing,
    calc_stats,
    setup_logging,
    to_csv_safe_dask,
    dask_client,
)

setup_logging()

MEMORY_LIMIT_DATASETS = [
    "alibaba21_memory_limit",
    "huawei23_memory_limit",
    "huawei25_memory_limit",
    "ibm26_memory_limit",
]

MEMORY_USAGE_DATASETS = [
    "azure19_memory_usage",
    "huawei23_memory_usage",
    "huawei25_memory_usage",
]

CPU_LIMIT_DATASETS = [
    "huawei23_cpu_limit",
    "huawei25_cpu_limit",
    "ibm26_cpu_limit"
]

CPU_USAGE_DATASETS = [
    "huawei23_cpu_usage",
    "huawei25_cpu_usage",
]

DATASETS = list(
    set(
        MEMORY_LIMIT_DATASETS
        + MEMORY_USAGE_DATASETS
        + CPU_LIMIT_DATASETS
        + CPU_USAGE_DATASETS
    )
)


@log_timing
@dask_client
def main():
    logging.info("Starting to load and process data...")

    for dataset in DATASETS:
        logging.info(f"Processing: {dataset}")
        df = data_loaders(dataset)()
        df = resource_usage_transformer(dataset)(df)
        to_csv_safe_dask(df, f"results/resource_usage/csv/{dataset}.csv")


def generate_stats():
    logging.info("Generating statistics for resource usage...")
    stats_data = {}
    for dataset in DATASETS:
        if dataset == "huawei23_memory_limit":
            mem_limit = "results/resource_usage/csv/huawei23_memory_limit.csv"
            mem_usage = "results/resource_usage/csv/huawei23_memory_usage.csv"
            if os.path.exists(mem_limit) and os.path.exists(mem_usage):
                df_limit = pd.read_csv(mem_limit)
                df_usage = pd.read_csv(mem_usage)
                merged = pd.merge(df_limit, df_usage, on="pool", how="inner")
                stats_data["huawei23_memory_usage_mb"] = (
                        merged["memory_limit_mb"] * merged["memory_usage_fraction"]).to_numpy()
        elif dataset == "huawei25_memory_usage":
            path = "results/resource_usage/csv/huawei25_memory_usage.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                stats_data["huawei25_memory_usage_mb"] = df["memory_usage_mb"].to_numpy()
        elif dataset == "azure19_memory_usage":
            path = "results/resource_usage/csv/azure19_memory_usage.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                stats_data["azure19_memory_usage_mb"] = df["memory_usage_mb"].to_numpy()
    if stats_data:
        calc_stats(stats_data, "results/resource_usage/stats/memory_usage_mb_stats.csv")

    limit_stats = {}
    for dataset in MEMORY_LIMIT_DATASETS:
        path = f"results/resource_usage/csv/{dataset}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            limit_stats[dataset] = df["memory_limit_mb"].to_numpy()
    if limit_stats:
        calc_stats(limit_stats, "results/resource_usage/stats/memory_limit_stats.csv")

    cpu_limit_stats = {}
    for dataset in CPU_LIMIT_DATASETS:
        path = f"results/resource_usage/csv/{dataset}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            cpu_limit_stats[dataset] = df["cpu_limit_cores"].to_numpy()
    if cpu_limit_stats:
        calc_stats(cpu_limit_stats, "results/resource_usage/stats/cpu_limit_stats.csv")

    cpu_usage_stats = {}
    for dataset in CPU_USAGE_DATASETS:
        path = f"results/resource_usage/csv/{dataset}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "cpu_usage_cores" in df.columns:
                cpu_usage_stats[dataset] = df["cpu_usage_cores"].to_numpy()
            elif "cpu_usage_fraction" in df.columns:
                cpu_usage_stats[dataset] = df["cpu_usage_fraction"].to_numpy()
    if cpu_usage_stats:
        calc_stats(cpu_usage_stats, "results/resource_usage/stats/cpu_usage_stats.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    logging.info(f"Plotting memory limit CDF...")
    plot_memory_limit_cdf([d.split("_")[0] for d in MEMORY_LIMIT_DATASETS])

    logging.info(f"Plotting CPU limit CDF...")
    plot_cpu_limit_cdf([d.split("_")[0] for d in CPU_LIMIT_DATASETS])

    logging.info("Plotting memory and CPU usage...")
    for base in ["huawei23", "huawei25"]:
        plot_memory_cpu_usage(base)
        plot_memory_cpu_usage_density(base)
