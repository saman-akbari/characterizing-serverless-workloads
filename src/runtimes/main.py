import logging

import pandas as pd

from src.runtimes.analysis import (
    compute_runtime_distribution,
    plot_runtime_distribution,
)
from src.runtimes.data_loaders import read_data
from src.runtimes.transformers import runtimes_transformer
from src.util import log_timing, setup_logging, dask_client, calc_stats

setup_logging()

DATASETS = [
    "alibaba21",
    "huawei25",
]


@log_timing
@dask_client
def main():
    logging.info("Starting to load and process runtime data...")

    for dataset in DATASETS:
        logging.info(f"Processing dataset: {dataset}")
        df = read_data(dataset, runtimes_transformer(dataset))
        compute_runtime_distribution(df, dataset)


def generate_stats():
    logging.info("Generating statistics for runtimes...")
    stats_data = {}
    for dataset in DATASETS:
        df = pd.read_csv(f"results/runtimes/csv/{dataset}.csv")
        stats_data[dataset] = df["count"].to_numpy()
    calc_stats(stats_data, "results/runtimes/stats/runtime_counts_stats.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    logging.info("Plotting runtime distributions...")
    plot_runtime_distribution(DATASETS)
