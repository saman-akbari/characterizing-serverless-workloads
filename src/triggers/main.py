import logging

import pandas as pd

from src.triggers.analysis import (
    compute_trigger_distribution,
    plot_trigger_distribution,
)
from src.triggers.data_loaders import read_data
from src.triggers.transformers import triggers_transformer
from src.util import log_timing, setup_logging, dask_client, calc_stats

setup_logging()

DATASETS = [
    "azure19",
    "huawei25",
]


@log_timing
@dask_client
def main():
    logging.info("Starting to load and process trigger data...")

    for dataset in DATASETS:
        logging.info(f"Processing dataset: {dataset}")
        df = read_data(dataset, triggers_transformer(dataset))
        compute_trigger_distribution(df, dataset)


def generate_stats():
    logging.info("Generating statistics for triggers...")
    stats_data = {}
    for dataset in DATASETS:
        df = pd.read_csv(f"results/triggers/csv/{dataset}.csv")
        stats_data[dataset] = df["count"].to_numpy()
    calc_stats(stats_data, "results/triggers/stats/trigger_counts_stats.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    logging.info("Plotting trigger distributions...")
    plot_trigger_distribution(DATASETS)
